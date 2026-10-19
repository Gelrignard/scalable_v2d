"""G1/Dex3 manipulation physics with an explicitly fixed lower body.

Root and leg transforms are frozen in the model, not stabilized by forces.
The arms, hands, optional waist, and free book remain dynamic. This is a
supported manipulation bench, not a locomotion controller.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from .robot_scene import RobotScene, numbers


class SupportedRobotScene(RobotScene):
    def __init__(self, kin, bounds, support_path, fixed_q, *, mass=1., friction=.6, dt=.001, move_waist=False):
        super().__init__(kin, bounds, support_path, mass=mass, friction=friction)
        self.fixed_q = np.array(fixed_q).copy()
        self.move_waist = move_waist
        self.movable = np.array([i for i, n in enumerate(kin.joint_names)
                                 if any(s in n for s in ('shoulder', 'elbow', 'wrist', 'hand')) or (move_waist and 'waist' in n)])
        xml = ET.fromstring(self.xml)
        world = xml.find('worldbody')
        pelvis = world.find('body')
        pelvis.remove(pelvis.find('freejoint'))
        pelvis.set('pos', numbers(fixed_q[:3]))
        pelvis.set('quat', numbers(fixed_q[[6, 3, 4, 5]]))
        for body in world.iter('body'):
            for joint in list(body.findall('joint')):
                name = joint.get('name')
                index = kin.joint_names.index(name)
                if index in self.movable:
                    joint.set('damping', '.02')
                    joint.set('armature', '.00149' if 'hand' in name else ('.00425' if 'wrist' in name else '.003609725'))
                    continue
                axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
                pivot = np.fromstring(joint.get('pos', '0 0 0'), sep=' ')
                rotation = Rotation.from_quat(np.fromstring(body.get('quat', '1 0 0 0'), sep=' '), scalar_first=True)
                change = Rotation.from_rotvec(axis*fixed_q[index+7])
                position = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
                body.set('pos', numbers(position+rotation.apply(pivot-change.apply(pivot))))
                body.set('quat', numbers((rotation*change).as_quat(scalar_first=True)))
                body.remove(joint)
        xml.find('option').set('timestep', str(dt))
        for geom in world.iter('geom'):
            geom.set('margin', '.0005')
            geom.set('solref', '.003 1')
            geom.set('solimp', '.95 .99 .001')
            geom.set('condim', '3')
            geom.set('friction', f'{friction} .00001 .00001')
        # Add the same manufacturer fingertip frames used by Pinocchio.
        from v2d_retarget.core.contacts import TIP_NAMES
        self.tip_names = TIP_NAMES
        for side in ('left', 'right'):
            hand = ET.parse(kin.config.urdf_path.parent/f'dex3_{side}.urdf').getroot()
            for joint in hand.findall('joint'):
                name = joint.find('child').get('link')
                if name in TIP_NAMES:
                    parent = joint.find('parent').get('link')
                    body = next(b for b in world.iter('body') if b.get('name') == parent)
                    ET.SubElement(body, 'site', name=name, pos=joint.find('origin').get('xyz'), size='.002')
        self.xml = ET.tostring(xml, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.full(len(kin.joint_names), -1, int)
        self.vadr = np.full(len(kin.joint_names), -1, int)
        for i in self.movable:
            joint = self.model.joint(kin.joint_names[i])
            self.qadr[i], self.vadr[i] = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.active_q = self.qadr[self.movable]
        self.active_v = self.vadr[self.movable]
        self.book_q = int(self.model.joint('book_root').qposadr[0])
        self.book_v = int(self.model.joint('book_root').dofadr[0])
        self.book_body = self.model.body('book').id
        self.book_geom = self.model.geom('book_geom').id
        self.floor_geom, self.table_geom = self.model.geom('floor').id, self.model.geom('table').id
        self.environment = {self.floor_geom, self.table_geom}
        self.tip_ids = np.array([self.model.site(n).id for n in TIP_NAMES])
        self.hand = np.array(['hand' in kin.joint_names[i] for i in self.movable])
        self.caps = np.minimum(kin.robot.model.effortLimit[6+self.movable], np.where(self.hand, .76, np.inf))

    def set_state(self, q, object_position, object_wxyz):
        self.data.qpos[self.active_q] = q[7+self.movable]
        self.data.qpos[self.book_q:self.book_q+3] = object_position
        self.data.qpos[self.book_q+3:self.book_q+7] = object_wxyz
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def robot_q(self):
        q = self.fixed_q.copy()
        q[7+self.movable] = self.data.qpos[self.active_q]
        return q

    def tips_and_jacobian(self):
        jacobians = []
        for site in self.tip_ids:
            j = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, j, None, int(site))
            jacobians.append(j[:, self.active_v])
        return self.data.site_xpos[self.tip_ids].copy(), np.asarray(jacobians)

    def rollout(self, q, positions, quaternions, times, *, force_ff=None, kp_arm=120., kp_hand=5., initial_q=None, initial_velocity=None, inertia_ff=False):
        relative = times-times[0]
        spline = CubicSpline(relative, q[:, 7+self.movable], axis=0)
        end = round(relative[-1]/self.model.opt.timestep)
        step_times = np.minimum(np.arange(end+1)*self.model.opt.timestep, relative[-1])
        commands, velocities = spline(step_times), spline(step_times, 1)
        acceleration = spline(step_times, 2)
        desired_p = np.column_stack([np.interp(step_times, relative, positions[:, j]) for j in range(3)])
        desired_r = Slerp(relative, Rotation.from_quat(quaternions, scalar_first=True))(step_times).as_quat(scalar_first=True)
        force = np.zeros((len(step_times), 6, 3)) if force_ff is None else CubicSpline(relative, force_ff, axis=0)(step_times)
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(q[0] if initial_q is None else initial_q, positions[0], quaternions[0])
        self.data.qvel[self.active_v] = velocities[0] if initial_velocity is None else initial_velocity
        self.data.qvel[self.book_v:self.book_v+3] = (positions[1]-positions[0])/(times[1]-times[0])
        rot = Rotation.from_quat(quaternions[:2], scalar_first=True)
        self.data.qvel[self.book_v+3:self.book_v+6] = (rot[0].inv()*rot[1]).as_rotvec()/(times[1]-times[0])
        mujoco.mj_forward(self.model, self.data)
        initial_pen = max([max(0., -c.dist) for c in self.data.contact], default=0.)
        kp = np.where(self.hand, kp_hand, kp_arm)
        kd = np.where(self.hand, .2, 2.)
        if inertia_ff:
            kd = np.where(self.hand, .4, 8.)
        sample_steps = np.round(relative/self.model.opt.timestep).astype(int)
        outq, outp, outr, outtip, outtau, outcontact = [], [], [], [], [], []
        outforces, outpoints, outvelocity = [], [], []
        max_p = max_r = max_pen = max_external = max_sat = 0.
        max_fraction = np.zeros(len(self.active_v))
        max_velocity = np.zeros(len(self.active_v))
        max_limit_violation = 0.
        saturated_steps = 0
        lower = self.kin.robot.model.lowerPositionLimit[7+self.movable]
        upper = self.kin.robot.model.upperPositionLimit[7+self.movable]
        pair, sample = '', 0
        for step, t in enumerate(step_times):
            max_velocity = np.maximum(max_velocity, np.abs(self.data.qvel[self.active_v]))
            angles = self.data.qpos[self.active_q]
            max_limit_violation = max(max_limit_violation, float(np.max(np.maximum(lower-angles, angles-upper))))
            max_p = max(max_p, float(np.linalg.norm(self.data.qpos[self.book_q:self.book_q+3]-desired_p[step])))
            angle = 2*np.arccos(np.clip(abs(self.data.qpos[self.book_q+3:self.book_q+7]@desired_r[step]), 0, 1))
            max_r = max(max_r, float(angle))
            for c in self.data.contact:
                if -c.dist > max_pen:
                    max_pen = float(-c.dist)
                    pair = self.model.geom(c.geom1).name+' / '+self.model.geom(c.geom2).name
            max_external = max(max_external, float(np.max(np.abs(self.data.qfrc_applied[self.book_v:self.book_v+6]))),
                               float(np.max(np.abs(self.data.xfrc_applied[self.book_body]))))
            if step == sample_steps[sample]:
                outq.append(self.robot_q())
                outp.append(self.data.qpos[self.book_q:self.book_q+3].copy())
                outr.append(self.data.qpos[self.book_q+3:self.book_q+7].copy())
                outtip.append(self.data.site_xpos[self.tip_ids].copy())
                outtau.append(self.data.qfrc_applied[self.active_v].copy())
                outvelocity.append(self.data.qvel[self.active_v].copy())
                counts = np.zeros(2, int)
                contact_forces, contact_points = np.zeros((2,3)), np.zeros((2,3))
                weights = np.zeros(2)
                for ci,c in enumerate(self.data.contact):
                    if self.book_geom not in (c.geom1,c.geom2): continue
                    other = c.geom2 if c.geom1 == self.book_geom else c.geom1
                    name = self.model.geom(other).name
                    if 'left_' not in name and 'right_' not in name: continue
                    hand = 0 if 'left_' in name else 1
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model,self.data,ci,wrench)
                    if np.linalg.norm(wrench[:3]) < 1e-5: continue
                    counts[hand] += 1
                    world_force = np.asarray(c.frame).reshape(3,3).T@wrench[:3]
                    if c.geom1 == self.book_geom: world_force *= -1
                    contact_forces[hand] += world_force
                    weight=np.linalg.norm(world_force)
                    contact_points[hand] += weight*c.pos
                    weights[hand] += weight
                contact_points /= np.maximum(weights[:,None],1e-12)
                outcontact.append(counts);outforces.append(contact_forces);outpoints.append(contact_points)
                sample += 1
                if sample == len(times): break
            raw = kp*(commands[step]-self.data.qpos[self.active_q])+kd*(velocities[step]-self.data.qvel[self.active_v])
            raw += self.data.qfrc_bias[self.active_v]
            if inertia_ff:
                generalized_acceleration = np.zeros(self.model.nv)
                generalized_acceleration[self.active_v] = acceleration[step]
                inertial = np.zeros(self.model.nv)
                mujoco.mj_mulM(self.model, self.data, inertial, generalized_acceleration)
                raw += inertial[self.active_v]
            if force_ff is not None:
                _, jac = self.tips_and_jacobian()
                raw += np.einsum('kij,ki->j', jac, force[step])
            max_sat = max(max_sat, float(np.max(np.abs(raw)/self.caps)))
            max_fraction = np.maximum(max_fraction, np.abs(raw)/self.caps)
            saturated_steps += int(np.any(np.abs(raw)>self.caps))
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.active_v] = np.clip(raw, -self.caps, self.caps)
            mujoco.mj_step(self.model, self.data)
        outp, outr = np.asarray(outp), np.asarray(outr)
        return dict(q=np.asarray(outq), object_positions=outp, object_wxyz=outr, fk_tip_positions=np.asarray(outtip),
                    command_q=q, timestamps=times, torques=np.asarray(outtau),
                    joint_velocities=np.asarray(outvelocity), contacts_per_hand=np.asarray(outcontact),
                    contact_forces_world=np.asarray(outforces), contact_points_world=np.asarray(outpoints),
                    position_error=np.linalg.norm(outp-positions, axis=1),
                    rotation_error=(Rotation.from_quat(outr, scalar_first=True).inv()*Rotation.from_quat(quaternions, scalar_first=True)).magnitude(),
                    max_position_error_m=max_p, max_rotation_error_rad=max_r, max_penetration_m=max_pen,
                    initial_max_penetration_m=initial_pen, worst_collision_pair=pair, max_torque_fraction_requested=max_sat,
                    torque_fraction_by_joint=max_fraction,
                    max_joint_velocity_rad_s=max_velocity, max_joint_limit_violation_rad=max_limit_violation,
                    saturation_steps=saturated_steps, torque_caps=self.caps, movable_joint_indices=self.movable,
                    max_object_applied_force=max_external, fixed_root_and_legs=True, dynamic_waist=self.move_waist)
