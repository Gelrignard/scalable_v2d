"""G1/Dex3 manipulation physics with an externally supported lower body.

Root and leg transforms are frozen in the model, not stabilized by forces.
The arms, hands, optional waist, and free book remain dynamic. This is a
supported manipulation bench, not a locomotion controller. An optional mocap
weld moves the pelvis along the reference while disabling frozen leg contacts.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp, RotationSpline
from .robot_scene import RobotScene, numbers
from .contact_measurement import object_contact_wrench, object_wrench_partition
from .torque_audit import TorquePeakAudit


class SupportedRobotScene(RobotScene):
    def __init__(self, kin, bounds, support_path, fixed_q, *, mass=1., friction=.6, dt=.001,
                 move_waist=False, moving_fixture=False, object_parts=None, floor_z=0.,object_inertial=None):
        super().__init__(kin, bounds, support_path, mass=mass, friction=friction, object_parts=object_parts, floor_z=floor_z,
                         object_inertial=object_inertial)
        self.moving_fixture = moving_fixture
        self.fixed_q = np.array(fixed_q).copy()
        self.move_waist = move_waist
        self.movable = np.array([i for i, n in enumerate(kin.joint_names)
                                 if any(s in n for s in ('shoulder', 'elbow', 'wrist', 'hand')) or (move_waist and 'waist' in n)])
        xml = ET.fromstring(self.xml)
        world = xml.find('worldbody')
        pelvis = world.find('body')
        if not moving_fixture:
            pelvis.remove(pelvis.find('freejoint'))
        else:
            ET.SubElement(world, 'body', name='external_pelvis_fixture', mocap='true',
                          pos=numbers(fixed_q[:3]), quat=numbers(fixed_q[[6,3,4,5]]))
            equality = ET.SubElement(xml, 'equality')
            ET.SubElement(equality, 'weld', name='pelvis_support', body1=pelvis.get('name'),
                          body2='external_pelvis_fixture', relpose='0 0 0 1 0 0 0',
                          solref='.002 1', solimp='.99 .999 .001')
            # Legs are a passive visual fixture. Their contacts must not turn
            # a prescribed walking path into an unintended balance experiment.
            for body in pelvis.findall('body'):
                if any(s in body.get('name', '') for s in ('hip', 'thigh')):
                    for geom in body.iter('geom'):
                        geom.set('contype', '0'); geom.set('conaffinity', '0')
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
        self.object_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.book_body))
        self.floor_geom = self.model.geom('floor').id
        self.table_geom = self.model.geom('table').id if self.table_radius else -1
        self.environment = {g for g in (self.floor_geom,self.table_geom) if g >= 0}
        if moving_fixture:
            self.root_q = int(self.model.joint('robot_root').qposadr[0])
            self.root_v = int(self.model.joint('robot_root').dofadr[0])
        self.tip_ids = np.array([self.model.site(n).id for n in TIP_NAMES])
        self.hand = np.array(['hand' in kin.joint_names[i] for i in self.movable])
        self.caps = np.minimum(kin.robot.model.effortLimit[6+self.movable], np.where(self.hand, .76, np.inf))

    def set_state(self, q, object_position, object_wxyz):
        if self.moving_fixture:
            self.data.qpos[self.root_q:self.root_q+7] = np.r_[q[:3],q[[6,3,4,5]]]
            self.data.mocap_pos[0] = q[:3]
            self.data.mocap_quat[0] = q[[6,3,4,5]]
        self.data.qpos[self.active_q] = q[7+self.movable]
        self.data.qpos[self.book_q:self.book_q+3] = object_position
        self.data.qpos[self.book_q+3:self.book_q+7] = object_wxyz
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def robot_q(self):
        q = self.fixed_q.copy()
        if self.moving_fixture:
            root = self.data.qpos[self.root_q:self.root_q+7]
            q[:7] = np.r_[root[:3],root[[4,5,6,3]]]
        q[7+self.movable] = self.data.qpos[self.active_q]
        return q

    def tips_and_jacobian(self):
        jacobians = []
        for site in self.tip_ids:
            j = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, j, None, int(site))
            jacobians.append(j[:, self.active_v])
        return self.data.site_xpos[self.tip_ids].copy(), np.asarray(jacobians)

    def rollout(self, q, positions, quaternions, times, *, force_ff=None, kp_arm=120., kp_hand=5., initial_q=None, initial_velocity=None, inertia_ff=False,
                command_velocity=None, force_points=None, initial_object_state=None, smooth_fixture=False,
                contact_controller=None, contact_normals_local=None, contact_active=None, object_feedback=None,wrench_residual=None,
                point_local_trajectory=None, object_local_trajectory=None, inertia_ff_scale=1.,
                contact_body_audit=False, surface_follow=None, surface_follow_plan=None):
        if not np.isfinite(inertia_ff_scale) or not 0<=inertia_ff_scale<=1:
            raise ValueError('Inertial feedforward scale must be in [0,1]')
        if (surface_follow is None) != (surface_follow_plan is None):
            raise ValueError('Surface feedback needs both a configuration and planning poses')
        if surface_follow is not None and (object_feedback is not None or wrench_residual is not None or contact_controller is None):
            raise ValueError('Select one explicit feedback adapter with the shared contact controller')
        relative = times-times[0]
        spline = (CubicSpline(relative, q[:, 7+self.movable], axis=0) if command_velocity is None else
                  CubicHermiteSpline(relative, q[:, 7+self.movable], command_velocity, axis=0))
        end = round(relative[-1]/self.model.opt.timestep)
        step_times = np.minimum(np.arange(end+1)*self.model.opt.timestep, relative[-1])
        commands, velocities = spline(step_times), spline(step_times, 1)
        acceleration = spline(step_times, 2)
        desired_p = np.column_stack([np.interp(step_times, relative, positions[:, j]) for j in range(3)])
        desired_r = Slerp(relative, Rotation.from_quat(quaternions, scalar_first=True))(step_times).as_quat(scalar_first=True)
        if self.moving_fixture:
            fixture_position_spline = CubicSpline(relative, q[:,:3])
            fixture_p = fixture_position_spline(step_times)
            fixture_rotation_spline = (RotationSpline if smooth_fixture else Slerp)(relative, Rotation.from_quat(q[:,3:7]))
            fixture_rotation = fixture_rotation_spline(step_times)
            fixture_r = fixture_rotation.as_quat(scalar_first=True)
            if smooth_fixture:
                fixture_a = fixture_position_spline(step_times, 2)
                fixture_alpha = fixture_rotation.apply(fixture_rotation_spline(step_times, 2))
        force = np.zeros((len(step_times), 6, 3)) if force_ff is None else (
            CubicSpline(relative, force_ff, axis=0)(step_times) if force_points is None else
            np.column_stack([np.interp(step_times, relative, column) for column in force_ff.reshape(len(times), -1).T]).reshape(len(step_times), -1, 3))
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(q[0] if initial_q is None else initial_q, positions[0], quaternions[0])
        self.data.qvel[self.active_v] = velocities[0] if initial_velocity is None else initial_velocity
        if self.moving_fixture:
            self.data.qvel[self.root_v:self.root_v+3] = (q[1,:3]-q[0,:3])/(times[1]-times[0])
            roots = Rotation.from_quat(q[:2,3:7])
            self.data.qvel[self.root_v+3:self.root_v+6] = (roots[0].inv()*roots[1]).as_rotvec()/(times[1]-times[0])
        self.data.qvel[self.book_v:self.book_v+3] = (positions[1]-positions[0])/(times[1]-times[0])
        rot = Rotation.from_quat(quaternions[:2], scalar_first=True)
        self.data.qvel[self.book_v+3:self.book_v+6] = (rot[0].inv()*rot[1]).as_rotvec()/(times[1]-times[0])
        if initial_object_state is not None:
            self.data.qpos[self.book_q:self.book_q+3] = initial_object_state['position']
            self.data.qpos[self.book_q+3:self.book_q+7] = initial_object_state['wxyz']
            self.data.qvel[self.book_v:self.book_v+3] = initial_object_state['linear_velocity']
            self.data.qvel[self.book_v+3:self.book_v+6] = Rotation.from_quat(initial_object_state['wxyz'], scalar_first=True).inv().apply(initial_object_state['angular_velocity_world'])
        mujoco.mj_forward(self.model, self.data)
        initial_pen = max([max(0., -c.dist) for c in self.data.contact], default=0.)
        kp = np.where(self.hand, kp_hand, kp_arm)
        kd = np.where(self.hand, .2, 2.)
        if inertia_ff:
            kd = np.where(self.hand, .4, 8.)
        controller = None
        if point_local_trajectory is not None and (object_feedback is not None or wrench_residual is not None or contact_controller is None):
            raise ValueError('Time-varying surface points currently require the base contact controller')
        if contact_controller is not None:
            from .contact_controller import G1ContactController
            if force_ff is None or force_points is None or contact_normals_local is None or contact_active is None:
                raise ValueError('Controller-aware replay requires forces, body points, inward normals and a contact schedule')
            if object_feedback is not None and wrench_residual is not None:
                raise ValueError('Select one object-feedback adapter per rollout')
            if surface_follow is not None:
                from .surface_follow import SurfaceFollowController
                controller = SurfaceFollowController(self, force_points, contact_normals_local, contact_active,
                    times, contact_controller, kp, point_local_trajectory=point_local_trajectory,
                    object_local_trajectory=object_local_trajectory, follow_config=surface_follow,
                    planning_poses=surface_follow_plan, nominal_positions=q[:, 7+self.movable],
                    nominal_velocities=spline(relative, 1))
            elif wrench_residual is not None:
                from .wrench_residual_controller import WrenchResidualController
                controller=WrenchResidualController(self,force_points,contact_normals_local,contact_active,times,contact_controller,kp,
                    object_positions=positions,object_wxyz=quaternions,residual_config=wrench_residual)
            elif object_feedback is None:
                controller = G1ContactController(self, force_points, contact_normals_local, contact_active,
                                                 times, contact_controller, kp, point_local_trajectory=point_local_trajectory,
                                                 object_local_trajectory=object_local_trajectory)
            else:
                from .object_relative_controller import ObjectRelativeContactController
                controller = ObjectRelativeContactController(self, force_points, contact_normals_local, contact_active,
                    times, contact_controller, kp, object_positions=positions,object_wxyz=quaternions,object_config=object_feedback)
        sample_steps = np.round(relative/self.model.opt.timestep).astype(int)
        command_sample_steps = set(sample_steps[:-1])
        outq, outp, outr, outtip, outtau, outcontact = [], [], [], [], [], []
        outforces, outpoints, outvelocity = [], [], []
        out_robot_wrench, out_environment_wrench, out_com, out_qvel = [], [], [], []
        max_p = max_r = max_pen = max_external = max_sat = 0.
        max_fraction = np.zeros(len(self.active_v))
        max_velocity = np.zeros(len(self.active_v))
        max_limit_violation = 0.
        saturated_steps = 0
        frame_peak_torque_fraction = np.zeros((len(times), len(self.active_v)))
        frame_peak_velocity = np.zeros_like(frame_peak_torque_fraction)
        frame_peak_limit_violation = np.zeros(len(times))
        frame_peak_penetration = np.zeros(len(times))
        interval_robot_impulse = np.zeros((len(times)-1, 6))
        interval_environment_impulse = np.zeros_like(interval_robot_impulse)
        interval_body_impulse = np.zeros((len(times)-1, self.model.nbody))
        interval_peak_robot_force = np.zeros(len(times)-1)
        interval_body_wrench = np.zeros((len(times)-1, self.model.nbody, 6)) if contact_body_audit else None
        interval_patch_wrench = np.zeros((len(times)-1, 6))
        interval_active_patch_wrench = np.zeros_like(interval_patch_wrench)
        torque_audit = TorquePeakAudit(len(times), self.caps,
                                      len(force_points) if controller is not None else 0)
        previous_guidance = np.zeros(len(self.active_v))
        lower = self.kin.robot.model.lowerPositionLimit[7+self.movable]
        upper = self.kin.robot.model.upperPositionLimit[7+self.movable]
        pair, sample = '', 0
        for step, t in enumerate(step_times):
            audit_frame = min(max(0, np.searchsorted(relative, t, side='right')-1), len(times)-1)
            frame_peak_velocity[audit_frame] = np.maximum(frame_peak_velocity[audit_frame], abs(self.data.qvel[self.active_v]))
            max_velocity = np.maximum(max_velocity, np.abs(self.data.qvel[self.active_v]))
            angles = self.data.qpos[self.active_q]
            frame_peak_limit_violation[audit_frame] = max(frame_peak_limit_violation[audit_frame],
                float(np.max(np.maximum(np.maximum(lower-angles, angles-upper), 0.))))
            max_limit_violation = max(max_limit_violation, float(np.max(np.maximum(lower-angles, angles-upper))))
            max_p = max(max_p, float(np.linalg.norm(self.data.qpos[self.book_q:self.book_q+3]-desired_p[step])))
            angle = 2*np.arccos(np.clip(abs(self.data.qpos[self.book_q+3:self.book_q+7]@desired_r[step]), 0, 1))
            max_r = max(max_r, float(angle))
            for c in self.data.contact:
                frame_peak_penetration[audit_frame] = max(frame_peak_penetration[audit_frame], -c.dist)
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
                out_qvel.append(self.data.qvel.copy())
                out_com.append(self.data.xipos[self.book_body].copy())
                counts = np.zeros(2, int)
                contact_forces, contact_points = np.zeros((2,3)), np.zeros((2,3))
                robot_wrench, environment_wrench = np.zeros(6), np.zeros(6)
                weights = np.zeros(2)
                for ci,c in enumerate(self.data.contact):
                    if not self.object_geoms.intersection((c.geom1,c.geom2)): continue
                    other = c.geom2 if c.geom1 in self.object_geoms else c.geom1
                    measured_wrench = object_contact_wrench(self.model, self.data, ci, self.object_geoms, self.book_body)
                    if other in self.environment:
                        environment_wrench += measured_wrench
                    else:
                        robot_wrench += measured_wrench
                    name = self.model.geom(other).name
                    if 'left_' not in name and 'right_' not in name: continue
                    hand = 0 if 'left_' in name else 1
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model,self.data,ci,wrench)
                    if np.linalg.norm(wrench[:3]) < 1e-5: continue
                    counts[hand] += 1
                    world_force = np.asarray(c.frame).reshape(3,3).T@wrench[:3]
                    if c.geom1 in self.object_geoms: world_force *= -1
                    contact_forces[hand] += world_force
                    weight=np.linalg.norm(world_force)
                    contact_points[hand] += weight*c.pos
                    weights[hand] += weight
                contact_points /= np.maximum(weights[:,None],1e-12)
                outcontact.append(counts);outforces.append(contact_forces);outpoints.append(contact_points)
                out_robot_wrench.append(robot_wrench); out_environment_wrench.append(environment_wrench)
                sample += 1
                if sample == len(times):
                    if controller is not None: controller.capture()  # final held command, no extra physics step
                    break
            raw = kp*(commands[step]-self.data.qpos[self.active_q])+kd*(velocities[step]-self.data.qvel[self.active_v])
            raw += self.data.qfrc_bias[self.active_v]
            if force_points is not None:
                raw -= self.data.qfrc_passive[self.active_v]
            if inertia_ff:
                generalized_acceleration = np.zeros(self.model.nv)
                generalized_acceleration[self.active_v] = acceleration[step]
                if self.moving_fixture and smooth_fixture:
                    generalized_acceleration[self.root_v:self.root_v+3] = fixture_a[step]
                    root_rotation = Rotation.from_quat(self.data.qpos[self.root_q+3:self.root_q+7], scalar_first=True)
                    generalized_acceleration[self.root_v+3:self.root_v+6] = root_rotation.inv().apply(fixture_alpha[step])
                inertial = np.zeros(self.model.nv)
                mujoco.mj_mulM(self.model, self.data, inertial, generalized_acceleration)
                raw += inertia_ff_scale*inertial[self.active_v]
            audit_position = kp*(commands[step]-self.data.qpos[self.active_q])
            audit_velocity = kd*(velocities[step]-self.data.qvel[self.active_v])
            audit_compensation = raw-(audit_position+audit_velocity)
            audit_guidance = np.zeros(len(self.active_v))
            if controller is not None:
                tracking = kp*(commands[step]-self.data.qpos[self.active_q])+kd*(velocities[step]-self.data.qvel[self.active_v])
                servo_q, feedforward = controller.step(t, commands[step], velocities[step], force[step], raw-tracking)
                servo_velocity = controller.last['servo_velocity'] if surface_follow is not None else velocities[step]
                raw = kp*(servo_q-self.data.qpos[self.active_q])+kd*(servo_velocity-self.data.qvel[self.active_v])+feedforward
                audit_position = kp*(servo_q-self.data.qpos[self.active_q])
                audit_velocity = kd*(servo_velocity-self.data.qvel[self.active_v])
                audit_guidance = feedforward-audit_compensation
                if step in command_sample_steps: controller.capture()
            elif force_ff is not None:
                if force_points is None:
                    _, jac = self.tips_and_jacobian()
                    raw += np.einsum('kij,ki->j', jac, force[step])
                else:
                    object_rotation = Rotation.from_quat(self.data.qpos[self.book_q+3:self.book_q+7], scalar_first=True)
                    for i, point in enumerate(force_points):
                        world_point = self.data.xpos[point.body]+self.data.xmat[point.body].reshape(3, 3)@point.local
                        local = object_rotation.inv().apply(world_point-self.data.qpos[self.book_q:self.book_q+3])
                        distance = np.linalg.norm(np.maximum(np.maximum(self.bounds[0]-local, local-self.bounds[1]), 0))
                        touching = any(self.book_body in (int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2]))
                                       and point.body in (int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])) for c in self.data.contact)
                        if touching or distance < .002:
                            jac = np.zeros((3, self.model.nv))
                            mujoco.mj_jac(self.model, self.data, jac, None, world_point, point.body)
                            raw += jac[:, self.active_v].T@force[step, i]
                audit_guidance = raw-(audit_position+audit_velocity+audit_compensation)
            last = {} if controller is None else controller.last
            torque_audit.update(audit_frame, t, raw,
                np.column_stack((audit_position, audit_velocity, audit_compensation, audit_guidance)),
                enabled=last.get('force_enabled'), touching=last.get('touching'),
                gap=last.get('contact_gate_gap_m'), contact_step=audit_guidance-previous_guidance)
            previous_guidance = audit_guidance.copy()
            max_sat = max(max_sat, float(np.max(np.abs(raw)/self.caps)))
            frame_peak_torque_fraction[audit_frame] = np.maximum(frame_peak_torque_fraction[audit_frame], abs(raw)/self.caps)
            max_fraction = np.maximum(max_fraction, np.abs(raw)/self.caps)
            saturated_steps += int(np.any(np.abs(raw)>self.caps))
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.active_v] = np.clip(raw, -self.caps, self.caps)
            if self.moving_fixture:
                self.data.mocap_pos[0] = fixture_p[min(step+1,end)]
                self.data.mocap_quat[0] = fixture_r[min(step+1,end)]
            mujoco.mj_step(self.model, self.data)
            # Integrate every solved contact, including short impacts missed by
            # the video-rate samples. mj_step retains the just-solved wrench.
            patches = None
            if contact_body_audit and controller is not None:
                patches = {p.body: (last['robot_surface_points_world'][i],
                                    last['contact_normal_world'][i], last['contact_active'][i])
                           for i,p in enumerate(force_points)}
            partition = object_wrench_partition(self.model, self.data, self.object_geoms,
                self.book_body, self.environment, detail=contact_body_audit, patches=patches)
            robot_w, environment_w, body_f = partition[:3]
            interval = min(np.searchsorted(sample_steps, step, side='right')-1, len(times)-2)
            dt = self.model.opt.timestep
            interval_robot_impulse[interval] += dt*robot_w
            interval_environment_impulse[interval] += dt*environment_w
            interval_body_impulse[interval] += dt*body_f
            if contact_body_audit:
                detail = partition[3]
                interval_body_wrench[interval] += dt*detail['body_wrench']
                interval_patch_wrench[interval] += dt*detail['selected_patch_wrench']
                interval_active_patch_wrench[interval] += dt*detail['active_selected_patch_wrench']
            interval_peak_robot_force[interval] = max(interval_peak_robot_force[interval],
                                                     np.linalg.norm(robot_w[:3]))
            if any(self.data.warning[int(kind)].number for kind in
                   (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)):
                raise RuntimeError(f'Unstable physics at step {step}; refusing a reset/stitched replay')
        outp, outr = np.asarray(outp), np.asarray(outr)
        return dict(q=np.asarray(outq), object_positions=outp, object_wxyz=outr, fk_tip_positions=np.asarray(outtip),
                    command_q=q, timestamps=times, torques=np.asarray(outtau),
                    joint_velocities=np.asarray(outvelocity), contacts_per_hand=np.asarray(outcontact),
                    generalized_velocities=np.asarray(out_qvel), object_com_world=np.asarray(out_com),
                    robot_contact_wrench_world=np.asarray(out_robot_wrench),
                    environment_contact_wrench_world=np.asarray(out_environment_wrench),
                    interval_robot_contact_impulse_world=interval_robot_impulse,
                    interval_environment_contact_impulse_world=interval_environment_impulse,
                    interval_robot_body_unsigned_impulse_ns=interval_body_impulse,
                    interval_peak_robot_contact_force_n=interval_peak_robot_force,
                    interval_timestamps=sample_steps*self.model.opt.timestep,
                    contact_forces_world=np.asarray(outforces), contact_points_world=np.asarray(outpoints),
                    position_error=np.linalg.norm(outp-positions, axis=1),
                    rotation_error=(Rotation.from_quat(outr, scalar_first=True).inv()*Rotation.from_quat(quaternions, scalar_first=True)).magnitude(),
                    max_position_error_m=max_p, max_rotation_error_rad=max_r, max_penetration_m=max_pen,
                    initial_max_penetration_m=initial_pen, worst_collision_pair=pair, max_torque_fraction_requested=max_sat,
                    torque_fraction_by_joint=max_fraction,
                    frame_peak_torque_fraction=frame_peak_torque_fraction,
                    frame_peak_joint_velocity_rad_s=frame_peak_velocity,
                    frame_peak_joint_limit_violation_rad=frame_peak_limit_violation,
                    frame_peak_penetration_m=frame_peak_penetration,
                    max_joint_velocity_rad_s=max_velocity, max_joint_limit_violation_rad=max_limit_violation,
                    saturation_steps=saturated_steps, torque_caps=self.caps, movable_joint_indices=self.movable,
                    max_object_applied_force=max_external, fixed_root_and_legs=not self.moving_fixture,
                    moving_external_pelvis_fixture=self.moving_fixture, dynamic_waist=self.move_waist,
                    **torque_audit.export(),
                    **(dict(interval_robot_body_wrench_impulse_world=interval_body_wrench,
                            interval_selected_patch_impulse_world=interval_patch_wrench,
                            interval_active_selected_patch_impulse_world=interval_active_patch_wrench,
                            robot_body_names=np.array([self.model.body(i).name for i in range(self.model.nbody)]))
                       if contact_body_audit else {}),
                    **({**controller.export(), 'controller_command_times': np.minimum(relative, step_times[-2])}
                       if controller is not None else {}))
