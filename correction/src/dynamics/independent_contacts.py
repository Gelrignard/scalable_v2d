"""Finite-mass Cartesian pads, a free object, and recorded force guidance.

This deliberately removes robot IK, joint limits, arm geometry and walking.
The object receives only gravity and physical contact forces. No constraint,
pose assignment or force actuator attaches it to a pad during a rollout.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .contact_controller import ContactControllerConfig, NormalForceFeedback, ContactEngagement
from .robot_scene import numbers
from .supported_contacts import SupportedContacts


class IndependentContacts(SupportedContacts):
    def __init__(self, bounds, *, object_parts=None, floor_z=0., object_inertial=None, **kwargs):
        super().__init__(bounds, **kwargs)
        root = ET.fromstring(self.xml)
        world = root.find('worldbody')
        world.find("geom[@name='floor']").set('pos', numbers([0, 0, floor_z]))
        book = world.find("body[@name='book']")
        mass = float(book.find('geom').get('mass'))
        if object_inertial is not None:
            com=np.asarray(object_inertial['com_local_m'],float)
            inertia=np.asarray(object_inertial['diagonal_kg_m2'],float)
            if com.shape!=(3,) or inertia.shape!=(3,) or not np.isfinite(np.r_[com,inertia]).all() or np.any(inertia<=0):
                raise ValueError('Invalid explicit object COM/inertia')
            ET.SubElement(book,'inertial',pos=numbers(com),mass=str(mass),diaginertia=numbers(inertia))
        if object_parts is not None:
            original = book.find('geom')
            mass = float(original.get('mass'))
            attributes = {k: v for k, v in original.attrib.items()
                          if k not in ('name', 'type', 'size', 'pos', 'mass')}
            book.remove(original)
            assets = ET.SubElement(root, 'asset')
            for i, (path, fraction) in enumerate(object_parts):
                name = f'object_part_{i}'
                ET.SubElement(assets, 'mesh', name=name, file=str(Path(path).resolve()))
                ET.SubElement(book, 'geom', name='book_geom' if i == 0 else f'object_geom_{i}',
                    type='mesh', mesh=name, mass=str(mass*fraction), **attributes)
        for geom in world.iter('geom'):
            geom.set('margin', '.0005')
            geom.set('solref', '.003 1')
            geom.set('solimp', '.95 .99 .001')
            geom.set('condim', '3')
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.book_body = self.model.body('book').id
        self.object_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.book_body))
        self.pad_index = {self.model.geom(f'pad{i}_geom').id: i for i in range(self.pad_count)}
        self.qadr = np.array([[int(self.model.joint(f'pad{i}_{j}').qposadr[0]) for j in range(3)] for i in range(self.pad_count)])
        self.vadr = np.array([[int(self.model.joint(f'pad{i}_{j}').dofadr[0]) for j in range(3)] for i in range(self.pad_count)])

    def contact_measurements(self, normals):
        forces, points, moments = (np.zeros((self.pad_count, 3)) for _ in range(3))
        matched = np.zeros((self.pad_count, 3))
        weights, counts = np.zeros(self.pad_count), np.zeros(self.pad_count, int)
        touching = np.zeros(self.pad_count, bool)
        for ci, contact in enumerate(self.data.contact):
            first = contact.geom1 in self.object_geoms
            if not first and contact.geom2 not in self.object_geoms:
                continue
            index = self.pad_index.get(contact.geom2 if first else contact.geom1)
            if index is None:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, ci, wrench)
            magnitude = np.linalg.norm(wrench[:3])
            if magnitude < 1e-6:
                continue
            axes = np.asarray(contact.frame).reshape(3, 3)
            sign = -1 if first else 1
            force = sign*axes.T@wrench[:3]
            forces[index] += force
            points[index] += magnitude*contact.pos
            moments[index] += sign*axes.T@wrench[3:]+np.cross(contact.pos-self.data.xipos[self.book_body], force)
            weights[index] += magnitude
            counts[index] += 1
            if sign*axes[0]@normals[index] > .5:
                matched[index] += force
                touching[index] = True
        points /= np.maximum(weights[:, None], 1e-12)
        return forces, points, moments, counts, matched, touching

    def rollout(self, times, targets, positions, quaternions, *, initial_pads,
                planned_forces=None, inward_normals_local=None, active=None,
                config=None, object_local_points=None):
        times, targets = np.asarray(times, float), np.asarray(targets, float)
        relative = times-times[0]
        count = len(times)
        if count < 2 or np.any(np.diff(times) <= 0):
            raise ValueError('Strictly increasing timestamps required')
        shape = (count, self.pad_count, 3)
        if targets.shape != shape or not np.isfinite(targets).all():
            raise ValueError('One finite Cartesian target per pad is required at each frame')
        guided = planned_forces is not None
        if guided and (np.shape(planned_forces) != shape or np.shape(inward_normals_local) != shape
                       or np.shape(active) != shape[:2]):
            raise ValueError('Force guidance requires matching forces, inward normals and schedule')
        forces = np.zeros(shape) if not guided else np.asarray(planned_forces, float)
        normals = np.zeros(shape) if not guided else np.array(inward_normals_local, dtype=float, copy=True)
        schedule = np.zeros(shape[:2], bool) if not guided else np.asarray(active, bool)
        normal_lengths = np.linalg.norm(normals, axis=-1)
        if not np.isfinite(forces).all() or not np.isfinite(normals).all() or np.any(normal_lengths[schedule] < 1e-8):
            raise ValueError('Invalid planned forces or active inward normals')
        normals /= np.maximum(normal_lengths[..., None], 1e-12)
        cfg = config or ContactControllerConfig(feedback_gain=.25)
        feedback = NormalForceFeedback(self.pad_count, cfg)
        engagement = ContactEngagement(self.pad_count, cfg.engagement_seconds)
        if object_local_points is not None:
            object_local_points = np.asarray(object_local_points, float)
            if object_local_points.shape != shape or not np.isfinite(object_local_points).all():
                raise ValueError('Finite object-frame anchors required for patch-proximity acquisition')
            if not cfg.patch_distance_gate:
                raise ValueError('Object anchors require an explicit patch-distance gate')
        m, d = self.model, self.data
        dt = m.opt.timestep
        sample_steps = np.round(relative/dt).astype(int)
        if np.any(np.diff(sample_steps) <= 0):
            raise ValueError('Source frames must be farther apart than physics steps')
        dense_times = np.minimum(np.arange(sample_steps[-1]+1)*dt, relative[-1])
        spline = CubicSpline(relative, targets, axis=0)
        commands, velocities = spline(dense_times), spline(dense_times, 1)
        desired_p = np.column_stack([np.interp(dense_times, relative, positions[:, i]) for i in range(3)])
        desired_r = Slerp(relative, Rotation.from_quat(quaternions, scalar_first=True))(dense_times).as_quat(scalar_first=True)
        k = np.clip(np.searchsorted(relative, dense_times, side='right')-1, 0, count-2)
        u = ((dense_times-relative[k])/(relative[k+1]-relative[k]))[:, None, None]
        dense_force = (1-u)*forces[k]+u*forces[k+1]
        dense_normal = (1-u)*normals[k]+u*normals[k+1]
        lengths = np.linalg.norm(dense_normal, axis=-1)
        dense_active = schedule[k] & schedule[k+1] & (lengths > .5)
        dense_normal /= np.maximum(lengths[..., None], 1e-12)
        dense_anchor = (None if object_local_points is None else
                        (1-u)*object_local_points[k]+u*object_local_points[k+1])
        mujoco.mj_resetData(m, d)
        d.qpos[:3], d.qpos[3:7] = positions[0], quaternions[0]
        d.qpos[self.qadr] = initial_pads
        # Both trials start at rest, before approach. No established-grasp reset.
        mujoco.mj_forward(m, d)
        fields = ('object_positions', 'object_wxyz', 'object_linear_velocity', 'pad_positions',
                  'pad_velocities', 'contact_forces', 'contact_points', 'contact_moments',
                  'contact_counts', 'motor_forces', 'commanded_forces', 'feedback_correction_n',
                  'force_enabled', 'contact_gate_gap_m', 'contact_eligible', 'contact_engagement')
        out = {name: [] for name in fields}
        frame = saturation = 0
        max_depth = initial_depth = max([max(0., -c.dist) for c in d.contact], default=0.)
        max_position = max_rotation = max_motor = max_external = 0.
        for step, time in enumerate(dense_times):
            rot = d.xmat[self.book_body].reshape(3, 3)
            normal = dense_normal[step]@rot.T
            measured, points, moments, counts, matched, touching = self.contact_measurements(normal)
            gap = np.full(self.pad_count, np.inf)
            eligible = dense_active[step] & touching
            if dense_anchor is not None:
                anchor_world = d.qpos[:3]+dense_anchor[step]@rot.T
                # A sphere has no fixed front patch. Distance to its surface
                # is center-to-anchor distance minus radius, not a box test.
                gap = np.maximum(np.linalg.norm(d.qpos[self.qadr]-anchor_world, axis=1)-self.radius, 0.)
                eligible = dense_active[step] & (touching | (gap < cfg.near_surface_m))
            gain = engagement.update(eligible, dense_active[step], dt)
            enabled = gain > 0.
            wanted = np.maximum(0., np.einsum('ij,ij->i', dense_force[step], normal))
            sensed = np.maximum(0., np.einsum('ij,ij->i', matched, normal))
            compression, correction = feedback.update(wanted, sensed, touching, dense_active[step], dt)
            tangent = dense_force[step]-np.einsum('ij,ij->i', dense_force[step], normal)[:, None]*normal
            tangent *= np.minimum(1., cfg.friction*compression/np.maximum(np.linalg.norm(tangent, axis=1), 1e-12))[:, None]
            commanded = tangent+compression[:, None]*normal
            commanded *= gain[:, None]
            motor = self.kp*(commands[step]-d.qpos[self.qadr])+self.kd*(velocities[step]-d.qvel[self.vadr])
            motor += commanded-self.pad_mass*m.opt.gravity
            magnitude = np.linalg.norm(motor, axis=1, keepdims=True)
            saturation += int(np.any(magnitude > self.force_limit))
            motor *= np.minimum(1., self.force_limit/np.maximum(magnitude, 1e-12))
            max_motor = max(max_motor, float(np.linalg.norm(motor, axis=1).max()))
            max_depth = max(max_depth, max([max(0., -c.dist) for c in d.contact], default=0.))
            max_position = max(max_position, float(np.linalg.norm(d.qpos[:3]-desired_p[step])))
            max_rotation = max(max_rotation, float(2*np.arccos(np.clip(abs(d.qpos[3:7]@desired_r[step]), 0, 1))))
            max_external = max(max_external, float(np.max(abs(d.qfrc_applied[:6]))), float(np.max(abs(d.xfrc_applied))))
            if step == sample_steps[frame]:
                values = (d.qpos[:3], d.qpos[3:7], d.qvel[:3], d.qpos[self.qadr], d.qvel[self.vadr],
                          measured, points, moments, counts, motor, commanded, correction, enabled,
                          gap, eligible, gain)
                for name, value in zip(fields, values): out[name].append(value.copy())
                frame += 1
                if frame == count: break
            d.qfrc_applied[:] = 0.
            d.qfrc_applied[self.vadr] = motor
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)  # contact locations/forces at the exported state
            if any(d.warning[int(kind)].number for kind in
                   (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
                raise RuntimeError(f'Unstable independent-pad physics at step {step}; refusing reset/stitched output')
            if not np.isfinite(d.qpos).all():
                raise FloatingPointError('Nonfinite independent-pad rollout')
        record = {name: np.asarray(values) for name, values in out.items()}
        error = np.linalg.norm(record['object_positions']-positions, axis=-1)
        angle = (Rotation.from_quat(record['object_wxyz'], scalar_first=True).inv()*
                 Rotation.from_quat(quaternions, scalar_first=True)).magnitude()
        record.update(targets=targets, planned_forces=forces, timestamps=times,
            simulation_timestamps=sample_steps*dt+times[0], position_error=error, rotation_error=angle,
            max_position_error_m=max_position, max_rotation_error_rad=max_rotation,
            max_penetration_m=max_depth, initial_max_penetration_m=initial_depth,
            max_motor_force_n=max_motor, max_object_applied_force=max_external,
            saturation_steps=saturation, steps=len(dense_times)-1)
        return record
