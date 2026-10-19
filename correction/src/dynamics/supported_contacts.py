"""Supported, force-limited fingertip experiment with a freely simulated object.

This intentionally removes the arm, hand link geometry, and locomotion. Each
contact pad has three actuated translational DOFs and finite mass. Only those
DOFs receive control forces; the book receives gravity and contact forces.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .robot_scene import numbers


class SupportedContacts:
    def __init__(self, bounds, *, mass=1., friction=.6, radius=.006,
                 stiffness=1200., force_limit=25., dt=.001, table=None):
        self.bounds = np.asarray(bounds)
        self.radius, self.kp, self.force_limit = radius, stiffness, force_limit
        self.pad_mass = .04
        self.kd = 2*np.sqrt(stiffness*self.pad_mass)
        root = ET.Element('mujoco', model='supported_fingertips')
        ET.SubElement(root, 'option', timestep=str(dt), gravity='0 0 -9.81',
                      integrator='implicitfast', iterations='60')
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='5 5 .1',
                      contype='1', conaffinity='3', friction=f'{friction} .00001 .00001')
        if table is not None:
            center, table_radius, height = table
            ET.SubElement(world, 'geom', name='table', type='cylinder',
                          pos=numbers(center), size=numbers([table_radius, height/2]),
                          contype='1', conaffinity='3', friction=f'{friction} .00001 .00001')
        book = ET.SubElement(world, 'body', name='book')
        ET.SubElement(book, 'freejoint', name='book_joint')
        ET.SubElement(book, 'geom', name='book_geom', type='box',
                      size=numbers(np.ptp(bounds, axis=0)/2), pos=numbers(np.mean(bounds, axis=0)),
                      mass=str(mass), contype='1', conaffinity='3', condim='3',
                      friction=f'{friction} .00001 .00001', solref='.003 1', solimp='.95 .99 .001')
        for i in range(6):
            pad = ET.SubElement(world, 'body', name=f'pad{i}')
            for j in range(3):
                ET.SubElement(pad, 'joint', name=f'pad{i}_{j}', type='slide',
                              axis=numbers(np.eye(3)[j]))
            ET.SubElement(pad, 'geom', name=f'pad{i}_geom', type='sphere', size=str(radius),
                          mass=str(self.pad_mass), contype='2', conaffinity='3', condim='3',
                          friction=f'{friction} .00001 .00001', solref='.003 1', solimp='.95 .99 .001')
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.book_geom = self.model.geom('book_geom').id
        self.qadr = np.array([[int(self.model.joint(f'pad{i}_{j}').qposadr[0]) for j in range(3)] for i in range(6)])
        self.vadr = np.array([[int(self.model.joint(f'pad{i}_{j}').dofadr[0]) for j in range(3)] for i in range(6)])

    def environment_penetration(self, positions, quaternions, pads):
        """Geometry-only audit used before reset, never during a rollout."""
        worst = 0.
        for p, r, contact in zip(positions, quaternions, pads):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:3], self.data.qpos[3:7] = p, r
            self.data.qpos[self.qadr] = contact
            mujoco.mj_forward(self.model, self.data)
            for c in self.data.contact:
                if self.model.geom_bodyid[c.geom1] == 0 or self.model.geom_bodyid[c.geom2] == 0:
                    worst = max(worst, -float(c.dist))
        return worst

    def rollout(self, times, targets, positions, quaternions, *, initial_pads,
                gains=1., initial_velocity=True):
        """Replay pad commands. Object pose is assigned exactly once, at reset."""
        relative = times-times[0]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = positions[0]
        self.data.qpos[3:7] = quaternions[0]
        self.data.qpos[self.qadr] = initial_pads
        if initial_velocity:
            self.data.qvel[:3] = (positions[1]-positions[0])/(times[1]-times[0])
            rotation = Rotation.from_quat(quaternions[:2, [1, 2, 3, 0]])
            self.data.qvel[3:6] = (rotation[0].inv()*rotation[1]).as_rotvec()/(times[1]-times[0])
            local_pad = rotation[0].inv().apply(initial_pads-positions[0])
            self.data.qvel[self.vadr] = self.data.qvel[:3]+rotation[0].apply(np.cross(self.data.qvel[3:6], local_pad))
        mujoco.mj_forward(self.model, self.data)
        initial_penetration = max([max(0., -c.dist) for c in self.data.contact], default=0.)
        sample_steps = np.round(relative/self.model.opt.timestep).astype(int)
        steps = np.arange(sample_steps[-1]+1)
        step_times = np.minimum(steps*self.model.opt.timestep, relative[-1])
        spline = CubicSpline(relative, targets, axis=0)
        commands, velocities = spline(step_times), spline(step_times, 1)
        max_command_speed = float(np.max(np.linalg.norm(velocities, axis=-1)))
        max_command_acceleration = float(np.max(np.linalg.norm(spline(step_times, 2), axis=-1)))
        reference_p = np.column_stack([np.interp(step_times, relative, positions[:, j]) for j in range(3)])
        reference_r = Slerp(relative, Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]))(step_times).as_quat()[:, [3, 0, 1, 2]]
        out_p, out_r, out_pad, out_force, out_depth, out_motor = [], [], [], [], [], []
        saturation = 0
        max_penetration = max_motor = max_external = max_position = max_rotation = 0.
        worst_pair, worst_time = '', 0.
        frame = 0
        for step in steps:
            max_position = max(max_position, float(np.linalg.norm(self.data.qpos[:3]-reference_p[step])))
            cosine = abs(float(self.data.qpos[3:7]@reference_r[step]))
            max_rotation = max(max_rotation, float(2*np.arccos(np.clip(cosine, 0., 1.))))
            for c in self.data.contact:
                if -c.dist > max_penetration:
                    max_penetration = float(-c.dist)
                    worst_pair = self.model.geom(c.geom1).name+' / '+self.model.geom(c.geom2).name
                    worst_time = float(self.data.time)
            max_external = max(max_external, float(np.max(np.abs(self.data.qfrc_applied[:6]))),
                               float(np.max(np.abs(self.data.xfrc_applied))))
            if step == sample_steps[frame]:
                out_p.append(self.data.qpos[:3].copy())
                out_r.append(self.data.qpos[3:7].copy())
                out_pad.append(self.data.qpos[self.qadr].copy())
                contact_forces = np.zeros((6, 3))
                depth = 0.
                for ic, c in enumerate(self.data.contact):
                    if self.book_geom not in (c.geom1, c.geom2):
                        continue
                    other = c.geom2 if c.geom1 == self.book_geom else c.geom1
                    name = self.model.geom(other).name
                    if name.startswith('pad'):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, ic, force)
                        sign = -1 if c.geom1 == self.book_geom else 1
                        contact_forces[int(name[3])] += sign*c.frame.reshape(3, 3).T@force[:3]
                        depth = max(depth, -c.dist)
                out_force.append(contact_forces)
                out_depth.append(depth)
                out_motor.append(self.data.qfrc_applied[self.vadr].copy())
                frame += 1
                if frame == len(times):
                    break
            raw = gains*self.kp*(commands[step]-self.data.qpos[self.qadr])
            raw += np.sqrt(gains)*self.kd*(velocities[step]-self.data.qvel[self.vadr])
            raw[:, 2] += self.pad_mass*9.81
            norm = np.linalg.norm(raw, axis=1, keepdims=True)
            saturation += int(np.any(norm > self.force_limit))
            self.data.qfrc_applied[:] = 0.
            self.data.qfrc_applied[self.vadr] = raw*np.minimum(1., self.force_limit/np.maximum(norm, 1e-12))
            max_motor = max(max_motor, float(np.max(np.linalg.norm(self.data.qfrc_applied[self.vadr], axis=1))))
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise FloatingPointError('Nonfinite supported contact rollout')
        p, r = np.asarray(out_p), np.asarray(out_r)
        errors = np.linalg.norm(p-positions, axis=1)
        angle = (Rotation.from_quat(r[:, [1, 2, 3, 0]]).inv()*
                 Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])).magnitude()
        return dict(object_positions=p, object_wxyz=r, pad_positions=np.asarray(out_pad),
                    contact_forces=np.asarray(out_force), penetration=np.asarray(out_depth),
                    motor_forces=np.asarray(out_motor), targets=targets, timestamps=times,
                    position_error=errors, rotation_error=angle,
                    initial_max_penetration_m=initial_penetration, saturation_steps=saturation,
                    steps=len(steps)-1, max_object_applied_force=max_external,
                    max_penetration_m=max_penetration, max_motor_force_n=max_motor,
                    worst_penetration_pair=worst_pair, worst_penetration_time_s=worst_time,
                    max_command_speed_m_s=max_command_speed,
                    max_command_acceleration_m_s2=max_command_acceleration,
                    max_position_error_m=max_position, max_rotation_error_rad=max_rotation)
