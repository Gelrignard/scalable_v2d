"""Explicit servo commands and bounded, contact-gated force feedback.

Forces are robot-on-object in world coordinates. The position adapter converts
the contact torque only; the existing gravity/inertia compensation is retained.
This is a local execution adapter, not a claim about the challenge action API.
"""
from dataclasses import dataclass, asdict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .contact_measurement import object_contact_wrench


@dataclass(frozen=True)
class ContactControllerConfig:
    interface: str = 'torque'
    feedback_gain: float = 0.
    filter_seconds: float = .02
    correction_limit_n: float = 2.
    correction_rate_n_s: float = 100.
    normal_limit_n: float = 15.
    friction: float = .45
    near_surface_m: float = .002
    patch_distance_gate: bool = False
    engagement_seconds: float = 0.
    arm_offset_rad: float = .08
    hand_offset_rad: float = .10
    offset_rate_rad_s: float = .5

    def __post_init__(self):
        if self.interface not in ('torque', 'position'):
            raise ValueError('interface must be torque or position')
        values = [v for k, v in asdict(self).items() if k != 'interface']
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Controller parameters must be finite and nonnegative')
        if min(self.filter_seconds, self.normal_limit_n, self.offset_rate_rad_s) <= 0:
            raise ValueError('Filter, normal limit and offset rate must be positive')


def position_target(q, contact_torque, kp, lower, upper, max_offset):
    """Convert contact torque to a servo equilibrium; expose lost torque.

    The velocity target stays the physical trajectory velocity, not the
    derivative of this equilibrium. Never add contact_torque a second time.
    """
    q, contact_torque, kp, lower, upper, max_offset = np.broadcast_arrays(
        *[np.asarray(a, float) for a in (q, contact_torque, kp, lower, upper, max_offset)])
    if not all(np.isfinite(a).all() for a in (q, contact_torque, kp, lower, upper, max_offset)):
        raise ValueError('Servo conversion requires finite inputs')
    if np.any(kp <= 0) or np.any(lower > upper) or np.any(max_offset < 0):
        raise ValueError('Invalid gain, limits or offset bound')
    requested = q+contact_torque/kp
    applied = np.clip(q+np.clip(contact_torque/kp, -max_offset, max_offset), lower, upper)
    return applied, requested, contact_torque-kp*(applied-q)


class NormalForceFeedback:
    """No integral state; no correction without a matched physical contact."""
    def __init__(self, count, config):
        self.config = config
        self.filtered = np.zeros(count)
        self.correction = np.zeros(count)

    def update(self, desired, measured, touching, active, dt):
        cfg = self.config
        desired, measured = np.asarray(desired), np.asarray(measured)
        enabled = np.asarray(touching, bool) & np.asarray(active, bool) & (desired > 1e-6)
        self.filtered += dt/(cfg.filter_seconds+dt)*(measured-self.filtered)
        self.filtered[~enabled] = 0.
        target = np.clip(cfg.feedback_gain*(desired-self.filtered),
                         -cfg.correction_limit_n, cfg.correction_limit_n)
        self.correction += np.clip(target-self.correction,
                                   -cfg.correction_rate_n_s*dt, cfg.correction_rate_n_s*dt)
        self.correction[~enabled] = 0.
        # All gains share the same force limits, so the feedback ablation does
        # not accidentally compare two different friction/force limiters.
        normal = np.clip(desired+self.correction, 0., cfg.normal_limit_n)
        return normal, self.correction.copy()


class ContactEngagement:
    """Optional low-pass gain on joint force guidance, not an object actuator.

    After a brief loss of eligible contact, guidance decays on the robot.
    Actual object force still requires simulator contact. An inactive planned
    mode clears the gain; the planned force profile must unload before it.
    """
    def __init__(self, count, seconds):
        self.seconds = seconds
        self.gain = np.zeros(count)

    def update(self, eligible, active, dt):
        if self.seconds == 0:
            self.gain[:] = eligible
        else:
            self.gain += dt/(self.seconds+dt)*(np.asarray(eligible, float)-self.gain)
        self.gain[~np.asarray(active, bool)] = 0.
        return self.gain.copy()


class G1ContactController:
    """One instance per uninterrupted rollout; never changes object state."""
    def __init__(self, scene, points, normals_local, active, times, config, kp,
                 point_local_trajectory=None, object_local_trajectory=None):
        self.scene, self.points, self.config, self.kp = scene, points, config, kp
        self.times = np.asarray(times)-times[0]
        self.normals, self.active = np.asarray(normals_local, float), np.asarray(active, bool)
        shape = (len(times), len(points))
        if self.normals.shape != (*shape, 3) or self.active.shape != shape:
            raise ValueError('Contact normals/schedule must match timestamps and robot points')
        norm = np.linalg.norm(self.normals, axis=-1)
        if not np.isfinite(self.normals).all() or np.any(norm[self.active] < 1e-8):
            raise ValueError('Every active contact needs a finite inward normal')
        self.normals = self.normals/np.maximum(norm[..., None], 1e-12)
        self.local_points = (np.tile(np.asarray([p.local for p in points]), (len(times), 1, 1))
                             if point_local_trajectory is None else np.asarray(point_local_trajectory, float))
        if self.local_points.shape != (*shape, 3) or not np.isfinite(self.local_points).all():
            raise ValueError('Robot surface trajectory must be finite and match contact channels')
        moving = np.linalg.norm(np.diff(self.local_points, axis=0), axis=2) > 1e-9
        if np.any(moving & (self.active[:-1] | self.active[1:])):
            raise ValueError('Surface changes require an inactive interval; rolling contacts are not implemented')
        self.object_points = None
        if config.patch_distance_gate:
            self.object_points = (np.tile(np.asarray([p.object_local for p in points]), (len(times), 1, 1))
                                  if object_local_trajectory is None else np.asarray(object_local_trajectory, float))
            if self.object_points.shape != (*shape, 3) or not np.isfinite(self.object_points).all():
                raise ValueError('Object patch trajectory must match finite contact channels')
        bodies = [p.body for p in points]
        if len(set(bodies)) != len(bodies):
            raise ValueError('Force sensing requires one contact channel per robot body')
        self.body_index = {body: i for i, body in enumerate(bodies)}
        self.feedback = NormalForceFeedback(len(points), config)
        self.engagement = ContactEngagement(len(points), config.engagement_seconds)
        self.offset = np.zeros(len(kp))
        self.lower = scene.kin.robot.model.lowerPositionLimit[7+scene.movable]
        self.upper = scene.kin.robot.model.upperPositionLimit[7+scene.movable]
        self.max_offset = np.where(scene.hand, config.hand_offset_rad, config.arm_offset_rad)
        self.logs = []
        self.max_conversion_residual = self.max_offset_rate = 0.
        self.incompatible_limit_steps = 0
        self.clipped_steps = self.feedback_steps = 0
        self.max_feedback = self.max_contact_command_step = 0.
        self.previous_contact_torque = np.zeros(len(kp))
        self.last = None
        # Optional, explicitly supplied by the observed-state planner. The
        # immutable demonstration arrays remain the default for old replays.
        self.reference_override = None

    def step(self, time, q, velocity, planned_force, compensation):
        s, cfg, dt = self.scene, self.config, self.scene.model.opt.timestep
        m, d = s.model, s.data
        k = min(max(0, np.searchsorted(self.times, time, side='right')-1), len(self.times)-2)
        u = np.clip((time-self.times[k])/(self.times[k+1]-self.times[k]), 0, 1)
        # No interpolation of force through an inactive endpoint.
        active = self.active[k] & self.active[k+1]
        normal = (1-u)*self.normals[k]+u*self.normals[k+1]
        lengths = np.linalg.norm(normal, axis=1)
        active &= lengths > .5  # reject an interpolated normal reversal
        normal /= np.maximum(lengths[:, None], 1e-12)
        rotation = Rotation.from_quat(d.qpos[s.book_q+3:s.book_q+7], scalar_first=True)
        normal = rotation.apply(normal)
        local_points = (1-u)*self.local_points[k]+u*self.local_points[k+1]
        object_points = ((1-u)*self.object_points[k]+u*self.object_points[k+1]
                         if self.object_points is not None else None)
        if self.reference_override is not None:
            override = self.reference_override
            active = override['active'].copy()
            normal = rotation.apply(override['inward'])
            local_points = override['robot_local']
            object_points = override['object_local']
        world_points = np.array([d.xpos[p.body]+d.xmat[p.body].reshape(3, 3)@local
                                 for p, local in zip(self.points, local_points)])
        measured, touching = np.zeros_like(planned_force), np.zeros(len(self.points), bool)
        measured_wrench = np.zeros((len(self.points), 6))
        measured_point = np.zeros_like(planned_force)
        measured_weight = np.zeros(len(self.points))
        for ci, contact in enumerate(d.contact):
            if not s.object_geoms.intersection((contact.geom1, contact.geom2)):
                continue
            object_first = contact.geom1 in s.object_geoms
            other = contact.geom2 if object_first else contact.geom1
            i = self.body_index.get(int(m.geom_bodyid[other]))
            if i is None:
                continue
            frame = np.asarray(contact.frame).reshape(3, 3)
            sign = -1 if object_first else 1
            # A different face or remote patch on this body is not our sensor.
            if sign*frame[0]@normal[i] < .5 or np.linalg.norm(contact.pos-world_points[i]) > .03:
                continue
            if self.reference_override is not None:
                anchor=d.qpos[s.book_q:s.book_q+3]+rotation.apply(object_points[i])
                if np.linalg.norm(contact.pos-anchor) > .025:
                    continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, wrench)
            measured[i] += sign*frame.T@wrench[:3]
            measured_wrench[i] += object_contact_wrench(m, d, ci, s.object_geoms, s.book_body)
            weight = np.linalg.norm(wrench[:3])
            measured_point[i] += weight*contact.pos
            measured_weight[i] += weight
            touching[i] |= wrench[0] > 1e-6
        measured_point /= np.maximum(measured_weight[:, None], 1e-12)
        local = rotation.inv().apply(world_points-d.qpos[s.book_q:s.book_q+3])
        gap = (np.linalg.norm(local-object_points, axis=1)
               if object_points is not None else
               np.linalg.norm(np.maximum(np.maximum(s.bounds[0]-local, local-s.bounds[1]), 0), axis=1))
        eligible = active & (touching | (gap < cfg.near_surface_m))
        engagement = self.engagement.update(eligible, active, dt)
        enabled = engagement > 0.
        wanted = np.maximum(0., np.einsum('ij,ij->i', planned_force, normal))
        sensed = np.maximum(0., np.einsum('ij,ij->i', measured, normal))
        normal_command, correction = self.feedback.update(wanted, sensed, touching, active, dt)
        # Use the signed normal component for the tangential projection, even
        # when a bad reference demands tension. The normal command remains >=0.
        tangent = planned_force-np.einsum('ij,ij->i', planned_force, normal)[:, None]*normal
        tangent *= np.minimum(1., cfg.friction*normal_command/np.maximum(np.linalg.norm(tangent, axis=1), 1e-12))[:, None]
        applied = tangent+normal_command[:, None]*normal
        applied *= engagement[:, None]
        torque = np.zeros(len(q))
        for i, point in enumerate(self.points):
            if not enabled[i]:
                continue
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, world_points[i], point.body)
            torque += jac[:, s.active_v].T@applied[i]
        command, requested, residual = q.copy(), q.copy(), np.zeros(len(q))
        feedforward = compensation+torque
        if cfg.interface == 'position':
            command, requested, _ = position_target(q, torque, self.kp, self.lower, self.upper, self.max_offset)
            target_offset = command-q
            old_offset = self.offset.copy()
            self.offset += np.clip(target_offset-self.offset, -cfg.offset_rate_rad_s*dt, cfg.offset_rate_rad_s*dt)
            command = np.clip(q+self.offset, self.lower, self.upper)
            self.offset = command-q
            # A bad physical reference can make the offset-rate box and the
            # absolute joint-limit box disjoint. Prioritize absolute limits,
            # record the conflict, and reject this contract; never hide it.
            self.incompatible_limit_steps += int(np.any(abs(self.offset-old_offset) > cfg.offset_rate_rad_s*dt+1e-10))
            self.max_offset_rate = max(self.max_offset_rate, float(np.max(abs(self.offset-old_offset))/dt))
            residual = torque-self.kp*self.offset
            feedforward = compensation.copy()
            self.clipped_steps += int(np.max(abs(residual)) > 1e-6)
        self.max_conversion_residual = max(self.max_conversion_residual, float(np.max(abs(residual))))
        self.max_feedback = max(self.max_feedback, float(np.max(abs(correction))))
        self.feedback_steps += int(np.any(abs(correction) > 1e-8))
        self.max_contact_command_step = max(self.max_contact_command_step, float(np.max(abs(torque-self.previous_contact_torque))))
        self.previous_contact_torque = torque
        self.last = dict(servo_q=command.copy(), requested_servo_q=requested.copy(), servo_velocity=velocity.copy(),
                         feedforward_torque=feedforward.copy(), contact_torque=torque, conversion_residual=residual,
                         measured_contact_force=measured, commanded_contact_force=applied,
                         measured_contact_wrench_world=measured_wrench,
                         measured_contact_point_world=measured_point,
                         measured_contact_weight_n=measured_weight,
                         robot_surface_points_world=world_points,
                         contact_normal_world=normal, contact_active=active,
                         contact_gate_gap_m=gap,
                         contact_eligible=eligible, contact_engagement=engagement,
                         feedback_correction_n=correction, touching=touching, force_enabled=enabled)
        return command, feedforward

    def capture(self):
        if self.last is not None:
            self.logs.append({key: value.copy() for key, value in self.last.items()})

    def export(self):
        arrays = {f'controller_{key}': np.asarray([row[key] for row in self.logs]) for key in self.logs[0]} if self.logs else {}
        return dict(**arrays, controller_interface=self.config.interface,
                    controller_clipped_steps=self.clipped_steps,
                    controller_incompatible_limit_steps=self.incompatible_limit_steps,
                    controller_feedback_steps=self.feedback_steps,
                    controller_max_feedback_n=self.max_feedback,
                    controller_max_conversion_residual_nm=self.max_conversion_residual,
                    controller_max_offset_rate_rad_s=self.max_offset_rate,
                    controller_rate_contract_satisfied=bool(self.max_offset_rate <= self.config.offset_rate_rad_s+1e-8),
                    controller_max_contact_torque_step_nm=self.max_contact_command_step)
