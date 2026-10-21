"""Observed-state force feedback around the preserved staged controller.

Motion, contact identities and phase clock stay nominal. Only a bounded force
residual is allocated at actual matched contacts. No predictive simulation.
"""
from dataclasses import dataclass, asdict
import json

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from .contact_controller import G1ContactController
from .object_relative_controller import norm_clip
from ..optimization.observed_force_update import force_update


@dataclass(frozen=True)
class StagedObservationConfig:
    gain: float = 1.
    position_kp: float = 8.
    position_kd: float = 3.
    rotation_kp: float = 12.
    rotation_kd: float = 3.
    force_limit_n: float = 3.
    moment_limit_nm: float = .15
    contact_delta_limit_n: float = .75
    contact_delta_rate_n_s: float = 10.

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Feedback settings must be finite and nonnegative')


class StagedObservationController(G1ContactController):
    def __init__(self, *args, observed_config, planning_poses, kd, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.interface != 'torque':
            raise ValueError('Observed force-stage currently requires the torque interface')
        self.observed = observed_config; self.kd = np.asarray(kd)
        # mj_step leaves solved-contact geometry at the previous integration
        # state. Refresh only kinematics in separate data; do not recompute
        # contacts or alter the inherited controller's sensing/physical state.
        self.kinematic_data = mujoco.MjData(self.scene.model)
        # Smooth positions, use twist only; never differentiate observations to
        # infer required per-frame object accelerations.
        window = min(21, len(self.times)//2*2-1)
        p = planning_poses['object_positions']
        self.position = CubicSpline(self.times, savgol_filter(p, window, min(3, window-1), axis=0))
        self.rotation = Slerp(self.times, Rotation.from_quat(planning_poses['object_wxyz'], scalar_first=True))
        self.delta = np.zeros((len(self.points), 3))
        self.solves = self.failures = self.conflicts = self.no_contact_steps = 0
        self.solve_seconds = self.max_delta = self.max_rate = self.max_violation = 0.
        self.events = []

    def desired_delta_wrench(self, time):
        s, cfg = self.scene, self.observed
        m, d = s.model, s.data
        time = float(np.clip(time, self.times[0], self.times[-1]))
        rotation = Rotation.from_quat(d.qpos[s.book_q+3:s.book_q+7], scalar_first=True)
        desired = self.rotation(time)
        lo, hi = max(self.times[0], time-.1), min(self.times[-1], time+.1)
        desired_omega = (self.rotation(hi)*self.rotation(lo).inv()).as_rotvec()/max(hi-lo, 1e-8)
        omega = rotation.apply(d.qvel[s.book_v+3:s.book_v+6])
        offset = rotation.apply(m.body_ipos[s.book_body])
        velocity = d.qvel[s.book_v:s.book_v+3]+np.cross(omega, offset)
        desired_offset = desired.apply(m.body_ipos[s.book_body])
        desired_velocity = self.position(time, 1)+np.cross(desired_omega, desired_offset)
        error = self.position(time)+desired_offset-(d.qpos[s.book_q:s.book_q+3]+offset)
        force = norm_clip(cfg.gain*m.body_mass[s.book_body]*(cfg.position_kp*error+
            cfg.position_kd*(desired_velocity-velocity)), cfg.force_limit_n)
        ri = (rotation*Rotation.from_quat(m.body_iquat[s.book_body], scalar_first=True)).as_matrix()
        inertia = ri@np.diag(m.body_inertia[s.book_body])@ri.T
        moment = norm_clip(cfg.gain*inertia@(cfg.rotation_kp*(desired*rotation.inv()).as_rotvec()+
            cfg.rotation_kd*(desired_omega-omega)), cfg.moment_limit_nm)
        return np.r_[force, moment]

    def physical_patches(self):
        """Actual force-weighted contact points/normals, same sensor match as base."""
        s, d = self.scene, self.scene.data
        kd = self.kinematic_data
        kd.qpos[:] = d.qpos
        kd.mocap_pos[:] = d.mocap_pos; kd.mocap_quat[:] = d.mocap_quat
        mujoco.mj_kinematics(s.model, kd); mujoco.mj_comPos(s.model, kd)
        points = self.last['robot_surface_points_world'].copy()
        matched = self.last['touching'] & self.last['contact_active']
        for i, point in enumerate(self.points):
            local = d.xmat[point.body].reshape(3, 3).T@(points[i]-d.xpos[point.body])
            points[i] = kd.xpos[point.body]+kd.xmat[point.body].reshape(3, 3)@local
        points[matched] = self.last['measured_contact_point_world'][matched]
        normals = np.zeros_like(points)
        for ci, contact in enumerate(d.contact):
            if not s.object_geoms.intersection((contact.geom1, contact.geom2)):
                continue
            first = contact.geom1 in s.object_geoms
            other = contact.geom2 if first else contact.geom1
            i = self.body_index.get(int(s.model.geom_bodyid[other]))
            if i is None or not matched[i]:
                continue
            normal = (-1 if first else 1)*np.asarray(contact.frame).reshape(3, 3)[0]
            if normal@self.last['contact_normal_world'][i] < .5 or np.linalg.norm(
                    contact.pos-self.last['robot_surface_points_world'][i]) > .03:
                continue
            force = np.zeros(6); mujoco.mj_contactForce(s.model, d, ci, force)
            normals[i] += max(0., force[0])*normal
        lengths = np.linalg.norm(normals, axis=1)
        matched &= lengths > 1e-8
        normals /= np.maximum(lengths[:, None], 1e-12)
        jac = []
        for point, world in zip(self.points, points):
            j = np.zeros((3, s.model.nv)); mujoco.mj_jac(s.model, kd, j, None, world, point.body)
            jac.append(j[:, s.active_v])
        return points, normals, matched, np.asarray(jac)

    def step(self, time, q, velocity, planned_force, compensation):
        command, feedforward = super().step(time, q, velocity, planned_force, compensation)
        s, cfg, dt = self.scene, self.observed, self.scene.model.opt.timestep
        goal = np.zeros(6); allocated = goal.copy(); matched = np.zeros(len(self.points), bool)
        old = self.delta.copy(); torque = np.zeros(len(q))
        if cfg.gain > 0 and (np.any(self.last['contact_active']) or np.any(abs(old) > 1e-10)):
            points, normals, matched, jac = self.physical_patches()
            self.no_contact_steps += int(not np.any(matched))
            goal = self.desired_delta_wrench(time)
            nominal_torque = self.kp*(command-s.data.qpos[s.active_q])+self.kd*(velocity-s.data.qvel[s.active_v])+feedforward
            self.delta, audit = force_update(points=points, normals=normals, touching=matched,
                com=self.kinematic_data.xipos[s.book_body], jacobians=jac,
                nominal_force=self.last['commanded_contact_force'], nominal_torque=nominal_torque,
                caps=s.caps, joint_position=s.data.qpos[s.active_q], lower=self.lower, upper=self.upper,
                goal=goal, previous=old, dt=dt, limit=cfg.contact_delta_limit_n,
                rate=cfg.contact_delta_rate_n_s, friction=self.config.friction, normal_cap=self.config.normal_limit_n)
            torque = jac.reshape(3*len(self.points), -1).T@self.delta.ravel()
            allocated = audit['delta_wrench']
            self.solves += 1; self.solve_seconds += audit['seconds']
            self.failures += int(audit['failed']); self.conflicts += int(audit['rate_conflict'])
            self.max_violation = max(self.max_violation, audit['static_violation'])
            if (audit['failed'] or audit['rate_conflict']) and len(self.events) < 20:
                self.events.append(dict(time_s=time, status=audit['status'],
                    rate_status=audit['rate_status'], rate_conflict=audit['rate_conflict']))
        self.max_delta = max(self.max_delta, float(abs(self.delta).max()))
        self.max_rate = max(self.max_rate, float(abs(self.delta-old).max()/dt))
        # Residual torque only acts through robot joints. A decaying command on
        # a separated body is not claimed as a realized object contact force.
        feedforward = feedforward+torque
        self.last['feedforward_torque'] = feedforward.copy()
        self.last['contact_torque'] = self.last['contact_torque']+torque
        self.last.update(observed_force_delta=self.delta.copy(), observed_delta_torque=torque,
            observed_matched_contact=matched, observed_goal_wrench=goal,
            observed_allocated_delta_wrench=allocated,
            observed_unmet_delta_wrench=goal-allocated)
        return command, feedforward

    def export(self):
        return dict(**super().export(), staged_observation_config=json.dumps(asdict(self.observed)),
            staged_observation_solves=self.solves, staged_observation_failures=self.failures,
            staged_observation_rate_conflicts=self.conflicts, staged_observation_no_contact_steps=self.no_contact_steps,
            staged_observation_solver_seconds=self.solve_seconds, staged_observation_max_delta_n=self.max_delta,
            staged_observation_max_rate_n_s=self.max_rate, staged_observation_max_static_violation=self.max_violation,
            staged_observation_events_json=json.dumps(self.events))
