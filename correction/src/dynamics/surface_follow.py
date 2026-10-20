"""Bounded local arm adaptation to measured object-surface displacement.

This deliberately changes the feedback adapter, not the saved trajectory or
force allocation. Perfect simulated object pose is a bench sensor assumption.
It does not by itself drive the object toward the task or certify contact.
"""
from dataclasses import dataclass, asdict
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .contact_controller import G1ContactController
from .object_relative_controller import joint_offset
from ..optimization.surface_reference import smoothstep


@dataclass(frozen=True)
class SurfaceFollowConfig:
    frames: tuple = (660, 675, 725, 746)
    gain: float = .5
    surface_displacement_limit_m: float = .015
    capture_m: float = .03
    joint_offset_limit_rad: float = .08
    joint_speed_limit_rad_s: float = .4
    joint_acceleration_limit_rad_s2: float = 2.
    update_seconds: float = .02
    regularization: float = .025
    response_frequency_rad_s: float = 12.

    def __post_init__(self):
        if len(self.frames) != 4 or not all(isinstance(i, int) for i in self.frames) or not (
                0 <= self.frames[0] < self.frames[1] <= self.frames[2] < self.frames[3]):
            raise ValueError('Follow envelope needs ordered frame indices')
        values = [v for k, v in asdict(self).items() if k != 'frames']
        if not np.isfinite(values).all() or min(values) <= 0 or self.gain > 1:
            raise ValueError('Follow parameters must be positive/finite and gain <=1')


class BoundedOffset:
    def __init__(self, size, config):
        self.config = config
        self.position = np.zeros(size); self.velocity = np.zeros(size)
        self.maximum_speed = self.maximum_acceleration = 0.
        self.conflicts = 0

    def step(self, target, q, lower, upper, dt):
        c = self.config; old = self.position.copy(); old_v = self.velocity.copy()
        acceleration = np.clip(c.response_frequency_rad_s**2*(target-old)-
                               2*c.response_frequency_rad_s*old_v,
                               -c.joint_acceleration_limit_rad_s2, c.joint_acceleration_limit_rad_s2)
        velocity = np.clip(old_v+dt*acceleration, -c.joint_speed_limit_rad_s, c.joint_speed_limit_rad_s)
        candidate = np.clip(old+dt*velocity, -c.joint_offset_limit_rad, c.joint_offset_limit_rad)
        # Zero offset must reproduce a nominal command exactly. Only an offset
        # that pushes further past a limit is clipped; inherited numerical
        # residuals belong to the nominal command's separate audit.
        lo = np.minimum(lower-q, 0.); hi = np.maximum(upper-q, 0.)
        self.position = np.clip(candidate, lo, hi)
        self.velocity = (self.position-old)/dt
        speed = float(np.max(abs(self.velocity)))
        acc = float(np.max(abs(self.velocity-old_v))/dt)
        self.maximum_speed = max(self.maximum_speed, speed)
        self.maximum_acceleration = max(self.maximum_acceleration, acc)
        self.conflicts += int(speed > c.joint_speed_limit_rad_s+1e-8 or
                             acc > c.joint_acceleration_limit_rad_s2+1e-8)
        return self.position, self.velocity


def safe_follow_columns(positions, velocities, times, lower, upper, hand, offset_limit, first):
    """Conservative room for every bounded offset on the remaining curve.

    Each cubic Hermite segment lies in the convex hull of four Bezier control
    points. Using their extrema avoids treating clear video samples as proof
    that between-frame joint-limit margins are sufficient.
    """
    q, v, t = np.asarray(positions), np.asarray(velocities), np.asarray(times)
    h = np.diff(t)[first:, None]
    controls = np.stack([q[first:-1], q[first:-1]+h*v[first:-1]/3,
                         q[first+1:]-h*v[first+1:]/3, q[first+1:]])
    minimum = controls.min(axis=(0, 1)); maximum = controls.max(axis=(0, 1))
    margin = np.minimum(minimum-lower, upper-maximum)
    enabled = ~np.asarray(hand) & (margin >= offset_limit+1e-4)
    return enabled, margin


class SurfaceFollowController(G1ContactController):
    def __init__(self, *args, planning_poses, follow_config, nominal_positions, nominal_velocities, **kwargs):
        super().__init__(*args, **kwargs)
        self.follow_config = follow_config
        if self.object_points is None:
            raise ValueError('Surface follow needs explicit object-patch trajectories')
        if follow_config.frames[-1] >= len(self.times):
            raise ValueError('Follow envelope extends beyond the complete episode')
        self.plan_positions = np.asarray(planning_poses['object_positions'])
        quats = np.asarray(planning_poses['object_wxyz'])
        if self.plan_positions.shape != (len(self.times), 3) or quats.shape != (len(self.times), 4):
            raise ValueError('Planning poses must match the full source clock')
        self.plan_rotation = Slerp(self.times, Rotation.from_quat(quats, scalar_first=True))
        self.follow = BoundedOffset(len(self.kp), follow_config)
        self.columns, self.joint_margin = safe_follow_columns(nominal_positions, nominal_velocities,
            self.times, self.lower, self.upper, self.scene.hand,
            follow_config.joint_offset_limit_rad, follow_config.frames[0])
        if not np.any(self.columns):
            raise ValueError('No arm joints have enough continuous margin for bounded feedback')
        self.target_offset = np.zeros(len(self.kp)); self.next_update = -1.
        self.update_count = 0; self.maximum_offset = 0.

    def step(self, time, q, velocity, planned_force, compensation):
        s, c = self.scene, self.follow_config
        envelope_times = self.times[list(c.frames)]
        weight = smoothstep((time-envelope_times[0])/(envelope_times[1]-envelope_times[0]))
        weight *= 1-smoothstep((time-envelope_times[2])/(envelope_times[3]-envelope_times[2]))
        if time+1e-9 >= self.next_update:
            self.next_update = time+c.update_seconds
            self.target_offset[:] = 0.
            if weight > 0:
                k = min(max(0, np.searchsorted(self.times, time, side='right')-1), len(self.times)-2)
                u = np.clip((time-self.times[k])/(self.times[k+1]-self.times[k]), 0, 1)
                local = (1-u)*self.local_points[k]+u*self.local_points[k+1]
                anchors = (1-u)*self.object_points[k]+u*self.object_points[k+1]
                r = Rotation.from_quat(s.data.qpos[s.book_q+3:s.book_q+7], scalar_first=True)
                actual = s.data.qpos[s.book_q:s.book_q+3]+r.apply(anchors)
                plan = ((1-u)*self.plan_positions[k]+u*self.plan_positions[k+1] +
                        self.plan_rotation(float(time)).apply(anchors))
                surface = np.array([s.data.xpos[p.body]+s.data.xmat[p.body].reshape(3, 3)@x
                                    for p, x in zip(self.points, local)])
                enabled = (self.active[k] & self.active[k+1] &
                           (np.linalg.norm(surface-actual, axis=1) <= c.capture_m))
                if np.any(enabled):
                    delta = actual-plan
                    delta *= np.minimum(1., c.surface_displacement_limit_m/
                                        np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
                    jac = []
                    for point, position in zip(self.points, surface):
                        j = np.zeros((3, s.model.nv))
                        mujoco.mj_jac(s.model, s.data, j, None, position, point.body)
                        jac.append(j[:, s.active_v])
                    cols = self.columns
                    self.target_offset[cols] = joint_offset(np.asarray(jac)[enabled][:, :, cols],
                        weight*c.gain*delta[enabled], c.regularization, c.joint_offset_limit_rad)
                    self.update_count += 1
        offset, rate = self.follow.step(self.target_offset, q, self.lower, self.upper, s.model.opt.timestep)
        self.maximum_offset = max(self.maximum_offset, float(np.max(abs(offset))))
        command, torque = super().step(time, q+offset, velocity+rate, planned_force, compensation)
        self.last['surface_follow_offset'] = offset.copy()
        self.last['surface_follow_velocity'] = rate.copy()
        return command, torque

    def export(self):
        return dict(**super().export(), surface_follow_config=json.dumps(asdict(self.follow_config)),
            surface_follow_joint_columns=np.flatnonzero(self.columns),
            surface_follow_nominal_joint_margin_rad=self.joint_margin,
            controller_surface_follow_updates=self.update_count,
            controller_surface_follow_max_offset_rad=self.maximum_offset,
            controller_surface_follow_max_speed_rad_s=self.follow.maximum_speed,
            controller_surface_follow_max_acceleration_rad_s2=self.follow.maximum_acceleration,
            controller_surface_follow_bound_conflicts=self.follow.conflicts,
            controller_surface_follow_bounds_satisfied=self.follow.conflicts == 0)
