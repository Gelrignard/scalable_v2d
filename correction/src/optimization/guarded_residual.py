"""Predictive acceptance of bounded feedback around an unchanged nominal.

Two closed-loop branches share current physical/controller state. Neither is
assumed safe: continuation can inherit nominal violations. This is a finite
candidate supervisor, not a solved unrestricted contact trajectory optimizer.
"""
from copy import copy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contact_mpc import clone_controller
from ..dynamics.execution_step import requested_torque
from ..dynamics.contact_measurement import object_contact_wrench


@dataclass(frozen=True)
class GuardSettings:
    horizon_s: float = 1.2
    period_s: float = .3
    impulse_window_s: float = .15
    improvement_fraction: float = .01
    disabled: bool = False
    progress_path: str | None = None

    def __post_init__(self):
        values = [self.horizon_s, self.period_s, self.impulse_window_s, self.improvement_fraction]
        if not np.isfinite(values).all() or min(values) <= 0 or not self.period_s <= self.horizon_s or self.improvement_fraction >= 1:
            raise ValueError('Finite positive settings, prefix <= horizon, improvement <1 required')


def clone_follow(controller, scene):
    out = clone_controller(controller, scene)
    out.follow = copy(controller.follow)
    out.follow.position = controller.follow.position.copy()
    out.follow.velocity = controller.follow.velocity.copy()
    out.target_offset = controller.target_offset.copy()
    return out


def controller_state(controller):
    """Include residual and contact-controller memory in prefix validation."""
    return dict(filtered=controller.feedback.filtered.copy(), correction=controller.feedback.correction.copy(),
        engagement=controller.engagement.gain.copy(), offset=controller.offset.copy(),
        previous_torque=controller.previous_contact_torque.copy(),
        residual=controller.follow.position.copy(), rate=controller.follow.velocity.copy(),
        target=controller.target_offset.copy(), next_update=np.array(controller.next_update))


def adoption_reasons(nominal, proposal, improvement):
    if not proposal['valid']:
        return ['invalid_prediction']
    if not nominal['valid']:
        return ['nominal_prediction_invalid_no_certified_comparison']
    reasons = []
    if proposal['task_cost'] >= nominal['task_cost']*(1-improvement)-1e-9:
        reasons.append('insufficient_task_improvement')
    for key in ('terminal_cost', 'impulse_cost'):
        if proposal[key] > nominal[key]*1.02+.01:
            reasons.append(key+'_worsened')
    for key, floor, tolerance in [('peak_torque_fraction', 1., 1e-6),
                                 ('penetration_m', 1e-5, 1e-6), ('joint_violation_rad', 1e-5, 1e-6)]:
        if proposal[key] > max(floor, nominal[key])+tolerance:
            reasons.append(key+'_worsened')
    if proposal['bound_conflicts']:
        reasons.append('residual_bound_conflict')
    return reasons


class GuardedResidual:
    def __init__(self, scene, controller, settings, plan):
        self.scene, self.controller, self.cfg, self.plan = scene, controller, settings, plan
        self.dt = scene.model.opt.timestep
        self.period = max(1, round(settings.period_s/self.dt))
        self.horizon = max(self.period, round(settings.horizon_s/self.dt))
        self.window = max(1, round(settings.impulse_window_s/self.dt))
        self.rows = []; self.pending = None; self.max_prediction_error = 0.
        self.evaluations = 0
        self.modes = np.zeros(len(plan['times']), dtype=bool)
        controller.feedback_enabled = False
        # Smooth secant reference velocities, no per-frame observed acceleration.
        count = len(plan['times']); width = max(1, round(.1/self.dt))
        lo = np.maximum(np.arange(count)-width, 0); hi = np.minimum(np.arange(count)+width, count-1)
        duration = (hi-lo)*self.dt
        rotations = Rotation.from_quat(plan['desired_r'], scalar_first=True)
        centers = plan['desired_p']+rotations.apply(scene.model.body_ipos[scene.book_body])
        velocity = (centers[hi]-centers[lo])/duration[:, None]
        omega = (rotations[hi]*rotations[lo].inv()).as_rotvec()/duration[:, None]
        inertia_rotation = (rotations*Rotation.from_quat(scene.model.body_iquat[scene.book_body], scalar_first=True)).as_matrix()
        angular = np.einsum('nij,j,nkj,nk->ni', inertia_rotation,
                           scene.model.body_inertia[scene.book_body], inertia_rotation, omega)
        self.mass = scene.model.body_mass[scene.book_body]
        self.desired_momentum = np.column_stack([self.mass*velocity, angular])

    def momentum(self, scene):
        d, m = scene.data, scene.model
        rotation = Rotation.from_quat(d.qpos[scene.book_q+3:scene.book_q+7], scalar_first=True)
        omega = rotation.apply(d.qvel[scene.book_v+3:scene.book_v+6])
        lever = rotation.apply(m.body_ipos[scene.book_body])
        velocity = d.qvel[scene.book_v:scene.book_v+3]+np.cross(omega, lever)
        ri = (rotation*Rotation.from_quat(m.body_iquat[scene.book_body], scalar_first=True)).as_matrix()
        angular = ri@np.diag(m.body_inertia[scene.book_body])@ri.T@omega
        return np.r_[self.mass*velocity, angular]

    def advance(self, scene, controller, step):
        p = self.plan
        raw, *_ = requested_torque(scene, controller, p['times'][step], p['commands'][step],
            p['velocities'][step], p['acceleration'][step], p['force'][step], p['kp'], p['kd'],
            inertia_ff=p['inertia_ff'], inertia_scale=p['inertia_scale'],
            fixture_a=p['fixture_a'][step], fixture_alpha=p['fixture_alpha'][step], surface_follow=True)
        scene.data.qfrc_applied[:] = 0.
        scene.data.qfrc_applied[scene.active_v] = np.clip(raw, -scene.caps, scene.caps)
        scene.data.mocap_pos[0] = p['fixture_p'][step+1]
        scene.data.mocap_quat[0] = p['fixture_r'][step+1]
        mujoco.mj_step(scene.model, scene.data)
        return raw

    def predict(self, step, enabled):
        s = copy(self.scene); s.data = copy(self.scene.data)
        ctl = clone_follow(self.controller, s)
        ctl.feedback_enabled = enabled; ctl.next_update = self.plan['times'][step]
        count = min(self.horizon, len(self.plan['times'])-1-step)
        p = self.plan; prefix = None
        score = terms = peak = penetration = violation = 0.
        impulse_cost = []; impulse = np.zeros(6); window_first = step
        initial_momentum = self.momentum(s); conflicts = ctl.follow.conflicts
        task = 0.
        for j in range(count):
            k = step+j
            raw = self.advance(s, ctl, k)
            peak = max(peak, float(np.max(abs(raw)/s.caps)))
            penetration = max(penetration, max([max(0., -c.dist) for c in s.data.contact], default=0.))
            angles = s.data.qpos[s.active_q]
            violation = max(violation, float(np.maximum(np.maximum(ctl.lower-angles, angles-ctl.upper), 0.).max()))
            for ci, c in enumerate(s.data.contact):
                if (c.geom1 in s.object_geoms) != (c.geom2 in s.object_geoms):
                    impulse += self.dt*object_contact_wrench(s.model, s.data, ci, s.object_geoms, s.book_body)
            if (j+1)%self.window == 0 or j+1 == count:
                duration = (k+1-window_first)*self.dt
                wanted = self.desired_momentum[k+1]-initial_momentum
                wanted[:3] -= self.mass*s.model.opt.gravity*duration
                weight = max(self.mass*np.linalg.norm(s.model.opt.gravity), 1.)
                scale = duration*np.r_[np.full(3, weight), np.full(3, .1*weight)]
                impulse_cost.append(float(np.sum(((impulse-wanted)/scale)**2)))
                initial_momentum = self.momentum(s); impulse[:] = 0.; window_first = k+1
            if j+1 == min(count, self.period):
                prefix = (k+1, s.data.qpos.copy(), s.data.qvel.copy(), controller_state(ctl))
            if (j+1)%20 == 0 or j+1 == count:
                e = (s.data.qpos[s.book_q:s.book_q+3]-p['desired_p'][k+1])/.05
                er = (Rotation.from_quat(p['desired_r'][k+1], scalar_first=True).inv()*
                      Rotation.from_quat(s.data.qpos[s.book_q+3:s.book_q+7], scalar_first=True)).as_rotvec()/.2
                task = float(e@e+er@er)
                weight = 3. if j+1 == count else 1.
                score += weight*task; terms += weight
            if any(s.data.warning[int(w)].number for w in
                   (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)):
                return dict(valid=False), None
        return dict(valid=True, task_cost=score/max(terms, 1), terminal_cost=task,
            impulse_cost=float(np.mean(impulse_cost)), peak_torque_fraction=peak,
            penetration_m=penetration, joint_violation_rad=violation,
            bound_conflicts=ctl.follow.conflicts-conflicts), prefix

    def verify(self, step):
        if self.pending is None or step != self.pending[0]:
            return
        _, q, v, memory = self.pending
        current = controller_state(self.controller)
        error = max(float(np.max(abs(self.scene.data.qpos-q))), float(np.max(abs(self.scene.data.qvel-v))),
                    *[float(np.max(abs(current[key]-value))) for key, value in memory.items()])
        self.max_prediction_error = max(self.max_prediction_error, error)
        if error > 1e-8:
            raise AssertionError(f'Guard prediction/execution mismatch {error}')
        self.pending = None

    def replan(self, step):
        started = perf_counter(); ctl = self.controller; p = self.plan
        time = p['times'][step]
        first = max(0, np.searchsorted(ctl.times, time, side='right')-1)
        last = min(len(ctl.times), np.searchsorted(ctl.times, time+self.cfg.horizon_s)+1)
        active = np.any(ctl.active[first:last])
        remaining = max(np.max(abs(ctl.follow.position)), np.max(abs(ctl.follow.velocity)))
        if self.cfg.disabled or (not active and remaining < 1e-12):
            ctl.feedback_enabled = False
            return
        before_q = self.scene.data.qpos.copy(); before_v = self.scene.data.qvel.copy()
        before_memory = controller_state(ctl)
        nominal, prefix = self.predict(step, False)
        proposal, proposal_prefix = self.predict(step, True)
        self.evaluations += 2
        reasons = adoption_reasons(nominal, proposal, self.cfg.improvement_fraction)
        chosen = not reasons
        if not nominal['valid']:
            raise RuntimeError('Nominal continuation unstable; no hidden reset or certified fallback')
        ctl.feedback_enabled = chosen; ctl.next_update = time
        self.pending = proposal_prefix if chosen else prefix
        np.testing.assert_array_equal(self.scene.data.qpos, before_q)
        np.testing.assert_array_equal(self.scene.data.qvel, before_v)
        after_memory = controller_state(ctl)
        for key in before_memory:
            if key != 'next_update':
                np.testing.assert_array_equal(after_memory[key], before_memory[key])
        self.rows.append(dict(time_s=float(time), enabled=chosen, reasons=reasons,
            nominal=nominal, proposal=proposal, wall_seconds=perf_counter()-started))
        if self.cfg.progress_path:
            Path(self.cfg.progress_path).write_text(json.dumps(self.rows, indent=2)+'\n')
        if len(self.rows)%5 == 1:
            print('guard', json.dumps(self.rows[-1]), flush=True)

    def command(self, step):
        self.verify(step)
        if step%self.period == 0:
            self.replan(step)
        self.modes[step] = self.controller.feedback_enabled
        p = self.plan
        return p['commands'][step], p['velocities'][step], p['acceleration'][step], p['force'][step]

    def export(self, samples):
        self.verify(len(self.plan['times'])-1)
        self.modes[-1] = self.controller.feedback_enabled
        return dict(guard_enabled=self.modes[samples], guard_log_json=json.dumps(self.rows),
            guard_settings=json.dumps(asdict(self.cfg)), guard_evaluations=self.evaluations,
            guard_max_prediction_state_error=self.max_prediction_error,
            guard_scope='bounded staged feedback with relative prediction guards; no feasibility certificate')
