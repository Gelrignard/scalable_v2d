"""Restricted single-shooting MPC through the actual supported-robot simulator.

One objective couples command, robot dynamics, emergent contacts and object
motion. The staged reference is a warm start and a restricted contact schedule,
not a separate future force solver. This is an experimental CPU benchmark.
"""
from copy import copy
from dataclasses import dataclass
from time import perf_counter
import json
import mujoco
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from ..dynamics.execution_step import requested_torque


@dataclass(frozen=True)
class MPCSettings:
    horizon_s: float = .75
    period_s: float = .5
    samples: int = 6
    seed: int = 42
    palm_bound_m: float = .02
    joint_offset_bound_rad: float = .15
    force_scale_bound: float = .5
    disabled: bool = False
    verify_prediction: bool = True
    progress_path: str | None = None

    def __post_init__(self):
        if not 0 < self.period_s <= self.horizon_s or self.samples < 0:
            raise ValueError('Require a positive prefix no longer than the horizon')
        values=[self.horizon_s,self.period_s,self.palm_bound_m,self.joint_offset_bound_rad,self.force_scale_bound]
        if not np.isfinite(values).all() or min(values)<=0 or self.force_scale_bound>=1:
            raise ValueError('Require finite positive bounds and force scale bound below one')


def clone_controller(controller, scene):
    """Copy all mutable feedback memory; share only immutable reference data."""
    out = copy(controller)
    out.scene = scene
    out.feedback = copy(controller.feedback)
    out.feedback.filtered = controller.feedback.filtered.copy()
    out.feedback.correction = controller.feedback.correction.copy()
    out.engagement = copy(controller.engagement)
    out.engagement.gain = controller.engagement.gain.copy()
    out.offset = controller.offset.copy()
    out.previous_contact_torque = controller.previous_contact_torque.copy()
    out.last = None if controller.last is None else {k:v.copy() for k,v in controller.last.items()}
    out.logs = []
    return out


class ContactMPC:
    def __init__(self, scene, controller, settings, plan):
        self.scene, self.controller, self.cfg, self.plan = scene, controller, settings, plan
        self.rng = np.random.default_rng(settings.seed)
        self.dt = scene.model.opt.timestep
        self.period = max(1, round(settings.period_s/self.dt))
        self.horizon = max(self.period, round(settings.horizon_s/self.dt))
        self.side = np.array([0 if 'left' in scene.model.body(p.body).name else 1
                              for p in controller.points])
        self.start = 0
        self.curve = None
        self.rows = []
        self.pending = None
        self.max_prediction_error = 0.
        self.total_evaluations = 0
        self.sampled_offset = np.zeros_like(plan['commands'])
        self.sampled_scale = np.ones((len(plan['times']),2))
        self.last_target = np.zeros(len(scene.movable))

    def basis(self):
        s = self.scene
        basis = np.zeros((len(s.movable),6))
        for k, side in enumerate(('left','right')):
            jac = np.zeros((3,s.model.nv))
            mujoco.mj_jacBody(s.model,s.data,jac,None,s.model.body(side+'_hand_palm_link').id)
            cols = np.array([i for i,j in enumerate(s.movable)
                             if side in s.kin.joint_names[j] and not s.hand[i]])
            j = jac[:,s.active_v[cols]]
            basis[cols,3*k:3*k+3] = j.T@np.linalg.solve(j@j.T+1e-3*np.eye(3),np.eye(3))
        return basis

    def continuation(self, step):
        n = len(self.scene.movable)
        if self.curve is None:
            return np.zeros(n+2),np.zeros(n+2),np.zeros(n+2)
        t = min((step-self.start)*self.dt,self.cfg.horizon_s)
        return tuple(self.curve(t,d) for d in range(3))

    def make_curve(self, initial, target, duration):
        # Quintic interpolation matches command position/velocity/acceleration
        # across replans. Delta parameterization alone would not guarantee this.
        return BPoly.from_derivatives([0.,duration],
            [[initial[0],initial[1],initial[2]],
             [target,np.zeros_like(target),np.zeros_like(target)]])

    def commands(self, first, count, curve):
        p = self.plan
        indices = np.arange(first,first+count)
        t = np.arange(count)*self.dt
        y,dy,ddy = (curve(t,d) for d in range(3))
        n = len(self.scene.movable)
        scale = np.clip(1+y[:,n:],1-self.cfg.force_scale_bound,1+self.cfg.force_scale_bound)
        return (p['commands'][indices]+y[:,:n],
                p['velocities'][indices]+dy[:,:n],
                p['acceleration'][indices]+ddy[:,:n],
                p['force'][indices]*scale[:,self.side,None])

    def predict(self, first, curve, count):
        """A branch from current state; never mutates the executed simulation."""
        live = self.scene
        s = copy(live)
        s.data = copy(live.data)
        ctl = clone_controller(self.controller,s)
        q,dq,ddq,f = self.commands(first,count,curve)
        if np.max(abs(q-self.plan['commands'][first:first+count])) > self.cfg.joint_offset_bound_rad+1e-9:
            return np.inf,None
        plan = self.plan
        score, terms, prefix = 0.,0,None
        lower,upper = ctl.lower,ctl.upper
        for j in range(count):
            k = first+j
            raw,*_ = requested_torque(s,ctl,plan['times'][k],q[j],dq[j],ddq[j],f[j],
                plan['kp'],plan['kd'],inertia_ff=plan['inertia_ff'],
                inertia_scale=plan['inertia_scale'],fixture_a=plan['fixture_a'][k],
                fixture_alpha=plan['fixture_alpha'][k])
            s.data.qfrc_applied[:] = 0.
            s.data.qfrc_applied[s.active_v] = np.clip(raw,-s.caps,s.caps)
            s.data.mocap_pos[0] = plan['fixture_p'][k+1]
            s.data.mocap_quat[0] = plan['fixture_r'][k+1]
            mujoco.mj_step(s.model,s.data)
            if j+1 == min(self.period,count):
                prefix = (first+j+1,s.data.qpos.copy(),s.data.qvel.copy(),
                          ctl.feedback.filtered.copy(),ctl.feedback.correction.copy())
            if (j+1)%20 == 0 or j+1 == count:
                pos = (s.data.qpos[s.book_q:s.book_q+3]-plan['desired_p'][k+1])/.03
                rotation = (Rotation.from_quat(plan['desired_r'][k+1],scalar_first=True).inv()*
                    Rotation.from_quat(s.data.qpos[s.book_q+3:s.book_q+7],scalar_first=True)).as_rotvec()/.15
                angle=s.data.qpos[s.active_q]
                violation=np.maximum(np.maximum(lower-angle,angle-upper),0)/.03
                saturation=np.maximum(abs(raw)/s.caps-1,0)
                penetration=max([max(0.,-c.dist-.002) for c in s.data.contact],default=0.)/.005
                term = pos@pos+rotation@rotation+.05*np.mean(violation**2)+.01*np.mean(saturation**2)+penetration**2
                weight = 3. if j+1 == count else 1.
                score += weight*term; terms += weight
            if any(s.data.warning[int(w)].number for w in
                   (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
                return np.inf,None
        offset=q-plan['commands'][first:first+count]
        score=score/max(terms,1)+.02*np.mean((offset/.1)**2)+.002*np.mean((dq-plan['velocities'][first:first+count])**2)
        return float(score),prefix

    def replan(self, step):
        start = perf_counter()
        cfg = self.cfg
        count = min(self.horizon,len(self.plan['times'])-1-step)
        initial = self.continuation(step)
        basis = self.basis()
        n = len(self.scene.movable)
        nominal=np.r_[np.linalg.lstsq(basis,self.last_target,rcond=None)[0],
                      initial[0][n:]]
        bounds=np.r_[np.full(6,cfg.palm_bound_m),np.full(2,cfg.force_scale_bound)]
        nominal=np.clip(nominal,-bounds,bounds)
        # Reuse the previous endpoint in the updated local basis, or smoothly
        # return toward the reference. This is not a guaranteed feasible shift
        # of the previous solution; rejected candidates remain explicit.
        candidates=[nominal,np.zeros(8)]
        active=np.any(abs(self.plan['force'][step:step+count])>1e-5)
        if cfg.disabled or not active:
            candidates=[np.zeros(8)]
        else:
            candidates.extend(np.clip(nominal+self.rng.normal(size=(cfg.samples,8))*bounds*.5,-bounds,bounds))
        before_q=self.scene.data.qpos.copy()
        before_v=self.scene.data.qvel.copy()
        best=None; costs=[]
        for theta in candidates:
            target=np.r_[basis@theta[:6],theta[6:]]
            curve=self.make_curve(initial,target,count*self.dt)
            value,prefix=self.predict(step,curve,count)
            costs.append(value);self.total_evaluations+=1
            if best is None or value<best[0]:
                best=(value,curve,prefix,target)
        np.testing.assert_array_equal(self.scene.data.qpos,before_q)
        np.testing.assert_array_equal(self.scene.data.qvel,before_v)
        if best is None or not np.isfinite(best[0]) or best[2] is None:
            raise RuntimeError('All MPC candidates rejected; no silent controller fallback')
        self.curve,self.pending=best[1],best[2]
        self.last_target=best[3][:n].copy()
        self.start=step
        self.rows.append(dict(step=step,time_s=float(self.plan['times'][step]),
            evaluations=len(costs),nominal_cost=float(costs[0]) if np.isfinite(costs[0]) else None,
            selected_cost=best[0],
            rejected_candidates=sum(not np.isfinite(x) for x in costs),
            wall_seconds=perf_counter()-start,optimizer_converged=False,
            max_prediction_state_error=self.max_prediction_error))
        if cfg.progress_path:
            from pathlib import Path
            Path(cfg.progress_path).write_text(json.dumps(self.rows,indent=2)+'\n')
        print('mpc',json.dumps(self.rows[-1]),flush=True)

    def verify_prefix(self, step):
        if self.pending is not None and step == self.pending[0]:
            _,q,v,filtered,correction=self.pending
            errors=[np.max(abs(self.scene.data.qpos-q)),np.max(abs(self.scene.data.qvel-v)),
                    np.max(abs(self.controller.feedback.filtered-filtered)),
                    np.max(abs(self.controller.feedback.correction-correction))]
            self.max_prediction_error=max(self.max_prediction_error,float(max(errors)))
            if self.cfg.verify_prediction and max(errors)>1e-8:
                raise AssertionError(f'Prediction/execution mismatch: {errors}')
            self.pending=None

    def command(self, step):
        self.verify_prefix(step)
        if step%self.period == 0:
            self.replan(step)
        local=(step-self.start)*self.dt
        y,dy,ddy=(self.curve(local,d) for d in range(3))
        n=len(self.scene.movable);p=self.plan
        scales=np.clip(1+y[n:],1-self.cfg.force_scale_bound,1+self.cfg.force_scale_bound)
        self.sampled_offset[step]=y[:n];self.sampled_scale[step]=scales
        return (p['commands'][step]+y[:n],p['velocities'][step]+dy[:n],
                p['acceleration'][step]+ddy[:n],p['force'][step]*scales[self.side,None])

    def export(self, samples):
        # The rollout samples its final state before asking for another command.
        # Verify that last partial prefix and record the planned endpoint without
        # applying another physics step or launching an empty horizon.
        final=len(self.plan['times'])-1
        self.verify_prefix(final)
        y=self.curve(min((final-self.start)*self.dt,self.cfg.horizon_s))
        n=len(self.scene.movable)
        self.sampled_offset[final]=y[:n]
        self.sampled_scale[final]=np.clip(1+y[n:],1-self.cfg.force_scale_bound,1+self.cfg.force_scale_bound)
        return dict(mpc_command_offset_rad=self.sampled_offset[samples],
            mpc_force_scale=self.sampled_scale[samples],
            mpc_log_json=json.dumps(self.rows),mpc_evaluations=self.total_evaluations,
            mpc_max_prediction_state_error=self.max_prediction_error,
            mpc_scope='restricted single-shooting joint physical dynamics; fixed nominal grasp schedule',
            mpc_full_nonlinear_optimum_certified=False)
