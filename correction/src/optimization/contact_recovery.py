"""Observed-state staged contact recovery with accumulated smooth commands.

Geometry and force QPs propose a command; cloned physical predictions compare
it with the shifted previous command. No object state or force is overwritten.
"""
from copy import copy
from dataclasses import dataclass, asdict
from time import perf_counter
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .delta_command import command_curve, held_value, solve_endpoint, bernstein_bounds
from .contact_mpc import clone_controller
from .surface_hypotheses import skew
from ..dynamics.observed_contact import ContactCharts, observe_patches, allocate_observed
from ..dynamics.execution_step import requested_torque
from ..dynamics.environment_contact import environment_contacts


@dataclass(frozen=True)
class RecoverySettings:
    period_s: float = .15
    horizon_s: float = .45
    acquisition_lead_s: float = .3
    arm_speed: float = 1.5
    hand_speed: float = 2.5
    arm_acceleration: float = 12.
    hand_acceleration: float = 20.
    force_rate_n_s: float = 60.
    task_step_m: float = .015
    task_step_rad: float = .06
    object_patch_radius_m: float = .08
    robot_patch_radius_m: float = .04
    progress_path: str | None = None

    def __post_init__(self):
        values = [v for k,v in asdict(self).items() if k != 'progress_path']
        if not np.isfinite(values).all() or min(values) <= 0 or self.period_s > self.horizon_s:
            raise ValueError('Recovery settings must be positive, with prefix <= horizon')


class CommandSegment:
    def __init__(self, curve, start, force0, force1, duration):
        self.curve, self.start = curve, start
        self.force0, self.force1, self.duration = force0.copy(), force1.copy(), duration

    def at(self, time):
        local = max(0., time-self.start)
        x = np.clip(local/self.duration, 0., 1.)
        blend = x**3*(10-15*x+6*x*x)
        return (*[held_value(self.curve, local, d) for d in range(3)],
                self.force0+blend*(self.force1-self.force0))


class ContactRecovery:
    def __init__(self, scene, controller, settings, plan):
        self.scene, self.controller, self.cfg, self.plan = scene, controller, settings, plan
        self.dt = scene.model.opt.timestep
        self.period = round(settings.period_s/self.dt)
        self.horizon = round(settings.horizon_s/self.dt)
        self.charts = ContactCharts(scene, controller.points)
        n = len(scene.movable)
        initial = (plan['commands'][0].copy(), np.zeros(n), np.zeros(n))
        self.segment = CommandSegment(command_curve(initial, initial[0], settings.horizon_s),
                    0., np.zeros((len(controller.points),3)), np.zeros((len(controller.points),3)), settings.horizon_s)
        self.speed = np.minimum(np.where(scene.hand, settings.hand_speed, settings.arm_speed),
                               .8*scene.kin.robot.model.velocityLimit[6+scene.movable])
        self.acceleration = np.where(scene.hand, settings.hand_acceleration, settings.arm_acceleration)
        self.rows = []; self.pending = None; self.max_prediction_error = 0.
        self.mode = None; self.held_chart = None; self.support_duration = 0.
        self.prior_chart = None
        self.logs = {k: [] for k in ('command', 'velocity', 'acceleration', 'force', 'gap',
                                    'touching', 'active', 'object_local', 'robot_local', 'inward')}
        self.last_observation = None

    def current_chart(self, step):
        ctl, s, p, cfg = self.controller, self.scene, self.plan, self.cfg
        time = p['times'][step]
        frame = min(np.searchsorted(ctl.times,time,side='right')-1,len(ctl.times)-1)
        future = min(np.searchsorted(ctl.times,time+cfg.acquisition_lead_s),len(ctl.times)-1)
        candidates = np.flatnonzero(np.any(ctl.active[frame:future+1],axis=1))
        chosen = frame+int(candidates[0]) if len(candidates) else frame
        active = ctl.active[chosen].copy()
        # Release only after measured environment support and slow motion have
        # persisted. No episode-specific frame labels or object resets.
        environment_force = np.zeros(3)
        for ci,c in enumerate(s.data.contact):
            first = c.geom1 in s.object_geoms
            other = c.geom2 if first else c.geom1
            if not (first or c.geom2 in s.object_geoms) or other not in s.environment: continue
            w=np.zeros(6); mujoco.mj_contactForce(s.model,s.data,ci,w)
            environment_force += (-1 if first else 1)*np.asarray(c.frame).reshape(3,3).T@w[:3]
        supported = (environment_force[2] >= .7*s.model.body_mass[s.book_body]*9.81 and
                     np.linalg.norm(s.data.qvel[s.book_v:s.book_v+3]) < .05 and
                     np.linalg.norm(s.data.qvel[s.book_v+3:s.book_v+6]) < .2)
        self.support_duration = self.support_duration+cfg.period_s if supported else 0.
        if not np.any(active) and self.mode is not None and np.any(self.mode) and self.support_duration < .15:
            return dict(self.held_chart, active=self.mode.copy()), 'support_pending'
        if not np.any(active) and self.held_chart is not None:
            self.mode=active.copy()
            return dict(self.held_chart,active=active), 'free'
        chart = self.charts.get(ctl.local_points[chosen],ctl.object_points[chosen],ctl.normals[chosen])
        if chart is self.prior_chart and self.held_chart is not None:
            chart = self.held_chart
        else:
            self.prior_chart = chart
        self.mode, self.held_chart = active.copy(), chart
        return dict(chart,active=active), ('acquire_or_manipulate' if np.any(active) else 'free')

    def propose(self, step, chart, obs, current):
        s,p,cfg = self.scene,self.plan,self.cfg
        end = min(step+self.horizon,len(p['times'])-1)
        h = cfg.horizon_s
        actual = s.data.qpos[s.active_q]
        qtarget = p['commands'][end].copy()
        rows,values = [],[]
        active = chart['active']
        r = Rotation.from_quat(s.data.qpos[s.book_q+3:s.book_q+7],scalar_first=True)
        delta_p = p['desired_p'][end]-s.data.qpos[s.book_q:s.book_q+3]
        delta_p *= min(1.,cfg.task_step_m/max(np.linalg.norm(delta_p),1e-12))
        delta_r = (Rotation.from_quat(p['desired_r'][end],scalar_first=True)*r.inv()).as_rotvec()
        delta_r *= min(1.,cfg.task_step_rad/max(np.linalg.norm(delta_r),1e-12))
        for i in np.flatnonzero(active):
            target = obs['anchors'][i]+.001*obs['inward'][i]
            if obs['touching'][i]:
                target += delta_p+np.cross(delta_r,obs['anchors'][i]-s.data.xipos[s.book_body])
            # Distant patches get an approach objective, not an invented load.
            rows.append(70*obs['jac'][i]); values.append(70*(target-obs['positions'][i]+obs['jac'][i]@actual))
            normal_jac = -skew(obs['normals'][i])@obs['jac_rot'][i]
            rows.append(1.5*normal_jac)
            values.append(1.5*(obs['inward'][i]-obs['normals'][i]+normal_jac@actual))
        # Linearized separation from all currently penetrating robot contacts.
        for contact in s.data.contact:
            if contact.dist >= -.0005: continue
            b1,b2 = (int(s.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2))
            j1,j2 = np.zeros((3,s.model.nv)),np.zeros((3,s.model.nv))
            mujoco.mj_jac(s.model,s.data,j1,None,contact.pos,b1)
            mujoco.mj_jac(s.model,s.data,j2,None,contact.pos,b2)
            j = np.asarray(contact.frame).reshape(3,3)[0]@(j2-j1)[:,s.active_v]
            if np.linalg.norm(j)>1e-10:
                rows.append(150*j[None,:]);values.append(np.array([150*(-.0005-contact.dist+j@actual)]))
        # Anti-windup preference uses measured tracking and available torque.
        # Physical prediction, not this approximate penalty, decides execution.
        tracking = np.diag(.15*p['kp']/s.caps)
        rows.append(tracking);values.append(tracking@(actual+cfg.period_s*s.data.qvel[s.active_v]))
        curve,audit = solve_endpoint(current[:3],h,qtarget,rows,values,
                    self.controller.lower,self.controller.upper,self.speed,self.acceleration)
        if curve is None: return None,dict(command=audit)
        force,force_audit = allocate_observed(s,obs,active,p['desired_p'][end],p['desired_r'][end],
                         current[0],current[1],p['kp'],p['kd'],current[3],cfg.period_s,
                         normal_cap=self.controller.config.normal_limit_n,friction=self.controller.config.friction)
        # Quintic force interpolation has peak slope 1.875/duration.
        force = current[3]+np.clip(force-current[3],-cfg.force_rate_n_s*h/1.875,cfg.force_rate_n_s*h/1.875)
        return CommandSegment(curve,p['times'][step],current[3],force,h),dict(command=audit,force=force_audit)

    def predict(self, step, segment, chart):
        live=self.scene;s=copy(live);s.data=copy(live.data)
        ctl=clone_controller(self.controller,s);p=self.plan
        ctl.reference_override=chart
        count=min(self.horizon,len(p['times'])-1-step)
        score=0.;terms=0;peak=0.;penetration=0.;prefix=None
        for j in range(count):
            k=step+j
            q,v,a,f=segment.at(p['times'][k])
            raw,*_=requested_torque(s,ctl,p['times'][k],q,v,a,f,p['kp'],p['kd'],
                inertia_ff=p['inertia_ff'],inertia_scale=p['inertia_scale'],
                fixture_a=p['fixture_a'][k],fixture_alpha=p['fixture_alpha'][k])
            peak=max(peak,float(np.max(abs(raw)/s.caps)))
            s.data.qfrc_applied[:]=0.;s.data.qfrc_applied[s.active_v]=np.clip(raw,-s.caps,s.caps)
            s.data.mocap_pos[0]=p['fixture_p'][k+1];s.data.mocap_quat[0]=p['fixture_r'][k+1]
            mujoco.mj_step(s.model,s.data)
            penetration=max(penetration,max([max(0.,-c.dist) for c in s.data.contact],default=0.))
            if j+1==min(count,self.period):
                prefix=(k+1,s.data.qpos.copy(),s.data.qvel.copy(),ctl.feedback.filtered.copy(),ctl.feedback.correction.copy())
            if (j+1)%30==0 or j+1==count:
                e=(s.data.qpos[s.book_q:s.book_q+3]-p['desired_p'][k+1])/.05
                er=(Rotation.from_quat(p['desired_r'][k+1],scalar_first=True).inv()*
                    Rotation.from_quat(s.data.qpos[s.book_q+3:s.book_q+7],scalar_first=True)).as_rotvec()/.2
                observed=observe_patches(s,ctl.points,chart)
                selected=chart['active']
                contact=(np.mean((np.minimum(observed['gap'][selected],.3)/.025)**2)+
                         np.mean((1-observed['normal_alignment'][selected])**2)) if np.any(selected) else 0.
                saturation=np.mean(np.maximum(abs(raw)/s.caps-1,0)**2)
                limit=np.maximum(np.maximum(ctl.lower-s.data.qpos[s.active_q],s.data.qpos[s.active_q]-ctl.upper),0)
                nominal=np.mean(((q-p['commands'][k])/.1)**2)
                weight=3. if j+1==count else 1.
                score+=weight*(e@e+er@er+contact+(1. if not np.any(selected) else .03)*nominal+
                               2*saturation+np.sum((limit/.01)**2)+(penetration/.005)**2)
                terms+=weight
            if any(s.data.warning[int(w)].number for w in
                   (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
                return dict(cost=None,peak_torque_fraction=peak,penetration_m=penetration,valid=False),None
        return dict(cost=float(score/max(terms,1)),peak_torque_fraction=peak,penetration_m=penetration,valid=True),prefix

    def verify(self, step):
        if self.pending is None or self.pending[0]!=step:return
        _,q,v,filtered,correction=self.pending
        error=max(np.max(abs(self.scene.data.qpos-q)),np.max(abs(self.scene.data.qvel-v)),
                  np.max(abs(self.controller.feedback.filtered-filtered)),
                  np.max(abs(self.controller.feedback.correction-correction)))
        self.max_prediction_error=max(self.max_prediction_error,float(error))
        if error>1e-8:raise AssertionError('Recovery prediction does not reproduce physical execution')
        self.pending=None

    def replan(self, step):
        start=perf_counter();p=self.plan;time=p['times'][step]
        before_q=self.scene.data.qpos.copy();before_v=self.scene.data.qvel.copy()
        chart,mode=self.current_chart(step)
        self.controller.reference_override=chart
        obs=observe_patches(self.scene,self.controller.points,chart)
        current=self.segment.at(time)
        proposal_chart,patch_changes=self.charts.propose_nearby(chart,self.prior_chart,chart['active'],
            object_radius=self.cfg.object_patch_radius_m,robot_radius=self.cfg.robot_patch_radius_m)
        proposal_observation=observe_patches(self.scene,self.controller.points,proposal_chart)
        candidate,audit=self.propose(step,proposal_chart,proposal_observation,current)
        warm,prefix=self.predict(step,self.segment,chart)
        selected='continue_previous';proposal=None
        if not warm['valid']:raise RuntimeError('Shifted command prediction unstable; no hidden reset')
        if candidate is not None:
            proposal,new_prefix=self.predict(step,candidate,proposal_chart)
            if (proposal['valid'] and proposal['cost']<warm['cost'] and
                proposal['peak_torque_fraction']<=max(1.,warm['peak_torque_fraction'])+.15 and
                proposal['penetration_m']<=max(.003,warm['penetration_m'])+.0005):
                self.segment=candidate;prefix=new_prefix;selected='observed_geometry_force_qp'
                chart=proposal_chart
                self.held_chart=chart
                self.controller.reference_override=chart
        self.pending=prefix
        np.testing.assert_array_equal(self.scene.data.qpos,before_q)
        np.testing.assert_array_equal(self.scene.data.qvel,before_v)
        self.rows.append(dict(time_s=float(time),mode=mode,selected=selected,active=chart['active'].tolist(),
            touching=obs['touching'].tolist(),gap_m=obs['gap'].tolist(),signed_distance_m=obs['signed_distance'].tolist(),
            command_initial_error_rad=float(np.max(abs(current[0]-before_q[self.scene.active_q]))),
            patch_proposals=patch_changes,patch_changes_accepted=bool(patch_changes and selected!='continue_previous'),
            warm=warm,proposal=proposal,optimization=audit,wall_seconds=perf_counter()-start))
        if self.cfg.progress_path:
            from pathlib import Path
            Path(self.cfg.progress_path).write_text(json.dumps(self.rows,indent=2)+'\n')
        if len(self.rows)%10==1:
            print('recovery',json.dumps({k:self.rows[-1][k] for k in ('time_s','mode','selected','wall_seconds')}),flush=True)

    def command(self,step):
        self.verify(step)
        if step%self.period==0:self.replan(step)
        values=self.segment.at(self.plan['times'][step])
        chart=self.controller.reference_override
        observed=observe_patches(self.scene,self.controller.points,chart) if step in self.plan['sample_steps'] else None
        if observed is not None:
            for key,value in zip(('command','velocity','acceleration','force'),values):self.logs[key].append(value.copy())
            for key,value in [('gap',observed['gap']),('touching',observed['touching']),('active',chart['active']),
                              ('object_local',chart['object_local']),('robot_local',chart['robot_local']),('inward',chart['inward'])]:
                self.logs[key].append(value.copy())
        return values

    def export(self,samples):
        final=len(self.plan['times'])-1;self.verify(final)
        values=self.segment.at(self.plan['times'][final]);chart=self.controller.reference_override
        observed=observe_patches(self.scene,self.controller.points,chart)
        for key,value in zip(('command','velocity','acceleration','force'),values):self.logs[key].append(value.copy())
        for key,value in [('gap',observed['gap']),('touching',observed['touching']),('active',chart['active']),
                          ('object_local',chart['object_local']),('robot_local',chart['robot_local']),('inward',chart['inward'])]:
            self.logs[key].append(value.copy())
        return dict(**{'recovery_'+k:np.asarray(v) for k,v in self.logs.items()},
            recovery_config=json.dumps(asdict(self.cfg)),recovery_log_json=json.dumps(self.rows),
            recovery_geometry_audit_json=json.dumps(self.charts.audit),
            recovery_max_prediction_state_error=self.max_prediction_error,
            recovery_scope='staged observed-state geometry/force proposals, bounded accumulated commands, physical prefix validation')
