"""Experimental bounded object-relative servo and object-wrench feedback.

Consumes the current simulated object pose as a sensor. No pose is written and
no wrench is applied to the object. This is a local controller experiment,
not an official policy adapter or proof of reference feasibility.
"""
from dataclasses import dataclass, asdict
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from .contact_controller import G1ContactController
from ..optimization.wrench_screen import screen_wrench


@dataclass(frozen=True)
class ObjectFeedbackConfig:
    update_seconds: float = .02
    translation_limit_m: float = .08
    rotation_limit_rad: float = .25
    joint_offset_rad: float = .3
    joint_offset_rate_rad_s: float = 1.
    joint_regularization: float = .015
    object_kp: float = 20.
    object_kd: float = 6.
    acceleration_limit_m_s2: float = 3.
    angular_kp: float = 10.
    angular_kd: float = 4.
    angular_acceleration_limit_rad_s2: float = 5.
    allocate_wrench: bool = True
    normal_follow_only: bool = False
    preserve_finger_shape: bool = False


def norm_clip(value, maximum):
    value = np.asarray(value, float)
    return value*min(1., maximum/max(float(np.linalg.norm(value)),1e-12))


def joint_offset(jacobian, displacement, regularization, limit):
    jacobian = np.asarray(jacobian).reshape(-1,np.asarray(jacobian).shape[-1])
    return np.clip(np.linalg.solve(jacobian.T@jacobian+regularization**2*np.eye(jacobian.shape[1]),
                                  jacobian.T@np.asarray(displacement).ravel()),-limit,limit)


class ObjectRelativeContactController(G1ContactController):
    def __init__(self, *args, object_positions, object_wxyz, object_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_config=object_config
        window=min(31,len(self.times)//2*2-1)
        self.position=CubicSpline(self.times,savgol_filter(object_positions,window,3,axis=0))
        self.rotation=Slerp(self.times,Rotation.from_quat(object_wxyz,scalar_first=True))
        self.anchors=np.array([p.object_local for p in self.points])
        self.next_update=-1.; self.follow_offset=np.zeros(len(self.kp)); self.target_offset=self.follow_offset.copy()
        self.force_hold=np.zeros((len(self.points),3))
        self.allocation_count=self.allocation_failures=0; self.max_allocation_residual=0.
        self.last_offset=np.zeros(len(self.kp)); self.max_follow_rate=0.

    def step(self,time,q,velocity,planned_force,compensation):
        s,cfg=self.scene,self.object_config; m,d=s.model,s.data
        k=min(max(0,np.searchsorted(self.times,time,side='right')-1),len(self.times)-2)
        active=self.active[k]&self.active[k+1]
        if time+1e-9>=self.next_update:
            self.next_update=time+cfg.update_seconds
            self.target_offset[:]=0.;self.force_hold=planned_force.copy()
            if np.any(active):
                observed_r=Rotation.from_quat(d.qpos[s.book_q+3:s.book_q+7],scalar_first=True)
                intended_r=self.rotation(float(np.clip(time,self.times[0],self.times[-1])))
                actual_p=d.qpos[s.book_q:s.book_q+3]
                intended_p=self.position(time)
                shift=norm_clip(actual_p-intended_p,cfg.translation_limit_m)
                rot=Rotation.from_rotvec(norm_clip((observed_r*intended_r.inv()).as_rotvec(),cfg.rotation_limit_rad))
                intended_levers=intended_r.apply(self.anchors)
                delta=shift+rot.apply(intended_levers)-intended_levers
                position=np.array([d.xpos[p.body]+d.xmat[p.body].reshape(3,3)@p.local for p in self.points])
                jac=[]
                for point,world in zip(self.points,position):
                    j=np.zeros((3,m.nv));mujoco.mj_jac(m,d,j,None,world,point.body)
                    jac.append(j[:,s.active_v])
                jac=np.asarray(jac)
                if cfg.normal_follow_only:
                    normal=observed_r.apply(self.normals[k])
                    delta=normal*np.sum(delta*normal,axis=1)[:,None]
                cols=~s.hand if cfg.preserve_finger_shape else np.ones(len(q),bool)
                self.target_offset[cols]=joint_offset(jac[active][:,:,cols],delta[active],cfg.joint_regularization,cfg.joint_offset_rad)
                if cfg.allocate_wrench:
                    mass=m.body_mass[s.book_body]; lever=observed_r.apply(m.body_ipos[s.book_body])
                    omega=observed_r.apply(d.qvel[s.book_v+3:s.book_v+6])
                    com_velocity=d.qvel[s.book_v:s.book_v+3]+np.cross(omega,lever)
                    desired_com=intended_p+intended_r.apply(m.body_ipos[s.book_body])
                    acceleration=norm_clip(cfg.object_kp*(desired_com-d.xipos[s.book_body])+
                        cfg.object_kd*(self.position(time,1)-com_velocity),cfg.acceleration_limit_m_s2)
                    angular=norm_clip(cfg.angular_kp*(intended_r*observed_r.inv()).as_rotvec()-cfg.angular_kd*omega,
                                      cfg.angular_acceleration_limit_rad_s2)
                    inertia_r=d.ximat[s.book_body].reshape(3,3)
                    inertia=inertia_r@np.diag(m.body_inertia[s.book_body])@inertia_r.T
                    wrench=np.r_[mass*(acceleration-m.opt.gravity),inertia@angular+np.cross(omega,inertia@omega)]
                    inward=observed_r.apply(self.normals[k])
                    allocation=screen_wrench(position,inward,d.xipos[s.book_body],wrench,jac,
                        d.qfrc_bias[s.active_v],s.caps,friction=self.config.friction,normal_caps=self.config.normal_limit_n,seconds=.05)
                    self.allocation_count+=1
                    if allocation['solved']:
                        self.force_hold=np.asarray(allocation['forces'])
                        self.max_allocation_residual=max(self.max_allocation_residual,allocation['force_residual_n'])
                    else:self.allocation_failures+=1
        dt=m.opt.timestep
        self.follow_offset+=np.clip(self.target_offset-self.follow_offset,
            -cfg.joint_offset_rate_rad_s*dt,cfg.joint_offset_rate_rad_s*dt)
        self.follow_offset=np.clip(q+self.follow_offset,self.lower,self.upper)-q
        self.max_follow_rate=max(self.max_follow_rate,float(np.max(abs(self.follow_offset-self.last_offset))/dt))
        self.last_offset=self.follow_offset.copy()
        command,feedforward=super().step(time,q+self.follow_offset,velocity,self.force_hold if cfg.allocate_wrench else planned_force,compensation)
        self.last['object_follow_offset']=self.follow_offset.copy()
        return command,feedforward

    def export(self):
        return dict(**super().export(),object_feedback_config=__import__('json').dumps(asdict(self.object_config)),
            object_wrench_allocation_count=self.allocation_count,object_wrench_allocation_failures=self.allocation_failures,
            object_wrench_max_force_residual_n=self.max_allocation_residual,object_follow_max_rate_rad_s=self.max_follow_rate)
