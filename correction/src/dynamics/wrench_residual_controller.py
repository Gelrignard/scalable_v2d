"""Bounded object-pose/velocity wrench residual on a frozen nominal grasp.

The object state is read as a sensor, never overwritten. The QP chooses robot
contact forces; joint actuators still execute every force through actual
contact. This local privileged-state controller is not a submitted RL policy.
"""
from dataclasses import dataclass,asdict
import json
import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation,Slerp
from .contact_controller import G1ContactController
from .object_relative_controller import norm_clip
from .sparse_forces import contact_basis,cross_matrix


@dataclass(frozen=True)
class WrenchResidualConfig:
    period_s:float=.02
    position_kp:float=20.
    position_kd:float=6.
    rotation_kp:float=60.
    rotation_kd:float=10.
    force_limit_n:float=3.
    moment_limit_nm:float=.2
    contact_residual_limit_n:float=3.
    contact_residual_rate_n_s:float=40.


class WrenchResidualController(G1ContactController):
    def __init__(self,*args,object_positions,object_wxyz,residual_config,**kwargs):
        super().__init__(*args,**kwargs);self.residual_config=residual_config
        self.position=CubicSpline(self.times,savgol_filter(object_positions,21,3,axis=0))
        self.rotation=Slerp(self.times,Rotation.from_quat(object_wxyz,scalar_first=True))
        self.next_update=-1.;self.residual=np.zeros((len(self.points),3));self.target=self.residual.copy()
        self.solves=self.failures=0;self.max_residual=self.max_rate=0.

    def step(self,time,q,velocity,planned_force,compensation):
        s=self.scene;m,d=s.model,s.data;cfg=self.residual_config
        k=min(max(0,np.searchsorted(self.times,time,side='right')-1),len(self.times)-2)
        active=self.active[k]&self.active[k+1]
        if time+1e-9>=self.next_update:
            self.next_update=time+cfg.period_s;self.target[:]=0.
            if np.any(active):
                rotation=Rotation.from_quat(d.qpos[s.book_q+3:s.book_q+7],scalar_first=True)
                desired=self.rotation(float(np.clip(time,self.times[0],self.times[-1])))
                points=np.array([d.xpos[p.body]+d.xmat[p.body].reshape(3,3)@p.local for p in self.points])
                jac=[]
                for p,world in zip(self.points,points):
                    j=np.zeros((3,m.nv));mujoco.mj_jac(m,d,j,None,world,p.body);jac.append(j[:,s.active_v])
                count=len(points);size=3*count;jt=np.asarray(jac).reshape(size,-1).T
                com=d.xipos[s.book_body];lever=rotation.apply(m.body_ipos[s.book_body])
                omega=rotation.apply(d.qvel[s.book_v+3:s.book_v+6]);vcom=d.qvel[s.book_v:s.book_v+3]+np.cross(omega,lever)
                desired_com=self.position(time)+desired.apply(m.body_ipos[s.book_body])
                delta_force=norm_clip(m.body_mass[s.book_body]*(cfg.position_kp*(desired_com-com)+
                    cfg.position_kd*(self.position(time,1)-vcom)),cfg.force_limit_n)
                lo=max(self.times[0],time-.1);hi=min(self.times[-1],time+.1)
                desired_omega=(self.rotation(hi)*self.rotation(lo).inv()).as_rotvec()/max(hi-lo,1e-8)
                ri=d.ximat[s.book_body].reshape(3,3);inertia=ri@np.diag(m.body_inertia[s.book_body])@ri.T
                delta_moment=norm_clip(inertia@(cfg.rotation_kp*(desired*rotation.inv()).as_rotvec()+
                    cfg.rotation_kd*(desired_omega-omega)),cfg.moment_limit_nm)
                grasp=np.column_stack([np.vstack((np.eye(3),cross_matrix(p-com))) for p in points])
                scale=np.array([10.,10.,10.,1.,1.,1.]);op=grasp/scale[:,None]
                goal=np.r_[delta_force,delta_moment]/scale
                base=planned_force.ravel();rows=[];lower=[];upper=[]
                normal=rotation.apply(self.normals[k])
                for i,n in enumerate(normal):
                    basis=contact_basis(n);block=np.zeros((5,size))
                    block[:,3*i:3*i+3]=np.array([[1,0,0]]+[[-.85*self.config.friction,a,b] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]])@basis.T
                    rows.extend(block);lower.extend(np.array([0.,-np.inf,-np.inf,-np.inf,-np.inf])-block@base)
                    upper.extend(np.array([self.config.normal_limit_n if active[i] else 0.,0.,0.,0.,0.])-block@base)
                bias=d.qfrc_bias[s.active_v]+jt@base
                rows.extend(jt);lower.extend(-s.caps-bias);upper.extend(s.caps-bias)
                rows.extend(np.eye(size));lower.extend(np.full(size,-cfg.contact_residual_limit_n));upper.extend(np.full(size,cfg.contact_residual_limit_n))
                hessian=op.T@op+.001*np.eye(size)
                problem=osqp.OSQP();problem.setup(P=sparse.csc_matrix(np.triu(2*hessian)),q=-2*(op.T@goal+.0005*self.residual.ravel()),
                    A=sparse.csc_matrix(np.asarray(rows)),l=np.asarray(lower),u=np.asarray(upper),verbose=False,
                    eps_abs=1e-5,eps_rel=1e-5,max_iter=1500,time_limit=.015,polishing=False)
                fit=problem.solve(raise_error=False);self.solves+=1
                if fit.info.status_val in (1,2) and fit.x is not None and np.isfinite(fit.x).all():
                    value=np.asarray(rows)@fit.x
                    if max(np.max(np.asarray(lower)-value),np.max(value-np.asarray(upper)))<=1e-3:
                        self.target=fit.x.reshape(count,3)
                    else:self.failures+=1
                else:self.failures+=1
        change=np.clip(self.target-self.residual,-cfg.contact_residual_rate_n_s*m.opt.timestep,cfg.contact_residual_rate_n_s*m.opt.timestep)
        self.residual+=change;self.max_residual=max(self.max_residual,float(abs(self.residual).max()))
        self.max_rate=max(self.max_rate,float(abs(change).max()/m.opt.timestep))
        result=super().step(time,q,velocity,planned_force+self.residual,compensation)
        self.last['wrench_force_residual']=self.residual.copy();return result

    def export(self):
        return dict(**super().export(),wrench_residual_config=json.dumps(asdict(self.residual_config)),
            wrench_residual_solves=self.solves,wrench_residual_failures=self.failures,
            wrench_residual_max_n=self.max_residual,wrench_residual_max_rate_n_s=self.max_rate)
