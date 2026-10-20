"""Transport an accepted hand shape with the object; solve arm/waist IK only.

Unloaded fingers retain the collision-checked shape instead of reverting to
noisy retargeted finger joints. Object observations provide the path; the
selected demonstration-supported grasp provides object-relative palm poses.
"""
from time import perf_counter
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .surface_hypotheses import contact_kinematics


class RigidGrasp:
    def __init__(self,scene,points,q,position,wxyz):
        self.scene,self.points,self.grasp=scene,points,q.copy()
        self.bodies=[scene.model.body(side+'_hand_palm_link').id for side in ('left','right')]
        scene.set_state(q,position,wxyz)
        r=Rotation.from_quat(wxyz,scalar_first=True)
        self.local_p=r.inv().apply(scene.data.xpos[self.bodies]-position)
        self.local_r=[r.inv()*Rotation.from_matrix(scene.data.xmat[b].reshape(3,3)) for b in self.bodies]
        self.outward=np.array([np.mean([p.object_outward for p in points if side in scene.model.body(p.body).name],axis=0)
                              for side in ('left','right')])
        self.outward/=np.linalg.norm(self.outward,axis=1)[:,None]

    def solve(self,nominal,position,wxyz,previous,*,clearance=0.,lift=0.,seconds=.3,
              palm_targets=None, finger_q=None, arm_neighborhood=.9, step_dt=None):
        s=self.scene; start=perf_counter(); q=nominal.copy(); q[7+s.movable[s.hand]]=self.grasp[7+s.movable[s.hand]]
        if finger_q is not None:
            q[7+s.movable[s.hand]] = finger_q[7+s.movable[s.hand]]
        cols=np.flatnonzero(~s.hand); ids=7+s.movable[cols]; n=len(ids)
        q[ids]=previous[ids]
        r=Rotation.from_quat(wxyz,scalar_first=True)
        target_p=position+r.apply(self.local_p+clearance*self.outward)+[0.,0.,lift]
        target_r=[r*local for local in self.local_r]
        if palm_targets is not None:
            target_p, target_r = palm_targets
        lower=np.maximum(s.kin.robot.model.lowerPositionLimit[ids],nominal[ids]-arm_neighborhood)
        upper=np.minimum(s.kin.robot.model.upperPositionLimit[ids],nominal[ids]+arm_neighborhood)
        if step_dt is not None:
            if not np.isfinite(step_dt) or step_dt <= 0:
                raise ValueError('Continuity requires a positive elapsed interval')
            # The nominal posture is already a soft residual. It must not force
            # a branch switch when its local box moves away from the last state.
            speed = s.kin.robot.model.velocityLimit[6+s.movable[cols]]
            delta = .8*speed*step_dt
            lower = np.maximum(s.kin.robot.model.lowerPositionLimit[ids], previous[ids]-delta)
            upper = np.minimum(s.kin.robot.model.upperPositionLimit[ids], previous[ids]+delta)
        best={'cost':np.inf,'x':np.clip(q[ids],lower+1e-8,upper-1e-8)}; cache={}
        pairs=s.model.ngeom**2
        class Deadline(Exception):pass
        def evaluate(x):
            if 'x' in cache and np.array_equal(cache['x'],x):return cache['r'],cache['j']
            if perf_counter()-start>seconds:raise Deadline()
            q[ids]=x;s.set_state(q,position,wxyz)
            residual=[];jac=[]
            for i,body in enumerate(self.bodies):
                jp,jr=np.zeros((3,s.model.nv)),np.zeros((3,s.model.nv))
                mujoco.mj_jacBody(s.model,s.data,jp,jr,body)
                current=Rotation.from_matrix(s.data.xmat[body].reshape(3,3))
                residual.extend(150*(s.data.xpos[body]-target_p[i]));jac.extend(150*jp[:,s.active_v[cols]])
                residual.extend(10*(current*target_r[i].inv()).as_rotvec());jac.extend(10*jr[:,s.active_v[cols]])
            collision=np.zeros(pairs); jc=np.zeros((pairs,n))
            for row,c in zip(s.contacts(s.movable),s.data.contact):
                key=int(c.geom1)*s.model.ngeom+int(c.geom2); violation=max(0.,-.00025-c.dist)
                if violation>collision[key] and np.linalg.norm(row['jacobian'][cols])>1e-10:
                    collision[key]=violation;jc[key]=-row['jacobian'][cols]
            residual=np.r_[residual,.1*(x-nominal[ids]),.25*(x-previous[ids]),500*collision]
            jac=np.vstack((jac,.1*np.eye(n),.25*np.eye(n),500*jc))
            cost=float(residual@residual)
            if cost<best['cost']:best.update(cost=cost,x=x.copy())
            cache.update(x=x.copy(),r=residual,j=jac);return residual,jac
        status='budget'
        try:
            fit=least_squares(lambda x:evaluate(x)[0],best['x'],jac=lambda x:evaluate(x)[1],bounds=(lower,upper),
                max_nfev=45,ftol=1e-6,xtol=1e-6,gtol=1e-5)
            status=str(fit.message)
        except Deadline:pass
        q[ids]=best['x'];s.set_state(q,position,wxyz)
        actual,_,_,_=contact_kinematics(s,self.points)
        intended=position+r.apply(np.array([p.object_local for p in self.points]))
        return q,dict(status=status,wall_seconds=perf_counter()-start,
            max_gap_m=float(np.linalg.norm(actual-intended,axis=1).max()),
            max_palm_gap_m=float(np.linalg.norm(s.data.xpos[self.bodies]-target_p,axis=1).max()),
            max_robot_penetration_m=max([max(0.,-c.dist) for c in s.data.contact],default=0.),
            continuity_interval_s=step_dt,
            max_joint_step_rad=float(np.max(abs(q[ids]-previous[ids]))),
            fixed_finger_shape=True)
