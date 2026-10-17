"""Load-weighted local projection of a saved robot reference onto contact targets.

Only arm and hand joints can change. Base, legs, waist, object and timing remain
identical. This is a kinematic projection, not inverse dynamics or a rollout.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.book_reference import load_book_reference
import pinocchio as pin
import pink
from pink.limits import Limit, ConfigurationLimit
from pink.tasks import Task, PostureTask
from v2d_retarget.dataprocessing import HumanMotion
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.core.targets import BodyTargets, build_body_targets
from v2d_retarget.core.contacts import ContactTargets, TIP_NAMES
from v2d_retarget.core.constraints import HullAvoidanceTask, OutputVelocityLimit
from v2d_retarget.core.solver import SolveResult
from v2d_retarget.core.diagnostics import evaluate


@dataclass
class RobotCorrectionResult:
    """A refined reference and diagnostics; serialization is a separate stage."""

    motion: HumanMotion
    robot: G1Kinematics
    solution: SolveResult
    body_targets: BodyTargets
    contacts: ContactTargets
    config: dict
    metrics: dict
    diagnostics: dict


class ReferenceNeighborhood(Limit):
    def __init__(self, kin, lower, upper):
        self.lower=lower
        self.upper=upper
        self.projection=np.eye(kin.robot.model.nv)

    def compute_qp_inequalities(self, configuration, dt):
        upper=self.upper-configuration.q[7:]
        lower=self.lower-configuration.q[7:]
        # The base starts at its reference each frame and never moves in IK.
        return np.vstack((self.projection,-self.projection)),np.r_[np.zeros(6),upper,np.zeros(6),-lower]


class PatchTrackingTask(Task):
    """Prioritize closing a loaded surface gap over exact tangential placement."""
    def __init__(self,kin,name):
        super().__init__(cost=1.,gain=1.,lm_damping=.01)
        self.frame=kin.robot.model.getFrameId(name)

    def set_target(self,point,normal,weight):
        self.target=point
        # World-coordinate SPD weighting; a tip can move tangentially to fit
        # the robot hand while its surface-normal contact gap is closed.
        self.weight=weight*(12*np.eye(3)+(180-12)*np.outer(normal,normal))

    def compute_error(self,configuration):
        return self.weight@(configuration.data.oMf[self.frame].translation-self.target)

    def compute_jacobian(self,configuration):
        jac=pin.getFrameJacobian(configuration.model,configuration.data,self.frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return self.weight@jac[:3]

    def __repr__(self):
        return f'PatchTrackingTask(frame={self.frame})'


def repair(directory):
    started=perf_counter()
    motion, before, _, cfg, surface, contacts = load_book_reference(
        directory, preserve_saved_object_pose=True)
    with np.load(directory/'wrench_plan.npz', allow_pickle=False) as saved_plan:
        plan = {key: saved_plan[key].copy() for key in saved_plan.files}
    ids=plan['ids']
    kin=G1Kinematics()
    body=build_body_targets(motion,kin.config)
    corrected=deepcopy(contacts)
    rotations=Rotation.from_quat(motion.object_wxyz,scalar_first=True).as_matrix()
    # Keep the demonstrated grasp neighborhood. Proxy contacts are projected
    # onto the actual mesh before asking the robot to attain them.
    local=plan['mesh_points'] + contacts.clearance_m*plan['mesh_normals']
    corrected.target_tips[ids]=np.einsum('tij,kj->tki',rotations[ids],local)+motion.object_positions[ids,None]
    corrected.anchors_local[ids]=plan['mesh_points']
    corrected.normals_local[ids]=plan['mesh_normals']
    force_norm=np.linalg.norm(plan['after_forces'],axis=-1)
    load_weight=np.sqrt(np.maximum(force_norm.mean(axis=0),.25))
    load_weight/=load_weight.mean()
    tip_tasks=[PatchTrackingTask(kin,n) for n in TIP_NAMES]
    posture=PostureTask(cost=.025,lm_damping=.001)
    continuity=PostureTask(cost=1.0,lm_damping=.001)
    avoidance=HullAvoidanceTask(kin,surface,400.)
    q=before['q'].copy()
    iterations=np.zeros(len(q),int)
    stops=['unchanged reference']*len(q)
    arm_names=('shoulder','elbow','wrist','hand')
    movable=np.array([any(s in name for s in arm_names) for name in kin.joint_names])
    radius=np.where(movable,np.array([1.5 if 'hand' in n else .9 for n in kin.joint_names]),0.)
    pad=15
    start=max(1,int(ids[0])-pad)
    end=min(len(q)-2,int(ids[-1])+pad)
    blends=np.zeros(len(q))
    for t in range(start,end+1):
        if t<ids[0]:
            blend=np.sin(.5*np.pi*(t-start)/(ids[0]-start))**2
        elif t>ids[-1]:
            blend=np.sin(.5*np.pi*(end-t)/(end-ids[-1]))**2
        else:
            blend=1.
        blends[t]=blend
    lower=before['q'][:,7:]-blends[:,None]*radius
    upper=before['q'][:,7:]+blends[:,None]*radius
    # Backward reachable intervals ensure the local solve can return through
    # every shrinking neighborhood without violating saved-frame velocity.
    for t in range(end,start-1,-1):
        budget=.9*kin.robot.model.velocityLimit[6:]*(motion.timestamps[t+1]-motion.timestamps[t])
        lower[t]=np.maximum(lower[t],lower[t+1]-budget)
        upper[t]=np.minimum(upper[t],upper[t+1]+budget)
    if np.any(lower[start:end+1]>upper[start:end+1]+1e-8):
        raise ValueError('No velocity-feasible return to the input reference')
    for t in range(start,end+1):
        blend=blends[t]
        initial=before['q'][t].copy()
        dt=float(motion.timestamps[t]-motion.timestamps[t-1])
        frame_budget=.9*kin.robot.model.velocityLimit[6:]*dt
        feasible_lower=np.maximum(lower[t],q[t-1,7:]-frame_budget)
        feasible_upper=np.minimum(upper[t],q[t-1,7:]+frame_budget)
        initial[7:]=np.clip(q[t-1,7:],feasible_lower,feasible_upper)
        kin.configuration.update(initial)
        posture.set_target(before['q'][t])
        continuous_target=before['q'][t].copy()
        continuous_target[7:]=q[t-1,7:]+before['q'][t,7:]-before['q'][t-1,7:]
        continuity.set_target(continuous_target)
        # Fade into and out of the saved FK; no abrupt start/end target jumps.
        desired=(1-blend)*before['fk_tip_positions'][t]+blend*corrected.target_tips[t]
        for j,task in enumerate(tip_tasks):
            normal=rotations[t]@corrected.normals_local[t,j]
            task.set_target(desired[j],normal,float(load_weight[j]))
        avoidance.set_object(rotations[t],motion.object_positions[t])
        dt=float(motion.timestamps[t]-motion.timestamps[t-1])
        # Optimization increments are not physical commands. Velocity limits
        # apply between saved frames; the local tube also guarantees a return.
        limits=[ConfigurationLimit(kin.robot.model,config_limit_gain=1.0),
            OutputVelocityLimit(kin.robot.model,q[t-1],dt,.9),
            ReferenceNeighborhood(kin,lower[t],upper[t])]
        tasks=tip_tasks+[posture,continuity,avoidance]
        def objective():
            return sum(float(np.sum((task.cost*np.asarray(task.compute_error(kin.configuration)))**2)) for task in tasks)
        for i in range(100):
            try:
                velocity=pink.solve_ik(kin.configuration,tasks,kin.dt,
                    solver=kin.solver,limits=limits,safety_break=True)
            except Exception as exc:
                raise RuntimeError(f'Local IK failed at frame {t}, iteration {i}, blend {blend}') from exc
            # Reject nonlinear steps that worsen the true geometry objective.
            # All interpolated candidates remain inside the convex joint boxes.
            old_q=kin.configuration.q.copy()
            old_cost=objective()
            accepted=False
            for backtrack in range(9):
                step=velocity*kin.dt*(.5**backtrack)
                kin.configuration.update(pin.integrate(kin.robot.model,old_q,step))
                if objective()<=old_cost+1e-10:
                    accepted=True
                    break
            if not accepted:
                kin.configuration.update(old_q)
                break
            if i>=4 and np.linalg.norm(velocity*kin.dt)<1e-5:
                break
        q[t]=kin.configuration.q.copy()
        # Exact assignment removes sub-ulp integration drift on frozen coordinates.
        q[t,:7]=before['q'][t,:7]
        q[t,7:][~movable]=before['q'][t,7:][~movable]
        iterations[t]=i+1
        stops[t]='local contact projection'
    solved=SolveResult(q,iterations,stops,perf_counter()-started)
    metrics,arrays=evaluate(kin,motion,q,body,corrected,surface)
    if metrics['speed_violating_transitions'] or metrics['joint_limit_violation_rad']>1e-6:
        raise ValueError('Repaired reference fails joint limits or output velocity limits')
    metrics['phase_before_anchor_error_mm']=float(before['contact_error_m'][ids].mean()*1000)
    metrics['phase_after_anchor_error_mm']=float(arrays['contact_error_m'][ids].mean()*1000)
    metrics['phase_before_max_probe_penetration_mm']=float(before['probe_penetration_m'][ids].max()*1000)
    metrics['phase_after_max_probe_penetration_mm']=float(arrays['probe_penetration_m'][ids].max()*1000)
    metrics['max_joint_change_rad']=float(np.max(np.abs(q-before['q'])))
    metrics['elapsed_seconds']=solved.elapsed_seconds
    metrics['load_weights']=load_weight.tolist()
    metrics['correction_settings']={'tip_normal_cost':180.,'tip_tangent_cost':12.,'avoidance_cost':400.,'posture_cost':.025,'correction_continuity_cost':1.,'nonlinear_backtracking':True,
        'arm_trust_radius_rad':.9,'finger_trust_radius_rad':1.5,'transition_padding_frames':pad}
    metrics['scope']='Local arm/hand kinematic projection using object-wrench load weights; no robot dynamics or physical rollout'
    metrics['modified_frame_range']=[start,end]
    return RobotCorrectionResult(motion, kin, solved, body, corrected, cfg, metrics, arrays)
