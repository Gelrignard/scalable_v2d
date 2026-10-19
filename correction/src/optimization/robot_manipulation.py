"""Supported G1 contact realization and a bounded, rollout-based command repair.

This is a local book-carry experiment. A solved force QP or IK problem is not
an execution certificate; the returned report keeps the tracking gate explicit.
"""
import json
from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from correction.src.dynamics.supported_robot import SupportedRobotScene
from correction.src.dynamics.sparse_forces import allocate_forces
from correction.src.dynamics.models import RigidBody
from correction.src.dynamics.object_wrench import object_wrench
from .supported_robot_ik import solve_frame


def metrics(result):
    return dict(max_position_error_mm=float(result['max_position_error_m'])*1000,
                max_rotation_error_deg=float(np.rad2deg(result['max_rotation_error_rad'])),
                rms_position_error_mm=float(np.sqrt(np.mean(result['position_error']**2)))*1000,
                rms_rotation_error_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error']**2)))),
                max_penetration_mm=float(result['max_penetration_m'])*1000,
                initial_penetration_mm=float(result['initial_max_penetration_m'])*1000,
                max_applied_torque_fraction=float(np.max(np.abs(result['torques'])/result['torque_caps'])),
                max_requested_torque_fraction=float(result['max_torque_fraction_requested']),
                saturation_steps=int(result['saturation_steps']),
                max_joint_limit_violation_rad=float(result['max_joint_limit_violation_rad']),
                max_object_applied_force=float(result['max_object_applied_force']),
                frames_with_both_hands_in_contact=int(np.sum(np.all(result['contacts_per_hand']>0,axis=1))))


def tracking_accepted(result):
    return bool(result['max_position_error_m'] < .01 and
                result['max_rotation_error_rad'] < np.deg2rad(5) and
                result['max_penetration_m'] < .001 and
                result['initial_max_penetration_m'] < .00005 and
                result['max_object_applied_force'] == 0. and
                result['max_joint_limit_violation_rad'] < .01)


def score(result):
    return float(np.mean((result['position_error']/.01)**2+(result['rotation_error']/.1)**2))


def realize_tip_update(scene, seed, targets, positions, rotations):
    """Small, bounded Jacobian updates; physical replay rejects bad contacts.

    This inner solve is not collision-certified. Clipping keeps commanded joints
    inside their limits; the actual free-dynamics replay remains the outer test.
    """
    q = seed.copy()
    indices = 7+scene.movable
    low = scene.kin.robot.model.lowerPositionLimit[indices]
    high = scene.kin.robot.model.upperPositionLimit[indices]
    for i in range(len(q)):
        for _ in range(12):
            scene.set_state(q[i],positions[i],rotations[i])
            tips,jac = scene.tips_and_jacobian()
            j = jac.reshape(18,-1)
            error = (targets[i]-tips).ravel()
            delta = j.T@np.linalg.solve(j@j.T+1e-5*np.eye(18),error)
            q[i,indices] = np.clip(q[i,indices]+np.clip(delta,-.1,.1),low,high)
            if np.max(np.linalg.norm(tips-targets[i],axis=1)) < .0003:
                break
    return q


def correct_commands(scene, q, p, r, times, force, initial, *, iterations=8):
    initial_velocity = CubicSpline(times-times[0],q[:,7+scene.movable])(0,1)
    def evaluate(commands):
        return scene.rollout(commands,p,r,times,force_ff=force,kp_arm=300.,kp_hand=20.,
                             inertia_ff=True,initial_q=initial,initial_velocity=initial_velocity)
    best = evaluate(q)
    seed_result = best
    best_score = score(best)
    shift,bias = np.zeros_like(p),np.zeros_like(p)
    rotations = Rotation.from_quat(r,scalar_first=True)
    nominal = []
    for state,position,rotation in zip(q,p,r):
        scene.set_state(state,position,rotation)
        nominal.append(scene.tips_and_jacobian()[0])
    nominal = np.asarray(nominal)
    history = []
    for iteration in range(iterations):
        dp = gaussian_filter1d(p-best['object_positions'],2,axis=0)
        dr = gaussian_filter1d((rotations*Rotation.from_quat(best['object_wxyz'],scalar_first=True).inv()).as_rotvec(),2,axis=0)
        improved = False
        for gain in (.6,.25):
            ps,rb = shift+gain*dp,bias+gain*dr
            ps *= np.minimum(1.,.03/np.maximum(np.linalg.norm(ps,axis=1,keepdims=True),1e-10))
            rb *= np.minimum(1.,.4/np.maximum(np.linalg.norm(rb,axis=1,keepdims=True),1e-10))
            targets = p[:,None]+ps[:,None]+np.einsum('tij,tkj->tki',Rotation.from_rotvec(rb).as_matrix(),nominal-p[:,None])
            trial = realize_tip_update(scene,q,targets,p,r)
            result = evaluate(trial)
            value = score(result)
            keep = value < best_score and result['max_penetration_m'] < .001
            history.append(dict(iteration=iteration,gain=gain,objective=value,kept=bool(keep)))
            print(f'Correction {iteration}, gain {gain}: {1000*result["max_position_error_m"]:.2f} mm, '
                  f'{np.rad2deg(result["max_rotation_error_rad"]):.2f} deg; kept={keep}',flush=True)
            if keep:
                best,best_score,q,shift,bias = result,value,trial,ps,rb
                improved = True
                break
        if not improved:
            break
    return best,seed_result,q,initial_velocity,history


def run(reference, *, time_scale=2., iterations=8, prepared=None):
    started = perf_counter()
    ref = reference
    with np.load(ref.input_path,allow_pickle=False) as data:
        original = data['q'][ref.ids].copy()
        metadata = json.loads(str(data['metadata_json']))
    kin = G1Kinematics()
    support = metadata['source']['support_surface']
    fixed = original[0].copy()
    rotations = Rotation.from_quat(ref.quaternions,scalar_first=True)
    p,r = ref.positions,ref.quaternions
    initial_scene = SupportedRobotScene(kin,ref.bounds,support,fixed)
    if prepared is None:
        # Two hands oppose each other across the thin box faces. A fingertip
        # frame is inside the real distal geometry, hence a 14 mm site offset.
        if int(np.argmin(np.ptp(ref.bounds,axis=0))) != 1:
            raise ValueError('This prototype currently requires a box thin along local Y')
        local = rotations[0].inv().apply(ref.robot_tips[0]-p[0])
        local[:3,1],local[3:,1] = ref.bounds[0,1]-.01453,ref.bounds[1,1]+.01463
        initial,_ = solve_frame(initial_scene,fixed,p[0]+rotations[0].apply(local),p[0],r[0],
                                evaluations=250,collision_weight=1000.,tangent_weight=60.)
        # Bounded tangential regrasp: bring the opposing contact patches into
        # alignment. This is a book-specific candidate, not a general planner.
        local[:3,[0,2]] -= .04
        local[3:,[0,2]] += .04
        initial,grasp_info = solve_frame(initial_scene,initial,p[0]+rotations[0].apply(local),p[0],r[0],
                                         evaluations=250,collision_weight=1000.,tangent_weight=60.)
        initial_scene.set_state(initial,p[0],r[0])
        local = rotations[0].inv().apply(initial_scene.tips_and_jacobian()[0]-p[0])
        scene = SupportedRobotScene(kin,ref.bounds,support,fixed,move_waist=True)
        targets = p[:,None]+np.einsum('tij,kj->tki',rotations.as_matrix(),local)
        q = [initial]
        for i in range(1,len(p)):
            state,info = solve_frame(scene,q[-1],targets[i],p[i],r[i],reference=initial,
                                     evaluations=80,collision_weight=1000.,posture_weight=.1)
            q.append(state)
            if i%10 == 0:
                print(f'G1 contact IK {i}/{len(p)-1}: max site error {info["max_tip_error_mm"]:.2f} mm',flush=True)
        q = np.asarray(q)
    else:
        q,local = prepared['q'].copy(),prepared['local'].copy()
        if q.shape != original.shape or local.shape != (6,3) or not np.isfinite(q).all() or not np.isfinite(local).all():
            raise ValueError('Prepared reference must contain finite q [selected frames, robot nq] and local [6,3]')
        if not np.allclose(q[0,:19],fixed[:19],atol=1e-8):
            raise ValueError('Prepared reference has a different fixed pelvis/leg pose')
        initial = q[0].copy()
        grasp_info = dict(source='Provided initial robot reference; only correction time measured')
        scene = SupportedRobotScene(kin,ref.bounds,support,fixed,move_waist=True)
    prepared = dict(q=q.copy(),local=local.copy())
    times = (ref.times-ref.times[0])*time_scale
    size = np.ptp(ref.bounds,axis=0)
    body = RigidBody(1.,ref.bounds.mean(0),np.diag((sum(size**2)-size**2)/12),np.array([0.,0.,-9.81]),'Assumed uniform 1 kg book')
    visible = np.all(np.isfinite(ref.all_poses),axis=1)
    dynamics = object_wrench(ref.all_times*time_scale,ref.all_poses,visible,body,window=11)
    required = dynamics['required_wrench_object'][ref.ids]
    points = local.copy()
    points[:3,1],points[3:,1] = ref.bounds[0,1],ref.bounds[1,1]
    inward = np.zeros((6,3));inward[:3,1]=1;inward[3:,1]=-1
    plan = allocate_forces(np.tile(points,(len(times),1,1)),np.tile(inward,(len(times),1,1)),
                           np.ones((len(times),6),bool),required,times,body.com_object_m,
                           cap=20.,mu=.6,time_limit=3.,effort=1e-7)
    if plan.status not in ('solved','solved inaccurate'):
        raise RuntimeError(f'Force planning failed: {plan.status}')
    force = np.einsum('tij,tkj->tki',rotations.as_matrix(),plan.forces)
    corrected,seed,q,velocity,history = correct_commands(scene,q,p,r,times,force,initial,iterations=iterations)
    # Common initial grasp, time scale, motor limits and initial velocity. Only
    # reference commands/force guidance differ; initial acquisition is excluded.
    blend = np.clip((times-.1)/.4,0,1);blend = (blend**2*(3-2*blend))[:,None]
    baseline_q = prepared['q']+blend*(original-prepared['q'])
    baseline = scene.rollout(baseline_q,p,r,times,kp_arm=300.,kp_hand=20.,inertia_ff=True,
                             initial_q=initial,initial_velocity=velocity)
    geometry = scene.rollout(prepared['q'],p,r,times,kp_arm=300.,kp_hand=20.,inertia_ff=True,
                             initial_q=initial,initial_velocity=velocity)
    records = dict(baseline=baseline,geometry_only=geometry,wrench_seed=seed,corrected=corrected)
    report = dict(input_reference=str(ref.input_path),frame_range=[int(ref.ids[0]),int(ref.ids[-1])],
                  source_duration_s=float(ref.times[-1]-ref.times[0]),rollout_duration_s=float(times[-1]),
                  time_scale=time_scale,method_seconds=perf_counter()-started,prepared_reference_used=grasp_info.get('source'),
                  fixed_root_and_legs=True,dynamic_waist=True,active_joints=31,
                  parameters=dict(book_mass_kg=1.,friction=.6,timestep_s=.001,kp_arm_and_waist=300.,kp_hand=20.,
                                  kd_arm_and_waist=8.,kd_hand=.4,finger_torque_limit_nm=.76),
                  accepted_tracking=tracking_accepted(corrected),ready_for_training=False,
                  force_qp_status=plan.status,force_feasible_frames=int(np.sum(plan.feasible)),
                  max_force_plan_residual_n=float(np.max(np.linalg.norm(plan.residual[:,:3],axis=1))),
                  max_torque_plan_residual_nm=float(np.max(np.linalg.norm(plan.residual[:,3:],axis=1))),
                  max_hand_site_change_from_human_mm=float(np.max(np.linalg.norm(corrected['fk_tip_positions']-ref.human_tips,axis=-1)))*1000,
                  accepted_demonstration_locality=bool(np.max(np.linalg.norm(corrected['fk_tip_positions']-ref.human_tips,axis=-1)) < .08),
                  history=history,**{name:metrics(record) for name,record in records.items()})
    interface = dict(frame_ids=ref.ids,timestamps=times,source_timestamps=ref.times,q=q,initial_q=initial,
                     initial_velocity=velocity,fixed_q=fixed,object_positions=p,object_wxyz=r,force_ff=force,
                     bounds=ref.bounds,table_center=ref.table[0],table_radius=ref.table[1],table_height=ref.table[2],
                     object_mesh=str(ref.mesh_path),support_path=support,prepared_q=prepared['q'],local=local,
                     force_plan_residual=plan.residual)
    return scene,report,records,interface
