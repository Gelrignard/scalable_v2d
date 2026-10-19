"""Bounded full-episode diagnostic: contact repair, wrench guidance, physical replay.

Unlike the book experiment, this starts at the actual first frame and does not
prepare a grasp. Contact hypotheses may fail. All failed frames stay in output.
"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from correction.src.preprocessing.convex_object import decompose
from correction.src.dynamics.supported_robot import SupportedRobotScene
from correction.src.dynamics.models import RigidBody
from correction.src.dynamics.object_wrench import object_wrench
from correction.src.dynamics.sparse_forces import allocate_forces, ForceResult
from .robot_manipulation import metrics, tracking_accepted, score


def repair(scene, original, targets, positions, rotations, times, *, iterations=12):
    """Damped task-space/collision least squares, bounded by joint speed and pose.

    This cheap inner iteration is a candidate generator, not a feasibility proof.
    It explicitly preserves base commands and the common initial configuration.
    """
    q = original.copy()
    idx = 7+scene.movable
    low = scene.kin.robot.model.lowerPositionLimit[idx]
    high = scene.kin.robot.model.upperPositionLimit[idx]
    speed = scene.kin.robot.model.velocityLimit[6+scene.movable]*.9
    for i in range(1,len(q)):
        lo = np.maximum(low, q[i-1,idx]-speed*(times[i]-times[i-1]))
        hi = np.minimum(high, q[i-1,idx]+speed*(times[i]-times[i-1]))
        q[i,idx] = np.clip(q[i,idx],lo,hi)
        for _ in range(iterations):
            scene.set_state(q[i],positions[i],rotations[i])
            tips,jac = scene.tips_and_jacobian()
            j = jac.reshape(18,-1)
            error = (targets[i]-tips).ravel()
            a = [j, .025*np.eye(len(idx))]
            b = [error, .025*(original[i,idx]-q[i,idx])]
            for row in scene.contacts(scene.movable):
                if row['distance'] < 0 and np.linalg.norm(row['jacobian']) > 1e-8:
                    a.append(4.*row['jacobian'][None,:])
                    b.append(np.array([-4.*row['distance']]))
            a,b = np.vstack(a),np.concatenate(b)
            delta = np.linalg.solve(a.T@a+1e-4*np.eye(len(idx)),a.T@b)
            q[i,idx] = np.clip(q[i,idx]+np.clip(delta,-.08,.08),lo,hi)
            if np.max(np.abs(delta)) < 1e-4:
                break
        if i % 200 == 0:
            print(f'Full-episode correction {i}/{len(q)}',flush=True)
    return q


def force_plan(scene, data, mesh, times, floor_z):
    p,r = data['object_positions'], data['object_wxyz']
    rotations = Rotation.from_quat(r,scalar_first=True)
    rm = rotations.as_matrix()
    m = scene.model; body = scene.book_body
    axes = Rotation.from_quat(m.body_iquat[body],scalar_first=True).as_matrix()
    inertia = axes@np.diag(m.body_inertia[body])@axes.T
    physical = RigidBody(float(m.body_mass[body]),m.body_ipos[body],inertia,np.array([0.,0.,-9.81]),
                         'Assumed mass and uniform density over supplied convex decomposition')
    dynamics = object_wrench(times,np.c_[p,r],np.ones(len(times),bool),physical)
    valid = dynamics['valid_wrench']
    required = dynamics['required_wrench_object'].copy()
    # Keep endpoint frames. A stated quasi-static fallback replaces unavailable
    # endpoint derivatives, without labelling those derivatives as measured.
    required[~valid,:3] = rotations[~valid].inv().apply(np.tile([0.,0.,physical.mass_kg*9.81],(sum(~valid),1)))
    required[~valid,3:] = 0.
    points = np.zeros((len(times),10,3)); normals = np.zeros_like(points)
    active = np.zeros((len(times),10),bool)
    points[:,:6] = data['contact_anchors_local']
    normals[:,:6] = -data['contact_normals_local']
    bad = np.linalg.norm(normals[:,:6],axis=2)<1e-8
    normals[:,:6][bad] = [0.,0.,1.]
    active[:,:6] = data['contact_active']
    # Four extreme vertices near the plane approximate the ground support patch.
    vertices = mesh.convex_hull.vertices
    lowest = np.zeros(len(times))
    for i in range(len(times)):
        world = vertices@rm[i].T+p[i]
        lowest[i] = world[:,2].min()
        candidates = np.flatnonzero(world[:,2] <= lowest[i]+.012)
        chosen = [candidates[np.argmin(world[candidates,0])],candidates[np.argmax(world[candidates,0])],
                  candidates[np.argmin(world[candidates,1])],candidates[np.argmax(world[candidates,1])]]
        points[i,6:] = vertices[chosen]
        normals[i,6:] = rm[i].T@np.array([0.,0.,1.])
        active[i,6:] = abs(lowest[i]-floor_z) < .025
    # Bound solver cost with short force-planning windows. The physical replay
    # below is still ONE uninterrupted episode, with no state resets at seams.
    plans = []
    for start in range(0,len(times),48):
        sl = slice(start,min(start+48,len(times)))
        plans.append(allocate_forces(points[sl],normals[sl],active[sl],required[sl],times[sl],physical.com_object_m,
            cap=np.r_[np.full(6,15.),np.full(4,physical.mass_kg*20.)],
            time_limit=1.,effort=1e-4,rate=1e-5,solver_tolerance=1e-5))
    plan = ForceResult(np.concatenate([a.forces for a in plans]),np.concatenate([a.residual for a in plans]),
        np.concatenate([a.feasible for a in plans]),'; '.join(sorted(set(a.status for a in plans))),
        sum(a.seconds for a in plans),sum(a.matrix_nnz for a in plans))
    force = np.einsum('tij,tkj->tki',rm,plan.forces[:,:6])
    force[~data['contact_active']] = 0.
    return force,plan,valid,lowest


def run(path, directory, *, mass):
    started = perf_counter()
    with np.load(path,allow_pickle=False) as archive:
        data = {k:archive[k].copy() for k in archive.files}
    original = data['q']; times = data['timestamps']-data['timestamps'][0]
    p,r = data['object_positions'],data['object_wxyz']
    metadata = json.loads(str(data['metadata_json']))
    mesh = trimesh.load(str(data['object_mesh']),force='mesh')
    parts = decompose(str(data['object_mesh']),Path(directory)/'collision')
    collision_vertices = np.concatenate([trimesh.load(part,force='mesh').vertices for part,_ in parts])
    initial_floor = float((Rotation.from_quat(r[0],scalar_first=True).apply(collision_vertices)+p[0])[:,2].min())
    # A shared scene-registration correction prevents a penetration impulse
    # from deciding the comparison before manipulation begins. Only allow a
    # small initial plane offset; airborne objects must not acquire a fake floor.
    floor_z = initial_floor if abs(initial_floor) <= .03 else 0.
    kin = G1Kinematics()
    scene = SupportedRobotScene(kin,mesh.bounds,metadata['source']['support_surface'],original[0],
        mass=mass,friction=.6,dt=.001,move_waist=True,moving_fixture=True,object_parts=parts,floor_z=floor_z)
    rotations = Rotation.from_quat(r,scalar_first=True).as_matrix()
    nominal = []
    for q,pos,rot in zip(original,p,r):
        scene.set_state(q,pos,rot)
        nominal.append(scene.tips_and_jacobian()[0])
    nominal = np.asarray(nominal)
    # The site is inside the manufacturer's distal link. A 10 mm offset is
    # a geometry heuristic; the actual collision mesh is checked in repair.
    local = data['contact_anchors_local']+.010*data['contact_normals_local']
    contact_targets = p[:,None]+np.einsum('tij,tkj->tki',rotations,local)
    delta = contact_targets-nominal
    delta *= np.minimum(1.,.08/np.maximum(np.linalg.norm(delta,axis=2,keepdims=True),1e-9))
    targets = nominal+data['contact_blend'][:,:,None]*delta
    targets[0] = nominal[0]
    geometry = repair(scene,original,targets,p,r,times)
    force,plan,valid,lowest = force_plan(scene,data,mesh,times,floor_z)
    velocity = CubicSpline(times,original[:,7+scene.movable])(0,1)
    def evaluate(q,ff=None):
        return scene.rollout(q,p,r,times,force_ff=ff,kp_arm=300.,kp_hand=20.,inertia_ff=True,
                             initial_q=original[0],initial_velocity=velocity)
    print('Replaying entire baseline episode',flush=True)
    baseline = evaluate(original)
    print('Replaying entire contact/wrench corrected episode',flush=True)
    corrected = evaluate(geometry,force)
    # One bounded feedback refinement, only while the object is still near its
    # reference. This cannot invent a recovery after the object is already lost.
    dp = gaussian_filter1d(p-corrected['object_positions'],2,axis=0)
    near = corrected['position_error'] < .05
    shift = .3*dp*np.minimum(1.,.02/np.maximum(np.linalg.norm(.3*dp,axis=1,keepdims=True),1e-9))
    trial_targets = targets+near[:,None,None]*data['contact_blend'][:,:,None]*shift[:,None]
    trial = repair(scene,geometry,trial_targets,p,r,times,iterations=6)
    print('Checking one bounded command refinement',flush=True)
    candidate = evaluate(trial,force)
    kept = bool(score(candidate)<score(corrected) and candidate['max_penetration_m'] <= max(.001,corrected['max_penetration_m']))
    seed = corrected
    if kept: corrected,geometry = candidate,trial
    report = dict(frames=len(times),source_duration_s=float(times[-1]),rollout_duration_s=float(times[-1]),
        input_reference=str(Path(path).resolve()),method_seconds=perf_counter()-started,
        mass_kg=mass,friction=.6,timestep_s=.001,convex_parts=len(parts),floor_z_m=floor_z,
        initial_collision_bottom_before_registration_m=initial_floor,
        force_window_frames=48,
        moving_external_pelvis_fixture=True,legs_frozen_with_contacts_disabled=True,
        object_is_single_rigid_body=True,stroller_wheels_articulated=False,
        force_qp_status=plan.status,force_feasible_frames=int(sum(plan.feasible)),
        force_plan_seconds=plan.seconds,endpoint_quasistatic_frames=int(sum(~valid)),
        ground_height_range_m=[float(lowest.min()),float(lowest.max())],
        refinement_kept=kept,accepted_tracking=tracking_accepted(corrected),ready_for_training=False,
        baseline=metrics(baseline),corrected=metrics(corrected),wrench_seed=metrics(seed),
        actual_max_root_position_error_m=float(np.max(np.linalg.norm(corrected['q'][:,:3]-original[:,:3],axis=1))),
        scope='Local assumed physics; full frames and original timing; no walking/balance, no object assistance or resets.')
    interface = dict(frame_ids=np.arange(len(times)),timestamps=times,source_timestamps=data['timestamps'],
        q=geometry,baseline_command_q=original,initial_q=original[0],initial_velocity=velocity,floor_z=floor_z,
        object_positions=p,object_wxyz=r,object_mesh=str(data['object_mesh']),bounds=mesh.bounds,
        force_ff=force,contact_active=data['contact_active'],force_plan_residual=plan.residual,
        valid_measured_wrench=valid,tip_targets=targets)
    return scene,report,dict(baseline=baseline,corrected=corrected,wrench_seed=seed,candidate=candidate),interface
