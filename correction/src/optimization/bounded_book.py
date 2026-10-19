"""Run bounded CPU contact correction and physical diagnostics on the Tier 1 book."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from correction.src.optimization.budget import Budget, BudgetExceeded
from correction.src.optimization.contact_hypotheses import hypotheses
from correction.src.optimization.window_robot import refine_window, project_temporal
from correction.src.dynamics.models import RigidBody
from correction.src.dynamics.object_wrench import object_wrench, world_points_to_object
from correction.src.dynamics.window_audit import attained, forces_with_support


def run(args):
    from correction.src.preprocessing.book_reference import load_book_reference
    from correction.src.dynamics.robot_scene import RobotScene
    from correction.src.pass_traj.bounded_reference import save_bounded
    from v2d_retarget.core.robot import G1Kinematics
    from v2d_retarget.core.contacts import TIP_NAMES
    started = perf_counter()
    if args.budget <= 5 or args.candidates < 1 or not (1 <= args.robot_candidates <= args.candidates):
        raise ValueError('budget must exceed 5 seconds and candidate counts must be positive and consistent')
    budget = Budget(args.budget-5)  # reserve time for a final temporal seam solve
    motion, before, metadata, config, surface, contacts = load_book_reference(args.input, True)
    kin = G1Kinematics()
    if not (4 <= args.start < args.stop <= len(motion.timestamps)-4) or args.padding < 0:
        raise ValueError('need nonnegative padding and four reference frames on either side')
    if args.window < 7 or args.overlap < 3 or args.overlap >= args.window-2:
        raise ValueError('need window >=7 and 3 <= overlap < window-2')
    lo, hi = max(4, args.start-args.padding), min(len(motion.timestamps)-4, args.stop+args.padding)
    scene = RobotScene(kin, surface.mesh.bounds, metadata['source']['support_surface'],
                       mass=args.mass, friction=args.friction)
    size, com = np.ptp(surface.mesh.bounds, axis=0), surface.mesh.bounds.mean(axis=0)
    body = RigidBody(args.mass, com, np.diag(args.mass/12*(np.sum(size**2)-size**2)),
                     np.array([0., 0., -9.81]), 'User-configurable rigid box assumptions, not verified challenge parameters')
    pose = np.column_stack((motion.object_positions, motion.object_wxyz))
    # Use the exact exported poses, not one trajectory for forces and another
    # for replay. No smoothing here: sensitivity to noisy derivatives is logged.
    dynamics = object_wrench(motion.timestamps, pose, np.ones(len(pose), bool), body, window=1)
    required = dynamics['required_wrench_object']
    if not np.isfinite(required[lo-2:hi+2]).all():
        raise ValueError('requested interval contains invalid object derivatives')
    source_local = world_points_to_object(pose, contacts.source_tips)
    q = before['q'].copy()
    movable = np.array([i for i, n in enumerate(kin.joint_names)
                        if any(word in n for word in ('shoulder', 'elbow', 'wrist', 'hand'))])
    planned_points, planned_normals = contacts.anchors_local.copy(), -contacts.normals_local.copy()
    planned_active, targets = contacts.active.copy(), contacts.target_tips.copy()
    report = dict(input_reference=str((args.input/'retargeted/contact_aware.npz').resolve()),
                  requested_frames_inclusive=[args.start, args.stop-1],
                  correction_frames_inclusive=[lo, hi-1], settings=vars(args).copy(),
                  physical_parameters=dict(mass_kg=args.mass, friction=args.friction,
                                           provenance=body.provenance, com_m=com.tolist(), inertia_kg_m2=body.inertia_com_object_kg_m2.tolist()),
                  object_reference='raw exported SE3 poses; spline derivatives without smoothing; object trajectory unchanged',
                  windows=[], budget_exhausted=False, accepted_as_feasible_manipulation=False,
                  ready_for_training=False, full_body_inverse_dynamics_enforced=False)
    report['numerical_limits'] = dict(arm_acceleration_rad_s2=80., finger_acceleration_rad_s2=200.,
        velocity_fraction_of_urdf=.9, provenance='acceleration bounds are engineering settings, not measured actuator limits')
    report['settings'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['settings'].items()}
    previous = None
    cursor = lo
    try:
        report['boundary_attempts'] = []
        for attempt in range(4):
            budget.check()
            projected, status = project_temporal(kin, q[lo-2:hi+2], before['q'][lo-2:hi+2],
                motion.timestamps[lo-2:hi+2], movable, trust=args.trust, time_limit=min(4., budget.remaining))
            report['boundary_attempts'].append(dict(start=lo, stop=hi, status=status))
            if projected is not None:
                break
            if attempt < 3:
                lo, hi = max(4, lo-20), min(len(q)-4, hi+20)
        cursor = lo
        report['correction_frames_inclusive'] = [lo, hi-1]
        report['initial_temporal_projection'] = status
        if projected is None:
            raise RuntimeError(f'Cannot establish a continuous initial reference: {status}')
        q[lo-2:hi+2] = projected
        while cursor < hi:
            budget.check()
            start, end = cursor-2, min(cursor+args.window-2, hi+2)
            ids = np.arange(start, end)
            if len(ids) < 5:
                break
            times = motion.timestamps[ids]
            positions, quats = motion.object_positions[ids], motion.object_wxyz[ids]
            tip_positions = np.array([[p.translation for p in kin.poses(qt, TIP_NAMES)] for qt in q[ids]])
            local = world_points_to_object(pose[ids], tip_positions)
            candidates = hypotheses(source_local[ids], local, surface.mesh.bounds, contacts.active[ids],
                                    previous, max_candidates=args.candidates)
            log = dict(frames_inclusive=[int(ids[0]), int(ids[-1])], candidates=[])
            ranked = []
            for candidate in candidates:
                budget.check()
                sample = np.unique(np.r_[np.arange(0, len(ids), 3), len(ids)-1])
                count = len(sample)
                force = forces_with_support(scene, np.repeat(candidate.points[None], count, axis=0),
                    np.repeat(candidate.inward[None], count, axis=0), candidate.active[sample],
                    required[ids[sample]], positions[sample], quats[sample], times[sample],
                    time_limit=min(1., max(.001, budget.remaining)))
                score = float(np.mean(np.linalg.norm(force.residual[:, :3], axis=1))
                              +10*np.mean(np.linalg.norm(force.residual[:, 3:], axis=1))+.05*candidate.evidence_cost)
                loads = np.sqrt(np.maximum(np.linalg.norm(force.forces[:, :6], axis=-1).mean(axis=0), .25))
                candidate.load_weights = loads/loads.mean()
                log['candidates'].append(dict(name=candidate.name, evidence_cost=candidate.evidence_cost,
                    screening_score=score, wrench_feasible_knots=int(force.feasible.sum()), knots=count,
                    force_solver_status=force.status, matrix_nnz=force.matrix_nnz))
                ranked.append((score, candidate))
            ranked.sort(key=lambda x: x[0])
            # The untouched window is an explicit candidate. Do not replace it
            # merely because an optimization returned finite numbers.
            base_report, _ = attained(kin, scene, q[ids], positions, quats, times, required[ids])
            def quality(metrics):
                return (10000*metrics['manipulation_penetration_squared_sum_m2']
                        + metrics['max_manipulation_penetration_mm']
                        + metrics['contact_gap_mean_mm']
                        + 2*(metrics['frames']-metrics['wrench_feasible_frames']))
            best_score = quality(base_report)
            best_q, chosen = q[ids].copy(), None
            log['before'] = base_report
            for _, candidate in ranked[:args.robot_candidates]:
                budget.check()
                candidate_q = q[ids].copy()
                solve_logs = []
                # Close the loop once: project, audit attained contacts, then
                # retry locally. The second round can improve the same mode;
                # alternative modes are still compared at this window level.
                for _ in range(2):
                    candidate_q, solve_log = refine_window(kin, scene, before['q'][ids], candidate_q,
                        positions, quats, times, candidate, movable, budget,
                        iterations=args.iterations, prefix=2, suffix=2 if end == hi+2 else 0,
                        trust=args.trust)
                    solve_logs.append(solve_log)
                    budget.check()
                    audit, attained_arrays = attained(kin, scene, candidate_q, positions, quats, times, required[ids])
                    if quality(audit) < best_score and audit['max_joint_speed_ratio'] <= 1.0001:
                        best_score, best_q, chosen = quality(audit), candidate_q.copy(), candidate
                    # Mechanical infeasibility after projection is explicit.
                    # Do not invent forces at the planned but unattained points.
                    if audit['wrench_feasible_frames'] == len(ids) and audit['collision_samples_over_2mm'] == 0:
                        break
                    loads = np.sqrt(np.maximum(np.linalg.norm(attained_arrays['forces'], axis=-1).mean(axis=0), .25))
                    candidate.load_weights = loads/loads.mean()
                log.setdefault('robot_trials', []).append(dict(name=candidate.name, audit=audit, solves=solve_logs))
            commit_end = end if end == hi+2 else end-args.overlap
            # Keep the lookahead as a continuous starting point for the next
            # window. Only the committed prefix is frozen on the next solve.
            take = np.arange(2, end-start)
            q[ids[take]] = best_q[take]
            if chosen is not None:
                planned_points[ids[take]] = chosen.points
                planned_normals[ids[take]] = chosen.inward
                planned_active[ids[take]] = chosen.active[take]
                rotations = Rotation.from_quat(quats[take], scalar_first=True).as_matrix()
                targets[ids[take]] = np.einsum('tij,kj->tki', rotations, chosen.points-.002*chosen.inward)+positions[take, None]
                previous = chosen
            log['selected'] = chosen.name if chosen else 'unchanged_input'
            log['committed_through_frame'] = int(commit_end-1)
            report['windows'].append(log)
            cursor = commit_end
            print(json.dumps(dict(committed=cursor-1, selected=log['selected'], elapsed=round(budget.elapsed, 2))), flush=True)
    except BudgetExceeded:
        report['budget_exhausted'] = True
    # Joining an optimized prefix to untouched input needs its own temporal
    # solve, especially on budget exhaustion. Its cost is reported separately.
    temporal_started = perf_counter()
    projected, status = project_temporal(kin, q[lo-2:hi+2], before['q'][lo-2:hi+2],
        motion.timestamps[lo-2:hi+2], movable, trust=args.trust, time_limit=4.)
    report['final_temporal_projection'] = status
    report['final_temporal_projection_seconds'] = perf_counter()-temporal_started
    if projected is not None:
        q[lo-2:hi+2] = projected
    report['correction_seconds'] = budget.elapsed
    report['processed_through_frame'] = min(cursor-1, hi-1)
    report['all_requested_frames_processed'] = cursor >= args.stop
    ids = np.arange(lo-2, hi+2)
    report['before'], _ = attained(kin, scene, before['q'][ids], motion.object_positions[ids], motion.object_wxyz[ids],
                                   motion.timestamps[ids], required[ids], midpoints=True)
    report['after'], arrays = attained(kin, scene, q[ids], motion.object_positions[ids], motion.object_wxyz[ids],
                                       motion.timestamps[ids], required[ids], midpoints=True)
    # Record changed-frame coverage; never label the whole episode corrected.
    changed = np.flatnonzero(np.max(np.abs(q-before['q']), axis=1) > 1e-10)
    report['changed_frames'] = changed.tolist()
    report['rollouts'] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rollout:
        from correction.src.optimization.rollout_refinement import compare_rollouts
        rollout_ids = np.arange(args.start, args.stop)
        records, report['rollouts'], report['control_refinement'] = compare_rollouts(scene, kin,
            before['q'][rollout_ids], q[rollout_ids], motion.object_positions[rollout_ids],
            motion.object_wxyz[rollout_ids], motion.timestamps[rollout_ids],
            planned_normals[rollout_ids], planned_active[rollout_ids])
        for label, rollout in records.items():
            np.savez_compressed(args.output/f'rollout_{label}.npz', **rollout)
    report['total_seconds'] = perf_counter()-started
    path = save_bounded(args.output, before, metadata, q, ids, targets[ids], planned_points[ids],
                        planned_normals[ids], planned_active[ids], arrays, report)
    print(json.dumps(dict(output=str(path), before=report['before'], after=report['after'],
                          correction_seconds=report['correction_seconds'], budget_exhausted=report['budget_exhausted'],
                          rollouts=report['rollouts']), indent=2), flush=True)
    return report
