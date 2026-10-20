"""Update one second-approach prediction, freeze prefix, audit before export."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.strategy_bundle import read_npz, load_strategy_scene
from correction.src.optimization.phase_prediction import updated_planning_poses
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.surface_reference import smoothstep, PhaseTiming
from correction.src.optimization.rigid_grasp import RigidGrasp
from correction.src.optimization.actuator_force_profile import allocate_profile
from correction.src.optimization.command_smoothing import refine_commands
from correction.src.optimization.finger_transition import audit_path
from correction.src.optimization.servo_preload import apply_offset
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.rollout_manipulation_sequence import select_phase


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh experiment directory')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    previous = json.loads((args.prior_plan/'report.json').read_text(encoding='utf-8'))
    seed = read_npz(args.source/'reference.npz')
    prediction = read_npz(args.source/'mechanics.npz')
    if saved.get('object_assistance') is not False or saved.get('resets_during_episode') != 0:
        raise ValueError('Prediction must come from an uninterrupted unassisted episode')
    if prediction['max_object_applied_force'] != 0 or len(prediction['q']) != len(seed['q']):
        raise ValueError('Incomplete or assisted physical predictor')
    if not np.array_equal(seed['object_positions'], read_npz(args.prior_plan/'reference.npz')['object_positions']):
        raise ValueError('Prior planning path does not share the task')
    _, _, _, scene = load_strategy_scene(saved['input_directory'])
    parse = lambda p: SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
    points = [parse(p) for p in saved['contacts']]
    phase = saved['phase_contacts'][1]
    phase_points = [parse(p) for p in phase['contacts']]
    timing = phase['timing']; boundary = timing['approach_start']
    # This prototype retains the same second-grasp shape and contact schedule.
    old_p, plan_p, plan_r, update = updated_planning_poses(
        seed['object_positions'], seed['object_wxyz'],
        prediction['object_positions'][boundary], prediction['object_wxyz'][boundary],
        np.asarray(previous['command']['second_planning_translation_m']),
        boundary=boundary, acquired=timing['contact_start'],
        fade_begin=timing['force_full'], fade_end=timing['contact_end'], stationary_approach=True)
    # Reuse the screened grasp, not a potentially displaced previous servo target.
    candidate_directory = Path(previous['configuration']['second_candidates'])
    chosen, _ = select_phase(candidate_directory, phase['frame'],
        previous['configuration']['second_family'], PhaseTiming(**timing))
    args.output.mkdir(parents=True)
    report = {k: saved[k] for k in ['input_directory', 'controller', 'physics', 'contacts',
                                     'phase_contacts', 'kp_hand', 'inertia_ff_scale', 'initialization_protocol']}
    report.update(status='building', source=str(args.source.resolve()), cases={}, ready_for_training=False,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        prediction_update=update, frozen_prefix_frame=boundary,
        object_assistance=False, resets_during_episode=0,
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            args.source/'reference.npz', args.source/'mechanics.npz', args.source/'report.json',
            args.prior_plan/'report.json', candidate_directory/'geometry_candidates.npz',
            candidate_directory/'report.json']},
        provenance=snapshot_sources(args.output, [Path(__file__), *[
            Path('correction/src')/p for p in ['optimization/phase_prediction.py',
                'optimization/command_smoothing.py', 'optimization/rigid_grasp.py',
                'optimization/actuator_force_profile.py', 'optimization/finger_transition.py',
                'optimization/servo_preload.py', 'dynamics/supported_robot.py']]]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    write()
    try:
        command = {k: v.copy() for k, v in seed.items()}
        index = {p.body: i for i, p in enumerate(points)}
        grasp = RigidGrasp(scene, phase_points, chosen.grasp_q,
                           seed['object_positions'][phase['frame']], seed['object_wxyz'][phase['frame']])
        audits = []
        for frame in range(boundary+1, len(seed['q'])):
            # Once the residual has completely decayed, retain the old target.
            if frame >= timing['contact_end']:
                continue
            scene.set_state(seed['q'][frame], old_p[frame], seed['object_wxyz'][frame])
            palm_p = scene.data.xpos[grasp.bodies].copy()
            palm_r = Rotation.from_matrix(scene.data.xmat[grasp.bodies].reshape(2, 3, 3))
            r = Rotation.from_quat(plan_r[frame], scalar_first=True)
            clearance = .055*(1-smoothstep((frame-timing['close_start'])/
                                           (timing['contact_start']-timing['close_start'])))
            # Explicit acquisition margin; close the final 5mm while force ramps.
            clearance += .005*(1-smoothstep((frame-timing['contact_start'])/
                                            (timing['force_full']-timing['contact_start'])))
            lift = .06*(1-smoothstep((frame-boundary)/(timing['contact_start']-boundary)))
            desired_p = plan_p[frame]+r.apply(grasp.local_p+clearance*grasp.outward)+[0., 0., lift]
            desired_r = Rotation.concatenate([r*x for x in grasp.local_r])
            approach_blend = smoothstep((frame-boundary)/(timing['close_start']-boundary))
            targets = ((1-approach_blend)*palm_p+approach_blend*desired_p,
                Rotation.from_rotvec(approach_blend*(desired_r*palm_r.inv()).as_rotvec())*palm_r)
            command['q'][frame], audit = grasp.solve(seed['q'][frame], plan_p[frame], plan_r[frame],
                command['q'][frame-1], palm_targets=targets, finger_q=seed['q'][frame],
                arm_neighborhood=.35, seconds=.3)
            audits.append(dict(frame=frame, **audit))
        report['ik'] = audits
        report['approach_model'] = dict(
            scope='Object remains at preceding predicted settled pose until acquisition; no state write',
            blend_to_new_hand_targets_frames=[boundary, timing['close_start']],
            final_clearance_before_force_ramp_m=.005,
            force_ramp_frames=[timing['contact_start'], timing['force_full']],
            preceding_prediction_600_to_610_drift_m=float(np.linalg.norm(
                prediction['object_positions'][610]-prediction['object_positions'][boundary])))
        frames = np.arange(len(seed['q']))[boundary:]
        blend = smoothstep((frames-timing['contact_start'])/(timing['force_full']-timing['contact_start']))
        blend *= smoothstep((timing['force_end']-frames)/(timing['force_end']-timing['contact_end']))
        forces, allocation = allocate_profile(scene, phase_points, command['q'][boundary:],
            plan_p[boundary:], plan_r[boundary:], seed['timestamps'][boundary:], blend,
            friction=saved['controller']['friction'], preload=2.)
        for channel, point in enumerate(phase_points):
            command['force_ff'][boundary+1:, index[point.body]] = forces[1:, channel]
        report['force_profile'] = allocation
        # Retain seed spline metadata for exact coefficient-prefix elimination.
        # New q is the tail fit objective; the old derivatives are a warm model.
        np.savez_compressed(args.output/'unrefined_command.npz', **command)
        command, optimization = refine_commands(scene, command, points, seconds=args.seconds,
            free_trust_rad=.35, continuous_bounds=True, freeze_prefix_frame=boundary)
        report['optimization'] = optimization
        if command is None:
            report['status'] = 'rejected_optimization'
            return
        prefix = {k: bool(np.array_equal(command[k][:boundary+1], seed[k][:boundary+1])) for k in [
            'q', 'command_velocity', 'command_acceleration', 'force_ff', 'contact_active',
            'robot_local_points', 'object_local_points', 'inward_normals_local']}
        common = {k: bool(np.array_equal(command[k], seed[k])) for k in [
            'timestamps', 'object_positions', 'object_wxyz', 'initial_q', 'initial_velocity',
            'baseline_command_q', 'contact_active']}
        common['root'] = bool(np.array_equal(command['q'][:, :7], seed['q'][:, :7]))
        _, bounds = apply_offset(scene, command, np.zeros((len(scene.movable), 1)), [0.])
        path = audit_path(scene, command)
        # Additional approach audit uses the predicted planning poses. It ends
        # before contact is scheduled, with the same clearance gate as free paths.
        planning = dict(command, object_positions=plan_p, object_wxyz=plan_r)
        approach = audit_path(scene, planning, start=boundary, end=timing['contact_start']-1)
        report.update(prefix_checks=prefix, common_checks=common, command_bounds=bounds,
                      free_path=path, predicted_approach=approach)
        accepted = all(prefix.values()) and all(common.values()) and bounds['accepted'] and path['accepted'] and approach['accepted']
        report['status'] = 'command_only' if accepted else 'rejected_before_replay'
        name = 'reference.npz' if accepted else 'diagnostic_rejected_reference.npz'
        np.savez_compressed(args.output/name, **command)
        np.savez_compressed(args.output/'planning_poses.npz', object_positions=plan_p, object_wxyz=plan_r)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-start
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        write()
        print(json.dumps({k: v for k, v in report.items() if k not in [
            'ik', 'contacts', 'phase_contacts', 'provenance', 'physics', 'source_hashes']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=45.)
    run(parser.parse_args())
