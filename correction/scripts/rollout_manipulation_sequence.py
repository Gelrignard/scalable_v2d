"""Bounded two-phase bin experiment with an uninterrupted dynamic object."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.preprocessing.supported_fixture import regularize_fixture
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.surface_reference import PhaseTiming
from correction.src.optimization.manipulation_sequence import ContactPhase, build_sequence
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def select_phase(directory, frame, family, timing):
    report = json.loads((directory/'report.json').read_text())
    candidates = [c for c in report['candidates'] if c['frame'] == frame and c['family'] == family
                  and c['candidate_for_acquisition'] and not c['static_hold_screen']['floor_contacts_included']]
    if len(candidates) != 1:
        raise ValueError('Requested phase must have one accepted unsupported candidate')
    candidate = candidates[0]
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in candidate['contacts']]
    q = read_npz(directory/'geometry_candidates.npz')['q'][candidate['solution_index']]
    return ContactPhase(points, q, frame, np.asarray(candidate['static_hold_screen']['forces']), timing), report


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    start = perf_counter(); initial_cpu = process_cost()['cpu_seconds']
    first, first_report = select_phase(args.first_candidates, args.first_frame, args.first_family,
        PhaseTiming(168, 192, 235, 270, 480, 501, 525, 550))
    second, second_report = select_phase(args.second_candidates, args.second_frame, args.second_family,
        PhaseTiming(600, 612, args.second_contact_start, 660, 725, 746, 770, 857))
    if first_report['source'] != second_report['source'] or first_report['contract'] != second_report['contract']:
        raise ValueError('Both phases must share the same model/controller contract')
    ref, raw, source_report, scene = load_strategy_scene(first_report['source'])
    original_root = raw['q'][:, :7].copy()
    raw['q'], fixture_report = regularize_fixture(raw['q'], 21)
    args.output.mkdir(parents=True)
    provenance = snapshot_sources(args.output, [Path(__file__), *[
        Path('correction/src')/name for name in (
            'optimization/manipulation_sequence.py', 'optimization/surface_reference.py',
            'optimization/rigid_grasp.py', 'optimization/actuator_force_profile.py',
            'dynamics/supported_robot.py', 'dynamics/contact_controller.py',
            'dynamics/robot_scene.py', 'preprocessing/strategy_bundle.py', 'preprocessing/supported_fixture.py')]])
    config = ContactControllerConfig(interface='torque', feedback_gain=.25,
                                      friction=first_report['contract']['contact']['friction'],
                                      patch_distance_gate=args.patch_distance_gate, near_surface_m=args.capture_gap)
    report = dict(status='building_reference', scope='two-phase supported-pelvis local prototype',
        input_directory=first_report['source'], physics=source_report['physics'],
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        candidate_hashes=[hashlib.sha256((d/'report.json').read_bytes()).hexdigest()
                          for d in (args.first_candidates, args.second_candidates)],
        fixture_regularization=fixture_report, controller=asdict(config), provenance=provenance,
        kp_hand=args.kp_hand, inertia_ff_scale=1.,
        frames=len(raw['q']), duration_s=float(raw['timestamps'][-1]-raw['timestamps'][0]),
        object_assistance=False, resets_during_episode=0, cases={}, ready_for_training=False)
    report['phase_contacts'] = [dict(frame=p.frame, timing=asdict(p.timing), contacts=[asdict(x) for x in p.points])
                                for p in (first, second)]
    second_shift = None
    if args.second_shift_from:
        predicted = read_npz(args.second_shift_from/'mechanics.npz')
        saved_reference = read_npz(args.second_shift_from/'reference.npz')
        np.testing.assert_array_equal(saved_reference['object_positions'], raw['object_positions'])
        if len(predicted['q']) != len(raw['q']) or predicted['max_object_applied_force'] != 0:
            raise ValueError('Translation update requires a prior unassisted full physical replay')
        second_shift = predicted['object_positions'][600]-raw['object_positions'][600]
        if np.linalg.norm(second_shift) > .12:
            raise ValueError('Prior free-state deviation exceeds the declared 120mm correction radius')
        report['prediction_update'] = dict(frame=600, translation_m=second_shift,
            sha256=hashlib.sha256((args.second_shift_from/'mechanics.npz').read_bytes()).hexdigest(),
            scope='offline command optimization using prior simulated state; no object state write or score-target change')
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2,
            default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v)+'\n')
    write()
    try:
        points, command, command_report = build_sequence(scene, raw, [first, second],
            friction=config.friction, transition_lift=args.transition_lift, preload=args.preload,
            second_shift=second_shift, arm_neighborhood=args.arm_neighborhood)
        report.update(status='replaying', command=command_report, contacts=[asdict(p) for p in points])
        write()
        np.savez_compressed(args.output/'reference.npz', **command,
            object_positions=raw['object_positions'], object_wxyz=raw['object_wxyz'],
            baseline_command_q=raw['q'], initial_q=ref['initial_q'], initial_velocity=ref['initial_velocity'],
            object_mesh=ref['object_mesh'], bounds=ref['bounds'], floor_z=ref['floor_z'],
            original_input_root=original_root, frame_ids=np.arange(len(raw['q'])))
        if args.plan_only:
            report['status']='command_only'
            return
        initial = None
        for name in args.cases:
            before = perf_counter(); print('Full episode: '+name, flush=True)
            result = scene.rollout(raw['q'] if name == 'original' else command['q'],
                raw['object_positions'], raw['object_wxyz'], command['timestamps'],
                force_ff=command['force_ff'] if name == 'mechanics' else np.zeros_like(command['force_ff']),
                force_points=points, contact_controller=config,
                contact_normals_local=command['inward_normals_local'],
                contact_active=command['contact_active'] if name == 'mechanics' else np.zeros_like(command['contact_active']),
                point_local_trajectory=command['robot_local_points'],
                object_local_trajectory=command['object_local_points'],
                command_velocity=None if name == 'original' else command['command_velocity'],
                kp_arm=300., kp_hand=args.kp_hand, inertia_ff=True, smooth_fixture=True,
                initial_q=ref['initial_q'], initial_velocity=ref['initial_velocity'])
            state = np.r_[result['q'][0], result['joint_velocities'][0], result['object_positions'][0], result['object_wxyz'][0]]
            if initial is None: initial = state
            np.testing.assert_array_equal(initial, state)
            if len(result['q']) != len(raw['q']) or result['max_object_applied_force'] != 0:
                raise AssertionError('Full uninterrupted unassisted physics required')
            np.savez_compressed(args.output/f'{name}.npz', **result)
            summary = assessed(result, command['timestamps'])
            summary['intervals'] = {}
            for label, a, b in [('first', 235, 480), ('free', 525, args.second_contact_start-1),
                                ('second', args.second_contact_start, 725), ('final', 770, 857)]:
                summary['intervals'][label] = dict(
                    rms_position_mm=float(1000*np.sqrt(np.mean(result['position_error'][a:b+1]**2))),
                    rms_rotation_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error'][a:b+1]**2)))),
                    max_requested_torque_fraction=float(result['frame_peak_torque_fraction'][a:b+1].max()),
                    max_joint_limit_violation_rad=float(result['frame_peak_joint_limit_violation_rad'][a:b+1].max()),
                    max_penetration_m=float(result['frame_peak_penetration_m'][a:b+1].max()))
            summary['wall_seconds'] = perf_counter()-before
            report['cases'][name] = summary; write()
            print(json.dumps(summary), flush=True)
        report.update(status='complete', initial_state_identical=True if len(args.cases)>1 else None)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-start
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= initial_cpu
        write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-candidates', type=Path, required=True)
    parser.add_argument('--first-frame', type=int, default=430)
    parser.add_argument('--first-family', default='observed_six')
    parser.add_argument('--second-candidates', type=Path, required=True)
    parser.add_argument('--second-frame', type=int, default=680)
    parser.add_argument('--second-family', default='centered_six')
    parser.add_argument('--second-contact-start', type=int, default=635)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=['original', 'geometry', 'mechanics'], default=['mechanics'])
    parser.add_argument('--transition-lift', type=float, default=.15)
    parser.add_argument('--preload', type=float, default=2.)
    parser.add_argument('--arm-neighborhood', type=float, default=.9)
    parser.add_argument('--second-shift-from', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--kp-hand', type=float, default=20.)
    parser.add_argument('--patch-distance-gate', action='store_true')
    parser.add_argument('--capture-gap', type=float, default=.002)
    run(parser.parse_args())
