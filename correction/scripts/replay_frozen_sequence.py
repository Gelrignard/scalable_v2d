"""Controller ablations on an immutable full-episode command reference."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def run(args):
    if args.output.exists(): raise ValueError('Use a fresh output directory')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    prior = json.loads((args.source/'report.json').read_text())
    reference = read_npz(args.source/'reference.npz')
    _, _, _, scene = load_strategy_scene(prior['input_directory'])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in prior['contacts']]
    options = dict(prior['controller'], patch_distance_gate=args.patch_distance_gate, near_surface_m=args.capture_gap)
    if args.engagement_seconds is not None:
        options['engagement_seconds'] = args.engagement_seconds
    config = ContactControllerConfig(**options)
    args.output.mkdir(parents=True)
    report = dict(status='running', source=str(args.source.resolve()), input_directory=prior['input_directory'],
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        controller=asdict(config), inertia_ff_scale=args.inertia_scale, kp_hand=args.kp_hand,
        contact_body_audit=args.audit_contact_bodies,
        initialization_protocol=prior.get('initialization_protocol','source physical state unchanged'),
        contacts=prior['contacts'], phase_contacts=prior.get('phase_contacts'), physics=prior['physics'],
        cases={}, ready_for_training=False, object_assistance=False, resets_during_episode=0,
        provenance=snapshot_sources(args.output, [Path(__file__), Path('correction/src/dynamics/contact_controller.py'),
                                                 Path('correction/src/dynamics/contact_measurement.py'),
                                                 Path('correction/src/dynamics/torque_audit.py'),
                                                 Path('correction/src/dynamics/supported_robot.py')]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    write()
    np.savez_compressed(args.output/'reference.npz', **reference)
    try:
        initial = None
        for case in args.cases:
            before = perf_counter()
            result = scene.rollout(reference['baseline_command_q'] if case=='original' else reference['q'],
                reference['object_positions'], reference['object_wxyz'], reference['timestamps'],
                force_ff=reference['force_ff'] if case=='mechanics' else np.zeros_like(reference['force_ff']),
                force_points=points, contact_controller=config, contact_normals_local=reference['inward_normals_local'],
                contact_active=reference['contact_active'] if case=='mechanics' else np.zeros_like(reference['contact_active']),
                point_local_trajectory=reference.get('robot_local_points'),
                object_local_trajectory=reference.get('object_local_points'),
                command_velocity=None if case=='original' else reference['command_velocity'],
                kp_arm=300., kp_hand=args.kp_hand, inertia_ff=True, inertia_ff_scale=args.inertia_scale, smooth_fixture=True,
                initial_q=reference['initial_q'], initial_velocity=reference['initial_velocity'],
                contact_body_audit=args.audit_contact_bodies)
            state = np.r_[result['q'][0], result['joint_velocities'][0], result['object_positions'][0], result['object_wxyz'][0]]
            if initial is None: initial = state
            np.testing.assert_array_equal(initial, state)
            if len(result['q']) != len(reference['q']) or result['max_object_applied_force'] != 0:
                raise AssertionError('Full uninterrupted unassisted physics required')
            np.savez_compressed(args.output/f'{case}.npz', **result)
            summary = assessed(result, reference['timestamps'])
            summary['intervals'] = {name: dict(
                rms_position_mm=float(1000*np.sqrt(np.mean(result['position_error'][a:b+1]**2))),
                rms_rotation_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error'][a:b+1]**2)))),
                max_requested_torque_fraction=float(result['frame_peak_torque_fraction'][a:b+1].max()))
                for name, a, b in [('first',235,480), ('free',525,644), ('second',645,725), ('final',770,857)]}
            summary['wall_seconds'] = perf_counter()-before
            report['cases'][case] = summary; write()
            print(json.dumps(dict(case=case, **summary)), flush=True)
        report.update(status='complete', initial_state_identical=True if len(args.cases)>1 else None)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-start
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=['original','geometry','mechanics'], default=['mechanics'])
    parser.add_argument('--patch-distance-gate', action='store_true')
    parser.add_argument('--capture-gap', type=float, default=.002)
    parser.add_argument('--inertia-scale', type=float, default=1.)
    parser.add_argument('--kp-hand', type=float, default=20.)
    parser.add_argument('--engagement-seconds', type=float)
    parser.add_argument('--audit-contact-bodies', action='store_true')
    run(parser.parse_args())
