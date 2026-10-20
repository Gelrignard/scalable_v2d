"""Freeze a failed sequence, refine its robot dynamics, then replay from frame0."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.command_smoothing import refine_commands
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def run(args):
    if args.output.exists(): raise ValueError('Use a fresh output directory')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.source/'report.json').read_text())
    reference = read_npz(args.source/'reference.npz')
    if args.initialize_rest:
        # A separate explicitly declared bench protocol. All comparison cases
        # must be replayed with this same state; old results are not comparable.
        reference['initial_velocity'] = np.zeros_like(reference['initial_velocity'])
    _, _, _, scene = load_strategy_scene(saved['input_directory'])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in saved['contacts']]
    args.output.mkdir(parents=True)
    report = dict(status='optimizing', source=str(args.source.resolve()), input_directory=saved['input_directory'],
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        contacts=saved['contacts'], phase_contacts=saved.get('phase_contacts'),
        controller=saved['controller'], physics=saved['physics'], cases={}, ready_for_training=False,
        kp_hand=saved.get('kp_hand',20.), inertia_ff_scale=saved.get('inertia_ff_scale',1.),
        initialization_protocol=('robot joints initially at rest; source initial posture and object state unchanged; '
                                 'requires new matched baseline' if args.initialize_rest else 'source physical state unchanged'),
        provenance=snapshot_sources(args.output, [Path(__file__),
            Path('correction/src/optimization/command_smoothing.py'),
            Path('correction/src/dynamics/contact_controller.py'), Path('correction/src/dynamics/supported_robot.py')]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    write()
    try:
        command, audit = refine_commands(scene, reference, points, stride=args.stride, seconds=args.seconds,
                                         diagnose=args.diagnose, free_trust_rad=args.free_trust,
                                         continuous_bounds=args.continuous_bounds)
        report['optimization'] = audit
        if command is None:
            report['status'] = 'rejected_before_replay'
            print(json.dumps(audit), flush=True)
            return
        np.savez_compressed(args.output/'reference.npz', **command)
        if args.plan_only:
            report['status']='command_only'
            print(json.dumps(audit),flush=True)
            return
        report['status'] = 'replaying'; write()
        result = scene.rollout(command['q'], command['object_positions'], command['object_wxyz'], command['timestamps'],
            force_ff=command['force_ff'], force_points=points,
            contact_controller=ContactControllerConfig(**saved['controller']),
            contact_normals_local=command['inward_normals_local'], contact_active=command['contact_active'],
            point_local_trajectory=command['robot_local_points'], command_velocity=command['command_velocity'],
            object_local_trajectory=command.get('object_local_points'),
            kp_arm=300., kp_hand=report['kp_hand'], inertia_ff=True,
            inertia_ff_scale=report['inertia_ff_scale'], smooth_fixture=True,
            initial_q=command['initial_q'], initial_velocity=command['initial_velocity'])
        if len(result['q']) != len(command['q']) or result['max_object_applied_force'] != 0:
            raise AssertionError('Requires an uninterrupted unassisted full episode')
        np.savez_compressed(args.output/'mechanics.npz', **result)
        summary = assessed(result, command['timestamps'])
        summary['intervals'] = {name: dict(
            rms_position_mm=float(1000*np.sqrt(np.mean(result['position_error'][a:b+1]**2))),
            rms_rotation_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error'][a:b+1]**2)))),
            max_requested_torque_fraction=float(result['frame_peak_torque_fraction'][a:b+1].max()))
            for name, a, b in [('first',235,480), ('free',525,644), ('second',645,725), ('final',770,857)]}
        report['cases']['mechanics'] = summary
        report['status'] = 'complete'
        print(json.dumps(summary), flush=True)
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
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--free-trust', type=float)
    parser.add_argument('--initialize-rest', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--continuous-bounds', action='store_true')
    run(parser.parse_args())
