"""Try a complete recorded pickup with the coupled G1/box-object optimizer."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from correction.src.preprocessing.g1_horizon import load_book_pickup
from correction.src.optimization.coupled_horizon import HorizonSettings, solve_horizon
from correction.src.dynamics.horizon_audit import audit_horizon
from correction.src.dynamics.horizon_collision import audit_robot_geometry
from correction.src.dynamics.g1_horizon_replay import replay_g1
from correction.src.pass_traj.horizon_result import save_horizon
from correction.src.pass_traj.controller_reference import save_controller_reference
from correction.src.optimization.horizon_initialization import initialize_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('correction/outputs/tier1_book/retargeted/contact_aware.npz'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/coupled_book_pickup'))
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--exact-hessian', action='store_true', help='Expensive full G1 second derivatives; quasi-Newton is the CPU default')
    parser.add_argument('--warm-start', type=Path, help='Saved plan at the same knots, used only as an initial optimizer guess')
    parser.add_argument('--collision-passes', type=int, default=1)
    parser.add_argument('--hypothesis', choices=('pinch_thickness', 'two_hand_edges', 'opposed_faces'), default='pinch_thickness')
    parser.add_argument('--initialize-contacts', action='store_true', help='Geometry-only warm start, followed by the full coupled solve')
    parser.add_argument('--controller-aware-position', action='store_true', help='Constrain servo offsets and limits in the coupled optimization')
    parser.add_argument('--compare-controllers', action='store_true', help='Run optional controller variants after a converged solve')
    args = parser.parse_args()
    started = perf_counter()
    ref, robot, scene, raw, ids = load_book_pickup(args.input, stride=args.stride, hypothesis=args.hypothesis)
    warm = None
    if args.warm_start:
        with np.load(args.warm_start) as archive:
            warm = {key: archive[key].copy() for key in archive.files}
        if not np.array_equal(warm['times'], ref.times):
            raise ValueError('Warm-start timestamps must match this problem')
    initialization = None
    if args.initialize_contacts:
        warm, initialization = initialize_contacts(ref, robot, scene)
    print(f'Coupled G1 pickup: {len(ids)} knots, {robot.n} joints, {len(ref.contacts)} contacts', flush=True)
    settings = HorizonSettings(max_seconds=args.seconds, max_iterations=args.iterations,
                               print_level=5 if args.verbose else 0, exact_hessian=args.exact_hessian)
    kp = np.where(scene.hand, 20., 300.)
    offset = np.where(scene.hand, .10, .08)
    if args.controller_aware_position:
        settings.servo_kp, settings.servo_max_offset = tuple(kp), tuple(offset)
    result = solve_horizon(ref, robot, settings, warm_start=warm)
    report = dict(scope='Recorded first pickup, real G1 dynamics, assumed rigid book box and supported pelvis',
                  accepted_physics=False, full_robot_collision_checked=False, initialization=initialization)
    records = {}
    history = []
    report['audit'] = audit_horizon(ref, robot, result)
    collision, cuts = audit_robot_geometry(ref, robot, scene, result)
    report['collision'] = collision
    report['diagnostic_only'] = not result['success']
    report['full_robot_collision_checked'] = True
    if result['success']:
        for _ in range(args.collision_passes):
            if collision['accepted_geometry']: break
            history.append(dict(collision=collision, solver_seconds=result['seconds']))
            trial = solve_horizon(ref, robot, settings, collision_cuts=cuts, warm_start=result)
            if not trial['success']:
                history.append(dict(rejected_collision_solve=trial['stages']))
                break
            result = trial
            collision, additional = audit_robot_geometry(ref, robot, scene, result)
            cuts.extend(additional)
        report['audit'] = audit_horizon(ref, robot, result)
        report['collision'] = collision
        records, replay_report = replay_g1(ref, robot, scene, raw, result, controller_variants=args.compare_controllers)
        report.update(replay_report)
        report['accepted_physics'] = bool(report['audit']['accepted_nominal'] and collision['accepted_geometry']
                                          and replay_report['accepted_correction'])
        report['full_robot_collision_checked'] = True
    report['collision_history'] = history
    report['total_experiment_seconds'] = perf_counter()-started
    report['warm_start'] = str(args.warm_start) if args.warm_start else None
    path = save_horizon(args.output, ref, result, report, records)
    # Diagnostic output remains separate even for a rejected optimizer iterate.
    save_controller_reference(args.output, ref, robot, result, kp=kp, max_offset=offset)
    print(json.dumps(report, indent=2), flush=True)
    print(json.dumps({key: value for key, value in result.items() if not hasattr(value, 'shape')}, indent=2), flush=True)
    print(path.resolve(), flush=True)


if __name__ == '__main__': main()
