"""Correct and physically replay six supported EEF contacts on the Tier 1 book."""
import argparse
from pathlib import Path

from correction.src.preprocessing.supported_reference import load_supported_reference
from correction.src.optimization.supported_manipulation import run, metrics, accepted
from correction.src.dynamics.supported_contacts import SupportedContacts
from correction.src.pass_traj.supported_reference import save_supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('correction/outputs/reorganization_check_20261017/retargeted/contact_aware.npz'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/supported_book_local_20261019'))
    parser.add_argument('--start', type=int, default=520)
    parser.add_argument('--stop', type=int, default=566, help='exclusive')
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--budget', type=float, default=45.)
    parser.add_argument('--verify', action='store_true', help='Replay unchanged commands under held-out parameters')
    args = parser.parse_args()
    if args.iterations < 0 or args.budget <= 0:
        parser.error('iterations must be nonnegative and budget must be positive')
    ref = load_supported_reference(args.input, args.start, args.stop)
    report, records, plan, xml = run(ref, iterations=args.iterations, budget=args.budget)
    if args.verify:
        checks = {}
        for name, parameters in [('half_timestep', dict(dt=.0005)),
                                 ('mass_0.9kg', dict(mass=.9)), ('mass_1.1kg', dict(mass=1.1)),
                                 ('friction_0.5', dict(friction=.5))]:
            scene = SupportedContacts(ref.bounds, table=ref.table, **parameters)
            record = scene.rollout(ref.times, records['corrected']['targets'], ref.positions,
                                   ref.quaternions, initial_pads=plan['initial_pads'])
            checks[name] = dict(parameters=parameters, accepted=accepted(record), **metrics(record, ref.times))
            records[f'verify_{name}'] = record
        report['held_out_replays_without_reoptimization'] = checks
    save_supported(args.output, ref, report, records, plan, xml)
    for name in ('baseline', 'geometry_only', 'wrench_seed', 'corrected'):
        m = report[name]
        print(f"{name}: max position {m['max_position_error_mm']:.2f} mm; "
              f"rotation {m['max_rotation_error_deg']:.2f} deg; penetration {m['max_penetration_mm']:.3f} mm", flush=True)
    print(f"Accepted supported contact demo: {report['accepted_supported_contact_demo']}; "
          f"{report['method_seconds']:.2f} s CPU; {report['rollout_evaluations']} rollouts", flush=True)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
