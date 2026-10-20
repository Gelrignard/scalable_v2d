"""Compare geometric initializations; a local IK failure is not infeasibility proof.

This CPU diagnostic runs before spending nonlinear dynamics solver time. It
does not export a corrected trajectory or change any observation/initial state.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from correction.src.preprocessing.g1_horizon import load_book_pickup
from correction.src.optimization.horizon_initialization import initialize_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('correction/outputs/tier1_book/retargeted/contact_aware.npz'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/contact_initialization_audit'))
    parser.add_argument('--evaluations', type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started, reports = perf_counter(), {}
    for name in ('pinch_thickness', 'two_hand_edges', 'opposed_faces'):
        ref, robot, scene, _, _ = load_book_pickup(args.input, stride=12, hypothesis=name)
        warm, report = initialize_contacts(ref, robot, scene, evaluations=args.evaluations)
        np.savez_compressed(args.output/f'{name}_initial_guess.npz', times=ref.times, **warm)
        report['max_contact_error_m'] = max(k['max_contact_error_m'] for k in report['knots'])
        report['max_penetration_m'] = max(k['max_penetration_m'] for k in report['knots'])
        report['geometry_initialization_within_tolerance'] = bool(
            report['max_contact_error_m'] < .001 and report['max_penetration_m'] < .0001)
        report['contact_surface_points'] = ref.metadata['contact_surface_points']
        report['accepted_correction'] = False
        report['failure_is_infeasibility_certificate'] = False
        reports[name] = report
        print(f"{name}: {1000*report['max_contact_error_m']:.2f} mm contact error, "
              f"{1000*report['max_penetration_m']:.2f} mm penetration", flush=True)
    result = dict(scope='Geometry-only initialization of recorded first pickup; no dynamics/replay acceptance',
                  total_seconds=perf_counter()-started, hypotheses=reports)
    path = args.output/'report.json'
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(path.resolve(), flush=True)


if __name__ == '__main__': main()
