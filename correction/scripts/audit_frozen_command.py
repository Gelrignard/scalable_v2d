"""Audit interpolated free motion and baseline compatibility before replay."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.finger_transition import audit_path
from correction.src.optimization.servo_preload import apply_offset
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh audit output directory')
    started = perf_counter()
    cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    old = json.loads((args.baseline/'report.json').read_text(encoding='utf-8'))
    reference = read_npz(args.source/'reference.npz')
    baseline = read_npz(args.baseline/'reference.npz')
    _, _, _, scene = load_strategy_scene(saved['input_directory'])
    args.output.mkdir(parents=True)
    # These arrays determine the common task and physical initial conditions;
    # the corrected upper-body commands and contact channels should differ.
    common = ['timestamps', 'object_positions', 'object_wxyz', 'initial_q',
              'initial_velocity', 'baseline_command_q']
    checks = {key: bool(key in reference and key in baseline and
                        np.array_equal(reference[key], baseline[key])) for key in common}
    checks['command_root'] = bool(np.array_equal(reference['q'][:, :7], baseline['q'][:, :7]))
    for key in ['input_directory', 'physics', 'kp_hand']:
        checks[key] = saved[key] == old[key]
    # Historical reports predate the optional engagement filter. Its original
    # behavior is exactly the declared zero-filter branch, verified separately
    # by the instrumentation-equivalence replays; do not infer other defaults.
    controllers = [dict(item['controller']) for item in [saved, old]]
    for controller in controllers:
        controller.setdefault('engagement_seconds', 0.)
    checks['controller'] = controllers[0] == controllers[1]
    geometry = audit_path(scene, reference)
    _, command_bounds = apply_offset(scene, reference, np.zeros((len(scene.movable), 1)), [0.])
    report = dict(
        source=str(args.source.resolve()), baseline=str(args.baseline.resolve()),
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        baseline_reference_sha256=hashlib.sha256((args.baseline/'reference.npz').read_bytes()).hexdigest(),
        common_task_checks=checks, inactive_path=geometry, command_bounds=command_bounds,
        normalized_controllers=controllers,
        compatibility_note='Missing historical engagement_seconds means the original unfiltered behavior (0 seconds)',
        passed=all(checks.values()) and geometry['accepted'] and command_bounds['accepted'],
        scope='Free interval525–600 geometry, full-command joint/speed samples and declared shared contract; not physical acceptance',
        provenance=snapshot_sources(args.output, [Path(__file__),
            Path('correction/src/optimization/finger_transition.py'),
            Path('correction/src/optimization/servo_preload.py')]),
        wall_seconds=perf_counter()-started, cost=process_cost())
    report['cost']['cpu_seconds'] -= cpu
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
