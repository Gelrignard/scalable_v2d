"""Compare the immutable-reference surface-feedback ablations."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import read_npz
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Preserve existing comparisons')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    seed_ref = read_npz(args.baseline/'reference.npz')
    seed_result = read_npz(args.baseline/'mechanics.npz')
    base_report = json.loads((args.baseline/'report.json').read_text(encoding='utf-8'))
    rows = []
    for directory in [args.baseline, *args.candidates]:
        command = read_npz(directory/'reference.npz')
        execution = read_npz(directory/'mechanics.npz')
        report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
        if command.keys() != seed_ref.keys():
            raise AssertionError('Reference schema changed')
        for key in command:
            np.testing.assert_array_equal(command[key], seed_ref[key])
        for key in ['q', 'joint_velocities', 'object_positions', 'object_wxyz']:
            np.testing.assert_array_equal(execution[key][:661], seed_result[key][:661])
        for key in ['input_directory', 'physics', 'controller', 'kp_hand', 'inertia_ff_scale']:
            if report[key] != base_report[key]:
                raise AssertionError('Base model/controller contract differs: '+key)
        if execution['max_object_applied_force'] != 0 or len(execution['q']) != len(seed_ref['q']):
            raise AssertionError('A full unassisted episode is required')
        contact = {}
        for channel, point in enumerate(report['contacts']):
            selected = np.flatnonzero(execution['controller_touching'][645:726, channel])+645
            if np.any(command['contact_active'][645:726, channel]):
                contact[str(execution['robot_body_names'][point['body']])] = dict(
                    sampled_touching_frames=len(selected),
                    last_sampled_touching_frame=int(selected[-1]) if len(selected) else None)
        rows.append(dict(name=directory.name, metrics=report['cases']['mechanics'],
            final_position_error_mm=float(1000*execution['position_error'][-1]),
            final_rotation_error_deg=float(np.rad2deg(execution['rotation_error'][-1])),
            second_contact_samples=contact, surface_feedback=report.get('surface_feedback'),
            followed_joint_names=report.get('followed_joint_names'),
            wall_seconds=report['wall_seconds'], cost=report['cost'],
            source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                directory/'report.json', directory/'reference.npz', directory/'mechanics.npz']}))
    args.output.mkdir(parents=True)
    result = dict(status='complete', verified_identical_reference_and_physical_prefix_through_660=True,
        note='Additional perfect-object-pose feedback is the declared controller variable. All rows fail task acceptance; no policy-cost claim.',
        rows=rows, provenance=snapshot_sources(args.output, [Path(__file__)]),
        wall_seconds=perf_counter()-start, cost=process_cost())
    result['cost']['cpu_seconds'] -= cpu
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([{k: row[k] for k in ['name', 'final_position_error_mm', 'final_rotation_error_deg',
        'second_contact_samples', 'followed_joint_names', 'wall_seconds']} for row in rows], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
