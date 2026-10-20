"""Verify the physical prefix and common task after an offline phase update."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import read_npz
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    path = args.directory/'prediction_replay_audit.json'
    if path.exists():
        raise ValueError('Preserve the existing audit')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    report = json.loads((args.directory/'report.json').read_text(encoding='utf-8'))
    construction = json.loads((Path(report['source'])/'report.json').read_text(encoding='utf-8'))
    source = Path(construction['source'])
    old_report = json.loads((source/'report.json').read_text(encoding='utf-8'))
    seed, new = read_npz(source/'reference.npz'), read_npz(args.directory/'reference.npz')
    first, second = read_npz(source/'mechanics.npz'), read_npz(args.directory/'mechanics.npz')
    boundary = construction['frozen_prefix_frame']
    checks = {}
    for key in ['timestamps', 'object_positions', 'object_wxyz', 'initial_q', 'initial_velocity', 'baseline_command_q']:
        checks['common_'+key] = bool(np.array_equal(seed[key], new[key]))
    for key in ['controller', 'kp_hand', 'inertia_ff_scale', 'physics', 'input_directory']:
        checks['contract_'+key] = report[key] == old_report[key]
    checks['root'] = bool(np.array_equal(new['q'][:, :7], seed['q'][:, :7]))
    differences = {}
    for key in ['q', 'joint_velocities', 'object_positions', 'object_wxyz']:
        checks['physical_prefix_'+key] = bool(np.array_equal(first[key][:boundary+1], second[key][:boundary+1]))
        differences[key] = float(np.max(abs(first[key][:boundary+1]-second[key][:boundary+1])))
    checks['full_episode'] = len(second['q']) == len(seed['q'])
    checks['unassisted'] = bool(second['max_object_applied_force'] == 0 and report['object_assistance'] is False)
    checks['no_resets'] = report['resets_during_episode'] == 0
    audit = dict(checks=checks, passed=all(checks.values()), maximum_prefix_differences=differences,
        frozen_prefix_frame=boundary, source=str(source.resolve()), scope='Exact physical prefix and common protocol; not task acceptance',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            source/'reference.npz', source/'mechanics.npz', args.directory/'reference.npz', args.directory/'mechanics.npz']},
        provenance=snapshot_sources(args.directory/'prediction_audit_provenance', [Path(__file__)]),
        wall_seconds=perf_counter()-start, cost=process_cost())
    audit['cost']['cpu_seconds'] -= cpu
    path.write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args())
