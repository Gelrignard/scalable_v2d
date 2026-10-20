"""Verify the full saved local comparison and separate ranking from success."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import read_npz
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    output = args.directory/'comparison_audit.json'
    if output.exists():
        raise ValueError('Preserve the existing comparison audit')
    began = perf_counter(); cpu = process_cost()['cpu_seconds']
    report = json.loads((args.directory/'report.json').read_text(encoding='utf-8'))
    if report['status'] not in ('complete', 'budget_exhausted'):
        raise ValueError('Wait for the active search to finish')
    source = Path(report['source'])
    seed = read_npz(source/'reference.npz')
    original_execution = read_npz(source/'mechanics.npz')
    rows = []
    for name, row in report['trials'].items():
        if row['status'] != 'complete':
            continue
        root = args.directory/name
        command, result = read_npz(root/'reference.npz'), read_npz(root/'mechanics.npz')
        for field in ('initial_q', 'initial_velocity', 'timestamps', 'object_positions',
                      'object_wxyz', 'force_ff', 'contact_active', 'baseline_command_q'):
            np.testing.assert_array_equal(command[field], seed[field])
        np.testing.assert_array_equal(command['q'][:, :7], seed['q'][:, :7])
        np.testing.assert_array_equal(command['q'][:192], seed['q'][:192])
        np.testing.assert_array_equal(command['q'][525:], seed['q'][525:])
        for field in ('q', 'joint_velocities', 'object_positions', 'object_wxyz'):
            np.testing.assert_array_equal(result[field][0], original_execution[field][0])
        if len(result['q']) != len(seed['q']) or result['max_object_applied_force'] != 0:
            raise AssertionError('Every compared trial must be complete and unassisted')
        metric = row['cases']['mechanics']
        rows.append(dict(
            trial=name, method=row['method'], parameters=row['parameters'],
            objective=row['objective'], physical_acceptance=metric['accepted_physics'],
            rms_position_mm=metric['rms_position_error_mm'],
            rms_rotation_deg=metric['rms_rotation_error_deg'],
            max_requested_torque_fraction=metric['max_requested_torque_fraction'],
            max_joint_limit_violation_rad=metric['max_joint_limit_violation_rad'],
            intervals=metric['intervals'], wall_seconds=row['wall_seconds'],
            cpu_seconds=row['cost']['cpu_seconds'],
            reference_sha256=hashlib.sha256((root/'reference.npz').read_bytes()).hexdigest(),
            execution_sha256=hashlib.sha256((root/'mechanics.npz').read_bytes()).hexdigest()))
    groups = {}
    for method in ('load_informed', 'uninformed'):
        trials = [row for row in rows if row['method'] == method]
        groups[method] = dict(
            complete_trials=len(trials), accepted_trials=sum(row['physical_acceptance'] for row in trials),
            best_ranked=min(trials, key=lambda row: row['objective']['total'])['trial'] if trials else None,
            best_position=min(trials, key=lambda row: row['rms_position_mm'])['trial'] if trials else None,
            replay_wall_seconds=sum(row['wall_seconds'] for row in trials),
            replay_cpu_seconds=sum(row['cpu_seconds'] for row in trials))
    result = dict(
        verified_shared_initial_task_clock_root_force_and_inactive_commands=True,
        source=str(args.directory.resolve()), baseline=report['baseline'], rows=rows, groups=groups,
        search_total_wall_seconds=report['wall_seconds'], search_total_cost=report['cost'],
        mechanics_proposal_seconds=report['load_proposal']['construction_seconds'],
        interpretation='One local seed, three proposals per method; no statistical, raw-retargeting or policy-cost claim',
        ranking_does_not_imply_physical_success=True,
        provenance=snapshot_sources(args.directory/'comparison_audit_provenance', [Path(__file__)]),
        wall_seconds=perf_counter()-began, cost=process_cost())
    result['cost']['cpu_seconds'] -= cpu
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(groups=groups, rows=rows), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args())
