"""Bounded C2 tail SQP with actual inactive-approach collision constraints."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.command_smoothing import refine_commands
from correction.src.optimization.finger_transition import audit_path
from correction.src.optimization.servo_preload import apply_offset
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh directory')
    started = perf_counter(); cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    if saved['status'] != 'rejected_before_replay':
        raise ValueError('This repair expects a saved, numerically solved but rejected approach')
    command = read_npz(args.source/'diagnostic_rejected_reference.npz')
    seed = read_npz(Path(saved['source'])/'reference.npz')
    plans = read_npz(args.source/'planning_poses.npz')
    _, _, _, scene = load_strategy_scene(saved['input_directory'])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in saved['contacts']]
    boundary = saved['frozen_prefix_frame']; end = saved['phase_contacts'][1]['timing']['contact_start']-1
    obstacle = dict(plans, frames=[boundary, end])
    args.output.mkdir(parents=True)
    report = {k: saved[k] for k in ['input_directory', 'controller', 'physics', 'contacts', 'phase_contacts',
        'kp_hand', 'inertia_ff_scale', 'initialization_protocol', 'source', 'frozen_prefix_frame', 'prediction_update', 'approach_model']}
    report.update(status='optimizing', rejected_approach_source=str(args.source.resolve()),
        source_reference_sha256=hashlib.sha256((args.source/'diagnostic_rejected_reference.npz').read_bytes()).hexdigest(),
        source_planning_sha256=hashlib.sha256((args.source/'planning_poses.npz').read_bytes()).hexdigest(),
        cases={}, iterations=[], ready_for_training=False, object_assistance=False, resets_during_episode=0,
        configuration=dict(max_iterations=3, seconds_per_solve=20., total_budget_seconds=120.,
                           clearance_m=.002, collision_audit_hz=150),
        provenance=snapshot_sources(args.output, [Path(__file__), *[
            Path('correction/src')/p for p in ['optimization/command_smoothing.py',
                'optimization/spline_clearance.py', 'optimization/finger_transition.py',
                'optimization/servo_preload.py', 'dynamics/collision_distance.py']]]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    write()
    try:
        remembered = []
        for iteration in range(3):
            if perf_counter()-started > 120:
                report['status'] = 'budget_exhausted'
                break
            refined, audit = refine_commands(scene, command, points, seconds=20., free_trust_rad=.35,
                continuous_bounds=True, freeze_prefix_frame=boundary, clearance_obstacle=obstacle,
                remembered_pairs=remembered)
            row = dict(iteration=iteration, optimization=audit)
            report['iterations'].append(row)
            if refined is None:
                report['status'] = 'rejected_optimization'; break
            command = refined
            remembered = audit['clearance_linearization']['remembered_pairs']
            planning = dict(command, **plans)
            geometry = audit_path(scene, planning, start=boundary, end=end)
            _, bounds = apply_offset(scene, command, np.zeros((len(scene.movable), 1)), [0.])
            prefix = {key: bool(np.array_equal(command[key][:boundary+1], seed[key][:boundary+1]))
                for key in ['q', 'command_velocity', 'command_acceleration', 'force_ff',
                            'contact_active', 'robot_local_points', 'object_local_points', 'inward_normals_local']}
            common = {key: bool(np.array_equal(command[key], seed[key]))
                for key in ['timestamps', 'object_positions', 'object_wxyz', 'initial_q',
                            'initial_velocity', 'baseline_command_q', 'contact_active']}
            common['root'] = bool(np.array_equal(command['q'][:, :7], seed['q'][:, :7]))
            row.update(geometry=geometry, bounds=bounds, prefix_checks=prefix, common_checks=common)
            np.savez_compressed(args.output/f'diagnostic_iteration_{iteration}.npz', **command)
            if geometry['accepted'] and bounds['accepted'] and all(prefix.values()) and all(common.values()):
                np.savez_compressed(args.output/'reference.npz', **command)
                np.savez_compressed(args.output/'planning_poses.npz', **plans)
                report['status'] = 'command_only'; break
            report['status'] = 'rejected_before_replay'; write()
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-started
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        write()
        print(json.dumps({k: v for k, v in report.items() if k not in [
            'contacts', 'phase_contacts', 'provenance', 'physics']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
