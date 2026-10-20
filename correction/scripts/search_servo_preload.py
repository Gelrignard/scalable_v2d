"""Bounded full-episode comparison of load-informed and uninformed commands.

This is a local comparison from one shared mechanics-derived seed, not a
raw-retargeting baseline or an end-to-end compute claim.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.servo_preload import command_basis, load_error_proposal, apply_offset, objective
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory; inspect active runs before restarting')
    started = perf_counter(); cpu = process_cost()['cpu_seconds']
    source = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    reference = read_npz(args.source/'reference.npz')
    execution = read_npz(args.source/'mechanics.npz')
    if len(execution['q']) != len(reference['q']) or execution['max_object_applied_force'] != 0:
        raise ValueError('Proposal needs a complete unassisted source execution')
    if source['kp_hand'] != 2. or source['inertia_ff_scale'] != 1.:
        raise ValueError('This local proposal assumes the declared kp_hand2/inertia1 contract')
    _, _, _, scene = load_strategy_scene(source['input_directory'])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in source['contacts']]
    controller = ContactControllerConfig(**source['controller'])
    args.output.mkdir(parents=True)
    report = dict(
        status='preparing', source=str(args.source.resolve()),
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        source_execution_sha256=hashlib.sha256((args.source/'mechanics.npz').read_bytes()).hexdigest(),
        shared_seed_scope='Both searches start from the same mechanics-derived geometric and force reference',
        comparison_scope='Local proposal comparison only; no raw-retargeting or end-to-end learning-cost claim',
        budget_seconds=args.seconds, maximum_proposals=6,
        objective_definition='mean[(position/0.05m)^2+(rotation/10deg)^2] + 0.1 mean torque-excess^2 + 100 joint-limit-excess^2/(0.01rad)^2',
        hard_acceptance='Existing assessed physical gates, unchanged; objective only ranks rejected/accepted trials',
        controller=asdict(controller), kp_hand=2., kp_arm=300., inertia_ff_scale=1.,
        basis_definition=['left hand shape <=0.08rad', 'right hand shape <=0.08rad',
                          'left wrist pitch <=0.02rad', 'right wrist pitch <=0.02rad'],
        parameter_box=[-1., 1.], envelope_frames=[192,270,480,525],
        command_bound_backtracking=[1., .5, .25, .125, .0625],
        random_seed=20261020, random_distribution='uniform[-1,1]^4',
        baseline=dict(objective=objective(execution), metrics=source['cases']['mechanics'],
                      prior_full_replay_wall_seconds=source['wall_seconds']),
        trials={}, ready_for_training=False, physical_success=False,
        provenance=snapshot_sources(args.output, [
            Path(__file__), Path('correction/src/optimization/servo_preload.py'),
            Path('correction/src/dynamics/supported_robot.py'),
            Path('correction/src/dynamics/contact_controller.py'),
            Path('correction/src/dynamics/contact_measurement.py'),
            Path('correction/src/dynamics/torque_audit.py')]))

    def save():
        report['wall_seconds'] = perf_counter()-started
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save()  # Configuration and objective are saved before any candidate replay.
    try:
        basis = command_basis(scene, reference)
        _, seed_bounds = apply_offset(scene, reference, basis, np.zeros(4))
        report['seed_command_bounds'] = seed_bounds
        if not seed_bounds['accepted']:
            report['status'] = 'invalid_seed_before_search'
            return
        before = perf_counter()
        prediction, diagnostic = load_error_proposal(scene, reference, execution, points, basis)
        report['load_proposal'] = dict(parameters=prediction.tolist(), audit=diagnostic,
                                      construction_seconds=perf_counter()-before)
        random = np.random.default_rng(20261020).uniform(-1., 1., (3, 4))
        trials = []
        # Interleave the two arms of the comparison to avoid giving one all
        # available time if the total process deadline is reached.
        for i, scale in enumerate((.25, .5, 1.)):
            trials.extend([(f'mechanics_{i}', 'load_informed', scale*prediction),
                           (f'search_{i}', 'uninformed', random[i])])
        np.savez_compressed(args.output/'search_basis.npz', basis=basis, prediction=prediction,
                            uninformed_parameters=random)
        report['proposals'] = {name: dict(method=method, parameters=p.tolist()) for name, method, p in trials}
        report['status'] = 'running'; save()
        for name, method, parameters in trials:
            if perf_counter()-started > args.seconds-120.:
                report['budget_stopped_before'] = name
                break  # Keep enough budget for one intact full rollout.
            trial_start = perf_counter(); trial_cpu = process_cost()['cpu_seconds']
            folder = args.output/name; folder.mkdir()
            attempts = []
            for factor in report['command_bound_backtracking']:
                command, numerical = apply_offset(scene, reference, basis, factor*parameters)
                attempts.append(dict(factor=factor, **numerical))
                if numerical['accepted']:
                    break
            row = dict(method=method, numerical=numerical, status='screened',
                       parameters=(factor*parameters).tolist(), proposed_parameters=parameters.tolist(),
                       bound_backtracking=attempts, input_directory=source['input_directory'],
                       source=str(args.source.resolve()), physics=source['physics'],
                       contacts=source['contacts'], phase_contacts=source.get('phase_contacts'),
                       controller=asdict(controller), kp_hand=2., inertia_ff_scale=1.,
                       cases={}, ready_for_training=False)
            report['trials'][name] = row
            np.savez_compressed(folder/('reference.npz' if numerical['accepted'] else 'diagnostic_rejected_reference.npz'),
                                **command)
            save()
            try:
                if not numerical['accepted']:
                    row['status'] = 'rejected_command_bounds'
                    continue
                # Explicitly verify the unchanged initial/task/world inputs.
                for key in ['initial_q', 'initial_velocity', 'timestamps', 'object_positions',
                            'object_wxyz', 'baseline_command_q']:
                    np.testing.assert_array_equal(command[key], reference[key])
                np.testing.assert_array_equal(command['q'][:, :7], reference['q'][:, :7])
                np.testing.assert_array_equal(command['q'][525:], reference['q'][525:])
                result = scene.rollout(command['q'], command['object_positions'], command['object_wxyz'],
                    command['timestamps'], force_ff=command['force_ff'], force_points=points,
                    contact_controller=controller, contact_normals_local=command['inward_normals_local'],
                    contact_active=command['contact_active'], point_local_trajectory=command['robot_local_points'],
                    object_local_trajectory=command['object_local_points'],
                    command_velocity=command['command_velocity'], kp_arm=300., kp_hand=2.,
                    inertia_ff=True, inertia_ff_scale=1., smooth_fixture=True,
                    initial_q=command['initial_q'], initial_velocity=command['initial_velocity'],
                    contact_body_audit=False)
                if len(result['q']) != len(reference['q']) or result['max_object_applied_force'] != 0:
                    raise AssertionError('Candidate must finish all frames without object assistance')
                np.testing.assert_array_equal(result['q'][0], execution['q'][0])
                np.testing.assert_array_equal(result['object_positions'][0], execution['object_positions'][0])
                np.savez_compressed(folder/'mechanics.npz', **result)
                summary = assessed(result, reference['timestamps'])
                summary['intervals'] = {phase: dict(
                    rms_position_mm=float(1000*np.sqrt(np.mean(result['position_error'][a:b+1]**2))),
                    rms_rotation_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error'][a:b+1]**2)))))
                    for phase, a, b in [('first',235,480),('free',525,644),('second',645,725),('final',770,857)]}
                row.update(status='complete', objective=objective(result),
                           cases={'mechanics': summary}, object_assistance=False, resets_during_episode=0)
                print(json.dumps(dict(trial=name, objective=row['objective'],
                    rms_mm=summary['rms_position_error_mm'], rms_deg=summary['rms_rotation_error_deg'],
                    torque=summary['max_requested_torque_fraction'], accepted=summary['accepted_physics'])), flush=True)
            except Exception as error:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                row['wall_seconds'] = perf_counter()-trial_start
                row['cost'] = process_cost(); row['cost']['cpu_seconds'] -= trial_cpu
                row['provenance_parent'] = str((args.output/'report.json').resolve())
                (folder/'report.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
                save()
        report['status'] = 'complete' if len(report['trials']) == len(trials) else 'budget_exhausted'
        eligible = [(name, row) for name, row in report['trials'].items() if row['status'] == 'complete']
        report['best_ranked'] = min(eligible, key=lambda item: item[1]['objective']['total'])[0] if eligible else None
        report['physical_success'] = any(row['cases']['mechanics']['accepted_physics'] for _, row in eligible)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=720.)
    run(parser.parse_args())
