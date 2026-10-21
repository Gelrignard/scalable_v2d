"""Full-episode observed force-stage feedback, with no MPC or predicted rollout."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from correction.src.preprocessing.staged_planning import resolve_planning
from correction.scripts.compare_g1_controllers import assessed
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.dynamics.staged_observation import StagedObservationConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    start = perf_counter(); initial_cpu = process_cost()['cpu_seconds']
    old = json.loads((args.source/'report.json').read_text())
    reference = read_npz(args.source/'reference.npz'); before = read_npz(args.source/'mechanics.npz')
    planning, planning_files, planning_contract = resolve_planning(args.source, reference, args.planning_sequence)
    cfg = StagedObservationConfig(gain=args.gain)
    args.output.mkdir(parents=True)
    files = [args.source/'reference.npz', args.source/'mechanics.npz', *planning_files]
    report = {key: old[key] for key in ('input_directory', 'controller', 'physics', 'contacts',
                                       'phase_contacts', 'kp_hand', 'inertia_ff_scale')}
    report.update(status='running', source=str(args.source.resolve()), cases={},
        planning_sequence=str(args.planning_sequence.resolve()) if args.planning_sequence else None,
        planning_contract=planning_contract, staged_observation=asdict(cfg),
        episode=old.get('episode'), source_planning_accepted=old.get('planning_accepted'),
        source_planning_failure=old.get('diagnostic_fallback'),
        scope='Saved staged motion and force guidance plus instantaneous observed-state force QP',
        prediction_rollouts=0, object_assistance=False, resets_during_episode=0, ready_for_training=False,
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limitations=[*old.get('limitations', []),
            'Fixed saved motion, grasp candidates and phase clock; no missing-contact recovery.',
            'Privileged actual simulator pose/twist and contact sensors, not a learned observation pipeline.',
            'Static no-worse force/torque constraints permit inherited violations; physical caps still checked.',
            'Contact-force deltas are robot guidance, not guaranteed realized contact forces.'],
        provenance=snapshot_sources(args.output, [Path(__file__), Path('correction/src/preprocessing/staged_planning.py'),
            *[Path('correction/src')/name for name in ('dynamics/staged_observation.py',
            'optimization/observed_force_update.py', 'dynamics/contact_controller.py', 'dynamics/supported_robot.py',
            'dynamics/execution_step.py', 'dynamics/contact_measurement.py', 'dynamics/torque_audit.py')]]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    write()
    np.savez_compressed(args.output/'reference.npz', **reference)
    np.savez_compressed(args.output/'planning_poses.npz', **planning)
    try:
        _, _, _, scene = load_strategy_scene(old['input_directory'])
        points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in row.items()}) for row in old['contacts']]
        result = scene.rollout(reference['q'], reference['object_positions'], reference['object_wxyz'],
            reference['timestamps'], force_ff=reference['force_ff'], force_points=points,
            contact_controller=ContactControllerConfig(**old['controller']),
            contact_normals_local=reference['inward_normals_local'], contact_active=reference['contact_active'],
            point_local_trajectory=reference['robot_local_points'], object_local_trajectory=reference['object_local_points'],
            command_velocity=reference['command_velocity'], kp_arm=300., kp_hand=old['kp_hand'],
            inertia_ff=True, inertia_ff_scale=old['inertia_ff_scale'], smooth_fixture=True,
            initial_q=reference['initial_q'], initial_velocity=reference['initial_velocity'], contact_body_audit=True,
            staged_observation=cfg, staged_observation_plan=planning)
        np.savez_compressed(args.output/'mechanics.npz', **result)
        if len(result['q']) != len(reference['q']) or result['max_object_applied_force'] != 0:
            raise AssertionError('Complete unassisted physical episode required')
        active_frames = np.flatnonzero(np.any(reference['contact_active'], axis=1))
        first = int(active_frames[0]) if len(active_frames) else len(reference['q'])
        stop = len(reference['q']) if args.gain == 0 else first
        keys = ('q', 'joint_velocities', 'object_positions', 'object_wxyz', 'torques', 'frame_peak_torque_fraction')
        identical = {key: bool(np.array_equal(result[key][:stop], before[key][:stop])) for key in keys}
        if not all(identical.values()):
            raise AssertionError('Disabled feedback must preserve nominal physics exactly')
        summary = assessed(result, reference['timestamps'])
        audit = {key.removeprefix('staged_observation_'): value.item() if isinstance(value, np.ndarray) else value
                 for key, value in result.items() if key.startswith('staged_observation_')}
        bounds_ok = (audit['rate_conflicts'] == 0 and audit['failures'] == 0 and audit['max_static_violation'] <= 1e-6
                     and audit['max_delta_n'] <= cfg.contact_delta_limit_n+1e-6
                     and audit['max_rate_n_s'] <= cfg.contact_delta_rate_n_s+1e-3)
        report.update(status='complete', cases=dict(mechanics=summary), frames=len(reference['q']),
            duration_s=float(reference['timestamps'][-1]-reference['timestamps'][0]),
            unchanged_physical_frames=stop, exact_nominal_checks=identical, observation_audit=audit,
            accepted=bool(summary['accepted_physics'] and bounds_ok and old.get('planning_accepted') is not False))
        print(json.dumps(dict(cases=report['cases'], observation_audit=audit), indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-start
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= initial_cpu
        write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--planning-sequence', type=Path,
        help='Required for the saved two-grasp bin plan; omitted only for verified single-interval transfers')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gain', type=float, default=1.)
    run(parser.parse_args())
