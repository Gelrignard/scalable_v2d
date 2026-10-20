"""Full-episode, single-change feedback ablation around a preserved nominal.

Use --gain 0 for exact physical regression, then a declared nonzero gain.
No command re-timing, force reallocation, contact-family search or object actuation.
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
from correction.src.optimization.surface_reference import smoothstep
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.dynamics.surface_follow import SurfaceFollowConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def inherited_planning_poses(reference, sequence):
    """Reconstruct the saved two-phase planner's poses, never its simulated outcome.

    build_sequence originally shifts its second planning path while retaining
    the unshifted object target for evaluation. Following that target instead
    would incorrectly apply the saved offline correction a second time.
    """
    poses = {key: reference[key].copy() for key in ('object_positions', 'object_wxyz')}
    shift = sequence['command']['second_planning_translation_m']
    if shift is not None:
        timing = sequence['phase_contacts'][1]['timing']
        frames = np.arange(len(reference['q']))
        weight = 1-smoothstep((frames-timing['force_full'])/(timing['contact_end']-timing['force_full']))
        weight[frames < timing['approach_start']] = 0.
        if np.any(reference['contact_active'][timing['approach_start']-1:timing['approach_start']+1]):
            raise ValueError('Planning chart switch must occur during an inactive interval')
        poses['object_positions'] += weight[:, None]*np.asarray(shift)
    return poses


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    start = perf_counter(); initial_cpu = process_cost()['cpu_seconds']
    old = json.loads((args.source/'report.json').read_text())
    reference = read_npz(args.source/'reference.npz')
    before = read_npz(args.source/'mechanics.npz')
    # Require actual ancestry, not another experiment with similar-looking poses.
    ancestor = args.source.resolve(); chain = []
    for _ in range(12):
        path = ancestor/'report.json'; chain.append(path)
        saved = json.loads(path.read_text())
        if ancestor == args.planning_sequence.resolve():
            break
        ancestor = Path(saved['source']).resolve()
    else:
        raise ValueError('Planning sequence is not in the source ancestry')
    sequence = saved
    sequence_reference = read_npz(args.planning_sequence/'reference.npz')
    for key in ('timestamps', 'object_positions', 'object_wxyz', 'object_local_points', 'contact_active'):
        np.testing.assert_array_equal(reference[key], sequence_reference[key])
    planning = inherited_planning_poses(reference, sequence)
    config = SurfaceFollowConfig(frames=None, gain=args.gain, surface_displacement_limit_m=.01,
        joint_offset_limit_rad=.04, joint_speed_limit_rad_s=.2, joint_acceleration_limit_rad_s2=1.)
    _, _, _, scene = load_strategy_scene(old['input_directory'])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in row.items()})
              for row in old['contacts']]
    args.output.mkdir(parents=True)
    files = [args.source/'reference.npz', args.source/'mechanics.npz', *chain,
             args.planning_sequence/'reference.npz']
    report = {key: old[key] for key in ('input_directory', 'controller', 'physics', 'contacts',
                                       'phase_contacts', 'kp_hand', 'inertia_ff_scale')}
    report['initialization_protocol'] = old.get('initialization_protocol',
        'Exact saved initial_q/initial_velocity and source object initial state; verified against saved physical replay')
    report.update(status='running', source=str(args.source.resolve()), cases={},
        surface_feedback=asdict(config), object_assistance=False, resets_during_episode=0,
        ready_for_training=False, planning_sequence=str(args.planning_sequence.resolve()),
        scope='Staged nominal plus bounded arm-only observed-surface residual; inherited force guidance and clock',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limitations=['Inherited contact schedule and candidate bodies; no automatic task/grasp discovery.',
                     'Local pose following does not jointly optimize task wrench or certify contact.',
                     'Nominal inertial feedforward retained; residual acceleration is not added to it.',
                     'Residual bounds are separate from total command/physical acceptance.'],
        provenance=snapshot_sources(args.output, [Path(__file__), *[Path('correction/src')/name for name in (
            'dynamics/surface_follow.py', 'dynamics/object_relative_controller.py', 'dynamics/contact_controller.py',
            'dynamics/supported_robot.py', 'dynamics/execution_step.py', 'dynamics/contact_measurement.py',
            'dynamics/torque_audit.py', 'optimization/manipulation_sequence.py')]]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    write()
    np.savez_compressed(args.output/'reference.npz', **reference)
    np.savez_compressed(args.output/'planning_poses.npz', **planning)
    try:
        result = scene.rollout(reference['q'], reference['object_positions'], reference['object_wxyz'],
            reference['timestamps'], force_ff=reference['force_ff'], force_points=points,
            contact_controller=ContactControllerConfig(**old['controller']),
            contact_normals_local=reference['inward_normals_local'], contact_active=reference['contact_active'],
            point_local_trajectory=reference['robot_local_points'], object_local_trajectory=reference['object_local_points'],
            command_velocity=reference['command_velocity'], kp_arm=300., kp_hand=old['kp_hand'],
            inertia_ff=True, inertia_ff_scale=old['inertia_ff_scale'], smooth_fixture=True,
            initial_q=reference['initial_q'], initial_velocity=reference['initial_velocity'],
            contact_body_audit=True, surface_follow=config, surface_follow_plan=planning)
        np.savez_compressed(args.output/'mechanics.npz', **result)
        if len(result['q']) != len(reference['q']) or result['max_object_applied_force'] != 0:
            raise AssertionError('Complete unassisted physical episode required')
        first = int(np.flatnonzero(np.any(reference['contact_active'], axis=1))[0])
        keys = ('q', 'joint_velocities', 'object_positions', 'object_wxyz', 'torques', 'frame_peak_torque_fraction')
        stop = len(reference['q']) if args.gain == 0 else first
        identical = {key: bool(np.array_equal(result[key][:stop], before[key][:stop])) for key in keys}
        if not all(identical.values()):
            raise AssertionError('Zero-residual execution must preserve nominal physics')
        summary = assessed(result, reference['timestamps'])
        report.update(status='complete', cases=dict(mechanics=summary),
            frames=len(reference['q']), duration_s=float(reference['timestamps'][-1]-reference['timestamps'][0]),
            unchanged_physical_frames=stop, exact_nominal_checks=identical,
            observed_residual=dict(updates=int(result['controller_surface_follow_updates']),
                max_offset_rad=float(result['controller_surface_follow_max_offset_rad']),
                max_speed_rad_s=float(result['controller_surface_follow_max_speed_rad_s']),
                max_acceleration_rad_s2=float(result['controller_surface_follow_max_acceleration_rad_s2']),
                bound_conflicts=int(result['controller_surface_follow_bound_conflicts'])),
            followed_joint_names=[scene.kin.joint_names[scene.movable[i]] for i in result['surface_follow_joint_columns']],
            accepted=bool(summary['accepted_physics'] and result['controller_surface_follow_bounds_satisfied']))
        print(json.dumps({key: report[key] for key in ('cases', 'exact_nominal_checks', 'observed_residual')}, indent=2), flush=True)
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
    parser.add_argument('--planning-sequence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gain', type=float, default=.25)
    run(parser.parse_args())
