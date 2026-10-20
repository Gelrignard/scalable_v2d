"""One full-episode surface-feedback ablation on an immutable command."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.dynamics.surface_follow import SurfaceFollowConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    old = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    reference = read_npz(args.source/'reference.npz')
    before = read_npz(args.source/'mechanics.npz')
    planning = read_npz(args.planning/'planning_poses.npz')
    _, _, _, scene = load_strategy_scene(old['input_directory'])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in old['contacts']]
    config = SurfaceFollowConfig()
    args.output.mkdir(parents=True)
    report = {k: old[k] for k in ['input_directory', 'controller', 'physics', 'contacts',
        'phase_contacts', 'kp_hand', 'inertia_ff_scale', 'initialization_protocol']}
    report.update(status='running', source=str(args.source.resolve()), cases={}, ready_for_training=False,
        surface_feedback=asdict(config), object_assistance=False, resets_during_episode=0,
        scope='Controller-only ablation; same commands/force plan/caps, additional perfect-pose bench feedback',
        joint_selection='Only non-finger joints with continuous Bezier-hull margin >= offset bound+0.1mrad over the entire remaining command',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            args.source/'reference.npz', args.source/'mechanics.npz', args.source/'report.json',
            args.planning/'planning_poses.npz']},
        provenance=snapshot_sources(args.output, [Path(__file__), *[
            Path('correction/src')/p for p in ['dynamics/surface_follow.py', 'dynamics/object_relative_controller.py',
                'dynamics/contact_controller.py', 'dynamics/supported_robot.py',
                'dynamics/contact_measurement.py', 'dynamics/torque_audit.py']]]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    write()
    np.savez_compressed(args.output/'reference.npz', **reference)
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
        prefix = {key: bool(np.array_equal(result[key][:config.frames[0]+1], before[key][:config.frames[0]+1]))
                  for key in ['q', 'joint_velocities', 'object_positions', 'object_wxyz']}
        if not all(prefix.values()):
            raise AssertionError('Feedback must preserve the entire physical prefix through activation')
        if len(result['q']) != len(reference['q']) or result['max_object_applied_force'] != 0:
            raise AssertionError('Full unassisted physical episode required')
        summary = assessed(result, reference['timestamps'])
        summary['intervals'] = {name: dict(
            rms_position_mm=float(1000*np.sqrt(np.mean(result['position_error'][a:b+1]**2))),
            rms_rotation_deg=float(np.rad2deg(np.sqrt(np.mean(result['rotation_error'][a:b+1]**2)))),
            max_requested_torque_fraction=float(result['frame_peak_torque_fraction'][a:b+1].max()))
            for name, a, b in [('first',235,480), ('free',525,644), ('second',645,725), ('final',770,857)]}
        report.update(status='complete', physical_prefix_checks=prefix, cases=dict(mechanics=summary),
            followed_joint_names=[scene.kin.joint_names[scene.movable[i]] for i in result['surface_follow_joint_columns']],
            nominal_joint_margin_rad=result['surface_follow_nominal_joint_margin_rad'].tolist(),
            accepted=bool(summary['accepted_physics'] and result['controller_surface_follow_bounds_satisfied']))
        print(json.dumps(dict(summary=summary, physical_prefix_checks=prefix), indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-start
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--planning', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
