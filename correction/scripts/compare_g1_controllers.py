"""Full saved episode, fixed candidate trajectory, controller-only ablations.

The saved candidate is a historical failed reference, not an accepted coupled
optimization result. Nothing here refits observations or recomputes forces from
observed accelerations. All physical failures remain in the exported episode.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from correction.src.dynamics.supported_robot import SupportedRobotScene
from correction.src.dynamics.symbolic_mechanics import BodyPoint
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.preprocessing.convex_object import decompose
from correction.src.optimization.robot_manipulation import metrics, tracking_accepted


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def assessed(record, times):
    result = metrics(record)
    acceleration = np.diff(record['joint_velocities'], axis=0)/np.diff(times)[:, None]
    eef_velocity = np.diff(record['fk_tip_positions'], axis=0)/np.diff(times)[:, None, None]
    eef_acceleration = np.diff(eef_velocity, axis=0)/(.5*(times[2:]-times[:-2]))[:, None, None]
    result.update(rms_joint_acceleration_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
                  rms_eef_acceleration_m_s2=float(np.sqrt(np.mean(eef_acceleration**2))),
                  max_joint_step_rad=float(np.max(abs(np.diff(record['q'][:, 7:], axis=0)))),
                  max_eef_step_m=float(np.max(np.linalg.norm(np.diff(record['fk_tip_positions'], axis=0), axis=-1))),
                  simulated_frames=len(record['q']), full_episode=True)
    for key, value in record.items():
        if key.startswith('controller_') and np.asarray(value).ndim == 0:
            result[key] = value.item() if isinstance(value, np.ndarray) else value
    result['accepted_physics'] = bool(tracking_accepted(record) and record['max_torque_fraction_requested'] <= 1.)
    return result


def choose_controller(cases):
    """Retain the old controller unless a candidate passes all declared gates."""
    baseline = cases['legacy_feedforward']
    eligible = []
    for name in ('gated_feedforward', 'position_equivalent', 'force_feedback', 'position_feedback'):
        trial = cases[name]
        reasons = []
        if not trial['accepted_physics']: reasons.append('physical acceptance failed')
        if trial['rms_position_error_mm'] > baseline['rms_position_error_mm']:
            reasons.append('object tracking regressed')
        if trial['rms_joint_acceleration_rad_s2'] > baseline['rms_joint_acceleration_rad_s2']*1.01:
            reasons.append('joint acceleration regressed by more than 1%')
        if trial.get('controller_max_conversion_residual_nm', 0.) > .01:
            reasons.append('servo limits prevented nominal torque realization')
        if trial.get('max_eef_step_m', 0.) > baseline.get('max_eef_step_m', np.inf)*1.01:
            reasons.append('end-effector continuity regressed by more than 1%')
        if trial.get('rms_eef_acceleration_m_s2', 0.) > baseline.get('rms_eef_acceleration_m_s2', np.inf)*1.01:
            reasons.append('end-effector acceleration regressed by more than 1%')
        if not trial.get('controller_rate_contract_satisfied', True):
            reasons.append('servo offset-rate and joint limits incompatible with input trajectory')
        trial['promotion_rejection_reasons'] = reasons
        if not reasons: eligible.append(name)
    selected = min(eligible, key=lambda name: cases[name]['rms_position_error_mm']) if eligible else 'legacy_feedforward'
    return dict(selected=selected, promoted=bool(eligible),
                fallback_is_feasible=bool(baseline['accepted_physics']),
                rule='Physical gate, no worse object RMS, joint/EEF acceleration and EEF step <=101% of fallback, conversion loss <=0.01 Nm, servo rate contract',
                note='Fallback preserves prior behavior; it is not automatically a feasible manipulation.')


def input_trajectory_audit(reference, kin):
    """Audit the robot spline, never differentiate perceived object motion."""
    movable = np.array([i for i, name in enumerate(kin.joint_names)
                        if any(part in name for part in ('shoulder', 'elbow', 'wrist', 'hand', 'waist'))])
    times = reference['timestamps']-reference['timestamps'][0]
    dense = np.minimum(np.arange(round(times[-1]/.001)+1)*.001, times[-1])
    spline = CubicSpline(times, reference['q'][:, 7+movable])
    q, v, a = (spline(dense, derivative) for derivative in range(3))
    lower, upper = kin.robot.model.lowerPositionLimit[7+movable], kin.robot.model.upperPositionLimit[7+movable]
    speed = kin.robot.model.velocityLimit[6+movable]
    violation = np.maximum(np.maximum(lower-q, q-upper), 0.)
    index = np.unravel_index(np.argmax(violation), violation.shape)
    return dict(max_joint_limit_violation_rad=float(violation.max()),
                worst_limit_joint=kin.joint_names[movable[index[1]]], worst_limit_time_s=float(dense[index[0]]),
                max_speed_fraction=float(np.max(abs(v)/speed)),
                max_joint_acceleration_rad_s2=float(np.max(abs(a))),
                rms_joint_acceleration_rad_s2=float(np.sqrt(np.mean(a*a))),
                accepted_kinematics=bool(violation.max() < 1e-5 and np.max(abs(v)/speed) <= 1.),
                scope='Same commanded robot cubic spline as replay; this does not audit object perception acceleration')


def reassess(output):
    """Recompute acceptance from saved physics without rerunning or changing it."""
    path = output/'report.json'
    report = json.loads(path.read_text())
    prior = output/'report_before_continuity_audit.json'
    if not prior.exists(): prior.write_bytes(path.read_bytes())
    reference = load(output/'reference.npz')
    report['input_trajectory_audit'] = input_trajectory_audit(reference, G1Kinematics())
    for name, old in report['cases'].items():
        record = load(output/f'{name}.npz')
        old.update(assessed(record, reference['timestamps']))
        config = old['controller_config']
        if config is not None:
            old['controller_rate_contract_satisfied'] = old['controller_max_offset_rate_rad_s'] <= config['offset_rate_rad_s']+1e-8
            old['sampled_frames_with_matched_contact'] = int(np.any(record['controller_touching'], axis=1).sum())
            old['sampled_frames_with_force_enabled'] = int(np.any(record['controller_force_enabled'], axis=1).sum())
    report['decision'] = choose_controller(report['cases'])
    report['assessment_note'] = 'Continuity and rate-contract assessment recomputed from saved, unchanged physical rollouts.'
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def run(directory, output, *, reuse_fixed_baselines=None):
    started = perf_counter()
    if directory.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a new output directory; inputs and previous experiments must stay intact')
    source = json.loads((directory/'report.json').read_text())
    raw, reference = load(directory/'retargeted/contact_aware.npz'), load(directory/'reference.npz')
    q, times = reference['q'], reference['timestamps']
    if len(times) != len(raw['timestamps']) or not np.array_equal(reference['frame_ids'], np.arange(len(times))):
        raise ValueError('This comparison requires every original frame')
    np.testing.assert_allclose(times-times[0], raw['timestamps']-raw['timestamps'][0], atol=1e-8)
    np.testing.assert_array_equal(q[:, :7], raw['q'][:, :7])
    mesh = trimesh.load(str(raw['object_mesh']), force='mesh')
    meta = json.loads(str(raw['metadata_json']))
    scene = SupportedRobotScene(G1Kinematics(), mesh.bounds, meta['source']['support_surface'], raw['q'][0],
        mass=source['mass_kg'], friction=source['friction'], move_waist=True, moving_fixture=True,
        floor_z=source['floor_z_m'], object_parts=decompose(str(raw['object_mesh']), directory/'collision'))
    points = [BodyPoint(int(scene.model.site_bodyid[i]), scene.model.site_pos[i].copy()) for i in scene.tip_ids]
    common = dict(kp_arm=300., kp_hand=20., inertia_ff=True, initial_q=reference['initial_q'],
                  initial_velocity=reference['initial_velocity'])
    configs = dict(original=None, legacy_feedforward=None, gated_feedforward=ContactControllerConfig(),
                   position_equivalent=ContactControllerConfig(interface='position'),
                   force_feedback=ContactControllerConfig(feedback_gain=.25),
                   position_feedback=ContactControllerConfig(interface='position', feedback_gain=.25))
    output.mkdir(parents=True)
    cases, initial = {}, None
    reused = None
    if reuse_fixed_baselines is not None:
        reused = json.loads((reuse_fixed_baselines/'report.json').read_text())
        for filename, digest in reused['input_hashes'].items():
            if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != digest:
                raise ValueError('Cannot reuse baselines from different inputs')
    for name, config in configs.items():
        print(f'Complete G1 episode {source["episode"]["episode_index"]}: {name}, {len(times)} frames', flush=True)
        options = dict(common)
        if name != 'original': options['force_ff'] = reference['force_ff']
        if config is not None:
            options.update(force_points=points, contact_controller=config,
                           contact_normals_local=-raw['contact_normals_local'], contact_active=reference['contact_active'])
        begin = perf_counter()
        if reused is not None and name in ('original', 'legacy_feedforward'):
            record = load(reuse_fixed_baselines/f'{name}.npz')
        else:
            record = scene.rollout(raw['q'] if name == 'original' else q, reference['object_positions'],
                                   reference['object_wxyz'], times, **options)
        state = np.r_[record['q'][0], record['joint_velocities'][0], record['object_positions'][0], record['object_wxyz'][0]]
        if initial is None: initial = state
        np.testing.assert_array_equal(initial, state)
        if len(record['q']) != len(times): raise AssertionError('Incomplete physical episode')
        np.savez_compressed(output/f'{name}.npz', **record)
        cases[name] = dict(assessed(record, times), wall_seconds=perf_counter()-begin,
                           controller_config=asdict(config) if config else None)
        if reused is not None and name in ('original', 'legacy_feedforward'):
            cases[name]['reused_unchanged_baseline'] = str((reuse_fixed_baselines/f'{name}.npz').resolve())
        # This numerical reproduction check protects the rollback branch.
        if name == 'legacy_feedforward':
            old = load(directory/'corrected.npz')
            cases[name]['max_change_from_saved_fallback_m'] = float(np.max(abs(old['object_positions']-record['object_positions'])))
            np.testing.assert_allclose(old['object_positions'], record['object_positions'], atol=1e-10)
        print(json.dumps(cases[name]), flush=True)
    report = dict(scope='Controller-only comparison on a historical failed full-episode reference, not new coupled-method success',
        episode=source['episode'], frames=len(times), duration_s=float(times[-1]-times[0]),
        input_directory=str(directory.resolve()), initial_state_identical=True,
        input_hashes={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                      for name in ('reference.npz', 'retargeted/contact_aware.npz', 'report.json')},
        physics=dict(mass_kg=source['mass_kg'], friction=source['friction'], timestep_s=scene.model.opt.timestep,
                     supported_recorded_pelvis=True, object_assistance=False, resets_during_episode=0),
        input_trajectory_audit=input_trajectory_audit(reference, scene.kin),
        cases=cases, decision=choose_controller(cases), ready_for_training=False, seconds=perf_counter()-started)
    np.savez_compressed(output/'reference.npz', **reference)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['decision']), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('correction/outputs/random_three_full_20261019/episode_000018'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/g1_controller_full_20261019'))
    parser.add_argument('--assess-existing', action='store_true', help='Recompute metrics from existing rollouts without simulating')
    parser.add_argument('--reuse-fixed-baselines', type=Path, help='Reuse only original and legacy rollouts with exactly matching input hashes')
    args = parser.parse_args()
    if args.assess_existing:
        print(json.dumps(reassess(args.output)['decision']))
    else:
        run(args.directory, args.output, reuse_fixed_baselines=args.reuse_fixed_baselines)
