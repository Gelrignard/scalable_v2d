"""Run the coupled solver and independent physics on a known-ground-truth case.

This manufactured two-pad experiment is a mechanics/uncertainty diagnostic,
not a challenge episode or evidence of full G1 success.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from correction.src.preprocessing.controlled_horizon import controlled_carry
from correction.src.optimization.coupled_horizon import HorizonSettings, solve_horizon, sample_solution
from correction.src.dynamics.horizon_audit import audit_horizon
from correction.src.dynamics.horizon_replay import replay_pads
from correction.src.pass_traj.horizon_result import save_horizon


def run(directory, *, seed=7, noise=.004, outlier=.025, knots=21, verify=False):
    started = perf_counter()
    ref, robot, truth = controlled_carry(seed=seed, noise=noise, outlier=outlier, knots=knots)
    result = solve_horizon(ref, robot, HorizonSettings())
    print(json.dumps(dict(stages=result['stages'], seconds=result['seconds'], constraint=result['max_constraint_violation'])), flush=True)
    audit = audit_horizon(ref, robot, result)
    report = dict(scope='Controlled two-pad phase with common initial grasp; not G1 or challenge data', audit=audit,
                  simulator=dict(cone='elliptic', impratio=20, noslip_iterations=10, tolerance=1e-10,
                                 nominal_timestep_s=.001, nominal_friction=.6),
                  robot_coordinate_units='meters (Cartesian pad centers)',
                  solver_contact_model='rigid sticking with a conservative friction pyramid',
                  earlier_pyramidal_replay=dict(max_position_error_mm=24.1741, accepted_physics=False,
                    explanation='Soft contact drift; retained as a model-mismatch finding, not discarded as a successful result'))
    records = {}
    if result['success']:
        records['baseline'] = replay_pads(ref, robot, {'q': ref.joint_reference}, force_guidance=False)
        records['trajectory_only'] = replay_pads(ref, robot, result, force_guidance=False)
        records['corrected'] = replay_pads(ref, robot, result, force_guidance=True)
        if verify:
            for name, kwargs in [('half_timestep', dict(dt=.0005)), ('mass_plus_10pct', dict(mass_scale=1.1)),
                                 ('lower_friction', dict(friction=.5))]:
                records[name] = replay_pads(ref, robot, result, **kwargs)
        for name, record in records.items():
            t = record['times']
            target = np.column_stack([np.interp(t, truth['times'], truth['positions'][:, j]) for j in range(3)])
            position_error = np.linalg.norm(record['object_positions']-target, axis=1)
            rotation_error = Rotation.from_quat(record['object_wxyz'], scalar_first=True).magnitude()
            report[name] = dict(rms_position_error_mm=1000*float(np.sqrt(np.mean(position_error**2))),
                                max_position_error_mm=1000*float(position_error.max()),
                                max_rotation_error_deg=float(np.rad2deg(rotation_error.max())),
                                max_penetration_mm=1000*record['max_penetration_m'],
                                max_requested_actuator_fraction=record['max_requested_actuator_fraction'],
                                max_object_applied_force=record['max_object_applied_force'])
            report[name]['accepted_physics'] = bool(position_error.max() < .01 and rotation_error.max() < np.deg2rad(5)
                and record['max_penetration_m'] < .001 and record['max_object_applied_force'] == 0
                and record['max_requested_actuator_fraction'] <= 1.)
        sample = sample_solution(ref, result, truth['times'])
        report['observation_rms_mm'] = float(1000*np.sqrt(np.mean(np.sum((ref.observed_positions-truth['positions'])**2, axis=1))))
        report['estimated_rms_mm'] = float(1000*np.sqrt(np.mean(np.sum((sample['position']-truth['positions'])**2, axis=1))))
        original = CubicSpline(ref.times, ref.joint_reference)
        report['baseline_command_rms_acceleration'] = float(np.sqrt(np.mean(original(truth['times'], 2)**2)))
        report['accepted_controlled_experiment'] = bool(audit['accepted_nominal'] and report['corrected']['accepted_physics']
            and report['corrected']['rms_position_error_mm'] < report['baseline']['rms_position_error_mm'])
        records['truth'] = truth
    else:
        report['accepted_controlled_experiment'] = False
    report['total_experiment_seconds'] = perf_counter()-started
    path = save_horizon(directory, ref, result, report, records)
    print(json.dumps(report, indent=2), flush=True)
    print(path.resolve(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/coupled_controlled'))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--knots', type=int, default=21)
    parser.add_argument('--noise', type=float, default=.004)
    parser.add_argument('--outlier', type=float, default=.025)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run(args.output, seed=args.seed, noise=args.noise, outlier=args.outlier, knots=args.knots, verify=args.verify)


if __name__ == '__main__': main()
