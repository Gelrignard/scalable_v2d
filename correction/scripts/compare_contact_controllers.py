"""Controlled controller ablation; not a replacement for full-episode G1 tests.

Reuse the saved corrected motion, common initial state, simulator and actuator
limits. Compare no squeeze, a position preload, force feedforward, and normal
force feedback. No optimizer or policy training is rerun.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from correction.src.dynamics.horizon_replay import replay_pads
from correction.src.dynamics.symbolic_mechanics import CartesianPads
from correction.src.preprocessing.horizon_reference import HorizonReference, ContactPatch


def run(directory, output):
    started = perf_counter()
    problem = json.loads((directory/'problem.json').read_text())
    problem['contacts'] = [ContactPatch(**p) for p in problem['contacts']]
    ref, robot = HorizonReference(**problem), CartesianPads()
    ref.validate(robot)
    with np.load(directory/'plan.npz') as data:
        plan = {key: data[key].copy() for key in data.files}
    with np.load(directory/'truth.npz') as data:
        truth = {key: data[key].copy() for key in data.files}
    output.mkdir(parents=True, exist_ok=True)
    conditions = dict(position_without_preload=dict(force_guidance=False),
                      position_with_preload=dict(force_guidance=False, normal_preload_n=6.),
                      force_feedforward=dict(force_guidance=True),
                      normal_force_feedback=dict(force_guidance=True, normal_feedback_gain=.25))
    cases, initial = {}, None
    for name, kwargs in conditions.items():
        record = replay_pads(ref, robot, plan, **kwargs)
        state = np.r_[record['q'][0], record['v'][0], record['object_positions'][0], record['object_wxyz'][0]]
        if initial is None: initial = state
        np.testing.assert_array_equal(initial, state)
        target = np.column_stack([np.interp(record['times'], truth['times'], truth['positions'][:, j]) for j in range(3)])
        error = np.linalg.norm(record['object_positions']-target, axis=1)
        rotation_error = Rotation.from_quat(record['object_wxyz'], scalar_first=True).magnitude()
        score = dict(max_position_error_mm=float(error.max()*1000), rms_position_error_mm=float(1000*np.sqrt(np.mean(error**2))),
                     max_rotation_error_deg=float(np.rad2deg(rotation_error.max())),
                     max_penetration_mm=float(1000*record['max_penetration_m']),
                     max_requested_actuator_fraction=float(record['max_requested_actuator_fraction']),
                     max_object_applied_force=float(record['max_object_applied_force']), controller=kwargs)
        score['accepted_physics'] = bool(error.max() < .01 and rotation_error.max() < np.deg2rad(5)
            and record['max_penetration_m'] < .001 and record['max_requested_actuator_fraction'] <= 1
            and record['max_object_applied_force'] == 0)
        np.savez_compressed(output/f'{name}.npz', **record)
        cases[name] = score
        print(name, json.dumps(score), flush=True)
    report = dict(scope='Two-second manufactured carry; same corrected reference, not full-episode G1 evidence',
                  plan_source=str(directory.resolve()), position_preload_offset_mm=5.,
                  initial_state_identical=True, seconds=perf_counter()-started, cases=cases)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('correction/outputs/coupled_controlled'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/contact_controller_comparison'))
    args = parser.parse_args()
    run(args.directory, args.output)
