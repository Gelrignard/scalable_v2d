"""Separate nominal joint/root inertial compensation from gravity demand.

This is a command audit, not a replacement for delivered torque/contact logs.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz


def run(directory):
    begin = perf_counter()
    report = json.loads((directory/'report.json').read_text())
    _, _, _, scene = load_strategy_scene(report['input_directory'])
    reference = read_npz(directory/'reference.npz')
    q, t = reference['q'], reference['timestamps']
    robot = CubicHermiteSpline(t, q[:, 7+scene.movable], reference['command_velocity'])
    root_p = CubicSpline(t, q[:, :3])
    root_r = RotationSpline(t, Rotation.from_quat(q[:, 3:7]))
    obj_p = CubicSpline(t, reference['object_positions'])
    obj_r = Slerp(t, Rotation.from_quat(reference['object_wxyz'], scalar_first=True))
    rows = []
    for time in np.arange(t[0], t[-1], .01):
        pose = q[0].copy()
        pose[:3], pose[3:7], pose[7+scene.movable] = root_p(time), root_r(time).as_quat(), robot(time)
        scene.set_state(pose, obj_p(time), obj_r(time).as_quat(scalar_first=True))
        parts = {}
        for kind in ('joint', 'root'):
            acceleration = np.zeros(scene.model.nv)
            if kind == 'joint':
                acceleration[scene.active_v] = robot(time, 2)
            else:
                acceleration[scene.root_v:scene.root_v+3] = root_p(time, 2)
                acceleration[scene.root_v+3:scene.root_v+6] = root_r(time, 2)
            inertial = np.zeros(scene.model.nv)
            mujoco.mj_mulM(scene.model, scene.data, inertial, acceleration)
            parts[kind] = inertial[scene.active_v]
        parts['gravity'] = scene.data.qfrc_bias[scene.active_v].copy()
        parts['sum'] = sum(parts.values())
        rows.append((time, parts))
    result = dict(scope='nominal dynamics at zero velocity; excludes actual-state error and contact forces',
                  sampling_seconds=.01, intervals={})
    for label, first, last in [('first', 235, 480), ('free', 525, 634), ('second', 635, 725)]:
        chosen = [(time, parts) for time, parts in rows if t[first] <= time <= t[last]]
        interval = {}
        for kind in ('joint', 'root', 'gravity', 'sum'):
            ratios = np.abs([parts[kind] for _, parts in chosen])/scene.caps
            frame, joint = np.unravel_index(ratios.argmax(), ratios.shape)
            interval[kind] = dict(peak_cap_fraction=float(ratios[frame, joint]),
                time_s=float(chosen[frame][0]), joint=scene.kin.joint_names[scene.movable[joint]])
        result['intervals'][label] = interval
    result['wall_seconds'] = perf_counter()-begin
    (directory/'command_dynamics_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
