"""Distinguish brief hand impacts from floor-supported drift in a saved replay."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    output = args.source/'release_impulse_audit.json'
    if output.exists():
        raise ValueError('Preserve the existing audit; use a new source run')
    prior = json.loads((args.source/'report.json').read_text())
    data = read_npz(args.source/'mechanics.npz')
    _, _, _, scene = load_strategy_scene(prior['input_directory'])
    t = data['interval_timestamps']
    robot = data['interval_robot_contact_impulse_world']
    support = data['interval_environment_contact_impulse_world']
    body_impulse = data['interval_robot_body_unsigned_impulse_ns']
    velocity = data['generalized_velocities'][:, scene.book_v:scene.book_v+3]
    mass = scene.model.body_mass[scene.book_body]
    # This check is valid for this source-derived model's zero COM offset.
    if np.linalg.norm(scene.model.body_ipos[scene.book_body]) > 1e-12:
        raise ValueError('Linear momentum check currently requires zero local COM')
    momentum_change = mass*np.diff(velocity, axis=0)
    gravity = mass*np.diff(t)[:, None]*scene.model.opt.gravity
    residual = momentum_change - gravity - robot[:, :3] - support[:, :3]
    rows = []
    for a, b in zip(args.frames[:-1], args.frames[1:]):
        cumulative = body_impulse[a:b].sum(axis=0)
        rows.append(dict(first_frame=a, end_frame=b,
            position_error_mm=(1000*data['position_error'][[a,b]]).tolist(),
            rotation_error_deg=np.rad2deg(data['rotation_error'][[a,b]]).tolist(),
            displacement_mm=(1000*(data['object_positions'][b]-data['object_positions'][a])).tolist(),
            velocity_m_s=velocity[[a,b]].tolist(),
            robot_impulse_world=robot[a:b].sum(axis=0).tolist(),
            environment_impulse_world=support[a:b].sum(axis=0).tolist(),
            peak_robot_force_n=float(data['interval_peak_robot_contact_force_n'][a:b].max()),
            robot_body_unsigned_impulse_ns={scene.model.body(i).name:float(v)
                for i,v in enumerate(cumulative) if v>1e-6}))
    result = dict(source=str(args.source.resolve()),
        source_sha256=hashlib.sha256((args.source/'mechanics.npz').read_bytes()).hexdigest(),
        note='Every 1ms solved contact integrated between rounded video timestamps. Moments about instantaneous COM.',
        max_linear_momentum_balance_residual_ns=float(np.linalg.norm(residual,axis=1).max()),
        cumulative_linear_momentum_balance_residual_ns=residual.sum(axis=0).tolist(),
        intervals=rows, wall_seconds=perf_counter()-start)
    if args.compare:
        old = read_npz(args.compare/'mechanics.npz')
        for key in ('q','object_positions','object_wxyz','torques'):
            np.testing.assert_array_equal(old[key],data[key])
        result['instrumentation_unchanged_physics_vs'] = str(args.compare.resolve())
    result['provenance'] = snapshot_sources(args.source/'release_impulse_provenance', [Path(__file__)])
    result['wall_seconds'] = perf_counter()-start
    result['cost'] = process_cost(); result['cost']['cpu_seconds'] -= cpu
    output.write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--frames',type=int,nargs='+',
                        default=[680,690,700,710,720,725,735,745,755,765,775,800,857])
    run(parser.parse_args())
