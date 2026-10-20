"""Decompose saved contact separation into tracking, geometry and prediction."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Preserve existing diagnostics; use a fresh directory')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    reference = read_npz(args.source/'reference.npz')
    execution = read_npz(args.source/'mechanics.npz')
    planning = read_npz(args.planning/'planning_poses.npz')
    _, _, _, scene = load_strategy_scene(saved['input_directory'])
    bodies = [p['body'] for p in saved['contacts']]
    names = [scene.model.body(body).name for body in bodies]
    args.output.mkdir(parents=True)
    rows = []
    for frame in range(args.first, args.last+1):
        local = reference['robot_local_points'][frame]
        actual_r = Rotation.from_quat(execution['object_wxyz'][frame], scalar_first=True)
        planned_r = Rotation.from_quat(planning['object_wxyz'][frame], scalar_first=True)
        actual_anchor = execution['object_positions'][frame]+actual_r.apply(reference['object_local_points'][frame])
        planned_anchor = planning['object_positions'][frame]+planned_r.apply(reference['object_local_points'][frame])
        world = []
        for q in [execution['q'][frame], reference['q'][frame]]:
            scene.set_state(q, execution['object_positions'][frame], execution['object_wxyz'][frame])
            world.append(np.array([scene.data.xpos[body]+scene.data.xmat[body].reshape(3, 3)@x
                                   for body, x in zip(bodies, local)]))
        tracking, geometry, prediction = world[0]-world[1], world[1]-planned_anchor, planned_anchor-actual_anchor
        separation = world[0]-actual_anchor
        np.testing.assert_allclose(tracking+geometry+prediction, separation, atol=1e-12, rtol=0)
        outward = -actual_r.apply(reference['inward_normals_local'][frame])
        frame_channels = []
        dt = execution['interval_timestamps'][frame+1]-execution['interval_timestamps'][frame]
        for channel, body in enumerate(bodies):
            if not reference['contact_active'][frame, channel]:
                continue
            frame_channels.append(dict(
                name=names[channel], channel=channel,
                separation_mm=(1000*separation[channel]).tolist(),
                outward_separation_mm=float(1000*np.dot(separation[channel], outward[channel])),
                tracking_mm=(1000*tracking[channel]).tolist(),
                nominal_geometry_mm=(1000*geometry[channel]).tolist(),
                object_prediction_mm=(1000*prediction[channel]).tolist(),
                touching=bool(execution['controller_touching'][frame, channel]),
                gate_gap_mm=float(1000*execution['controller_contact_gate_gap_m'][frame, channel]),
                planned_force_world_n=reference['force_ff'][frame, channel].tolist(),
                applied_guidance_world_n=execution['controller_commanded_contact_force'][frame, channel].tolist(),
                sampled_selected_force_world_n=execution['controller_measured_contact_force'][frame, channel].tolist(),
                interval_entire_body_force_world_n=(
                    execution['interval_robot_body_wrench_impulse_world'][frame, body, :3]/dt).tolist()))
        rows.append(dict(frame=frame,
            position_error_mm=float(1000*execution['position_error'][frame]),
            rotation_error_deg=float(np.rad2deg(execution['rotation_error'][frame])),
            environment_mean_force_n=(execution['interval_environment_contact_impulse_world'][frame, :3]/dt).tolist(),
            robot_mean_force_n=(execution['interval_robot_contact_impulse_world'][frame, :3]/dt).tolist(),
            channels=frame_channels))
    summary = {}
    for name in names:
        channels = [channel for row in rows for channel in row['channels'] if channel['name'] == name]
        if not channels:
            continue
        summary[name] = dict(samples=len(channels), touching_samples=sum(c['touching'] for c in channels),
            last_touching_frame=max((r['frame'] for r in rows for c in r['channels'] if c['name']==name and c['touching']), default=None),
            mean_magnitudes_mm={key: float(np.mean([np.linalg.norm(c[key]) for c in channels]))
                for key in ['separation_mm', 'tracking_mm', 'nominal_geometry_mm', 'object_prediction_mm']},
            max_outward_separation_mm=max(c['outward_separation_mm'] for c in channels))
    report = dict(status='complete', source=str(args.source.resolve()),
        frames=[args.first, args.last], summary=summary, rows=rows,
        definitions='p_actual - anchor_actual = (p_actual - p_command) + (p_command - anchor_planning) + (anchor_planning - anchor_actual)',
        note='Recomputed geometry only; no reconstructed forces. Controller fields are video-rate snapshots. Dense interval body force includes all patches on that body.',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            args.source/'report.json', args.source/'reference.npz', args.source/'mechanics.npz',
            args.planning/'planning_poses.npz']},
        provenance=snapshot_sources(args.output, [Path(__file__)]),
        wall_seconds=perf_counter()-start, cost=process_cost())
    report['cost']['cpu_seconds'] -= cpu
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(summary=summary, frames=[row for row in rows if row['frame'] in [
        650, 660, 670, 680, 690, 700, 710, 720]], wall_seconds=report['wall_seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--planning', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, default=645)
    parser.add_argument('--last', type=int, default=720)
    run(parser.parse_args())
