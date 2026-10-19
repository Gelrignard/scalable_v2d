"""Compare bounded correction candidates and actual diagnostic physics rollouts."""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import trimesh
import viser

from correction.scripts.view_book import NamedVisualizer
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.output import load_result


def serve(directory, port):
    report = json.loads((directory/'bounded_report.json').read_text())
    before, metadata = load_result(Path(report['input_reference']))
    after, _ = load_result(directory/'bounded_reference.npz')
    records = [before, after]
    rollouts = {}
    for label in report['rollouts']:
        with np.load(directory/f'rollout_{label}.npz', allow_pickle=False) as data:
            rollouts[label] = {k: data[k].copy() for k in data.files}
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid', width=6, height=6, plane='xy')
    mesh = trimesh.load(str(before['object_mesh']), force='mesh')
    offsets = [np.array([0., .6, 0.]), np.array([0., -.6, 0.])]
    robots, books, labels, clouds = [], [], [], []
    support = Path(metadata['source']['support_surface']).read_text()
    radius = float(re.search(r'double radius = ([\d.eE+-]+)', support).group(1))
    height = float(re.search(r'double height = ([\d.eE+-]+)', support).group(1))
    center = np.array(list(map(float, re.search(r'xformOp:translate = \(([^)]+)\)', support).group(1).split(','))))
    for i, offset in enumerate(offsets):
        kin = G1Kinematics()
        robot = NamedVisualizer(server, model=kin.robot.model, visual_model=kin.robot.visual_model,
                                 collision_model=kin.robot.collision_model, name=f'/side{i}/robot')
        robot.displayFrames(False)
        robot.displayCollisions(False)
        robots.append(robot)
        books.append(server.scene.add_mesh_trimesh(f'/side{i}/book', mesh))
        labels.append(server.scene.add_label(f'/side{i}/label', '', position=(0., float(offset[1]), 1.5)))
        clouds.append(server.scene.add_point_cloud(f'/side{i}/targets', points=np.zeros((6, 3), np.float32), colors=(30, 190, 110), point_size=.008))
        table = trimesh.creation.cylinder(radius=radius, height=height)
        table.visual.face_colors = [175, 157, 139, 180]
        server.scene.add_mesh_trimesh(f'/side{i}/table', table, position=center+offset)
    start, stop = report['requested_frames_inclusive']
    server.gui.add_markdown('## Book correction · diagnostic comparison\n\n**REJECTED for training: physical feasibility has not been established.**')
    options = ['Kinematic reference']
    if 'before' in rollouts and 'after' in rollouts:
        options += ['Physical rollout: corrected']
        options += [f'Physical rollout: {n}' for n in rollouts if n.startswith('preload')]
    mode = server.gui.add_dropdown('Playback', options=options, initial_value=options[0])
    frame = server.gui.add_slider('Frame', min=start, max=stop, step=1, initial_value=450 if start <= 450 <= stop else start)
    play = server.gui.add_checkbox('Play', initial_value=False)
    collision_geometry = server.gui.add_checkbox('Robot collision shapes', initial_value=False)
    focus = server.gui.add_button('Focus on book')
    status = server.gui.add_markdown('')
    b, a = report['before'], report['after']
    server.gui.add_markdown(f"CPU correction: **{report['correction_seconds']:.1f} s**.\n\n"
        f"Max hand/arm penetration: **{b['max_manipulation_penetration_mm']:.1f} → {a['max_manipulation_penetration_mm']:.1f} mm**.\n\n"
        f"Max joint acceleration: **{b['max_joint_acceleration_rad_s2']:.0f} → {a['max_joint_acceleration_rad_s2']:.0f} rad/s²**.\n\n"
        f"Fingertip wrench checks: **{a['wrench_feasible_frames']}/{a['frames']}**.\n\n"
        'Physical playback uses recorded free dynamics, with torque-limited PD and no object/base pose assistance. '
        'The local simulator and physical parameters are diagnostic assumptions, not the official challenge environment.')

    def update(_=None):
        t = int(frame.value)
        physical = mode.value != 'Kinematic reference'
        selected = 'after' if mode.value.endswith('corrected') else mode.value.split(': ')[-1]
        with server.atomic():
            for i, offset in enumerate(offsets):
                data = rollouts['before' if i == 0 else selected] if physical else records[i]
                j = min(t-start, len(data['q'])-1) if physical else t
                q = data['q'][j].copy()
                q[:3] += offset
                robots[i].display(q)
                robots[i].displayCollisions(collision_geometry.value)
                books[i].position = data['object_positions'][j]+offset
                books[i].wxyz = data['object_wxyz'][j]
                labels[i].text = ('BEFORE' if i == 0 else 'AFTER')+(' · physical rollout' if physical else ' · kinematic candidate')
                labels[i].position = data['object_positions'][j]+offset+np.array([0., 0., .25])
                clouds[i].visible = not physical
                if not physical:
                    clouds[i].points = (data['tip_targets'][t]+offset).astype(np.float32)
            status.content = f"Frame **{t}** · Time **{before['timestamps'][t]:.3f} s**\n\n"+(
                'Object and robot follow recorded dynamics. Falling or lost contact is a failure.' if physical else
                'Robot and object poses are assigned for reference inspection; this mode does not simulate contact.')

    def camera(client, close=False):
        center = before['object_positions'][int(frame.value)].copy() if close else np.array([-.4, 0., .7])
        client.camera.position = center+np.array([-1.4, -.05, .3] if close else [-2.7, -1.4, 1.3])
        client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)

    @server.on_client_connect
    def connect(client):
        camera(client)

    @focus.on_click
    def focus_camera(_):
        for client in server.get_clients().values():
            camera(client, True)

    for control in (mode, frame, collision_geometry):
        control.on_update(update)
    update()
    print(f'Bounded comparison: http://127.0.0.1:{port}', flush=True)
    while True:
        if play.value:
            frame.value = start if frame.value >= stop else frame.value+1
        time.sleep(1/30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('correction/outputs/bounded_book'))
    parser.add_argument('--port', type=int, default=8083)
    args = parser.parse_args()
    serve(args.directory, args.port)


if __name__ == '__main__':
    main()
