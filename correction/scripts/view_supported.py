"""View saved supported-EEF physics, with commanded and desired poses separated."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh
import viser


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def serve(directory, port=8083):
    report = json.loads((directory/'report.json').read_text())
    ref = read(directory/'reference.npz')
    records = {name: read(directory/f'{name}.npz') for name in ('baseline', 'geometry_only', 'wrench_seed', 'corrected')}
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.configure_theme(control_layout='collapsible', control_width='small', show_logo=False,
                               show_share_button=False, brand_color=(36, 150, 115))
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor', width=4., height=4., plane='xy')
    bounds = ref['bounds']
    book = trimesh.creation.box(extents=np.ptp(bounds, axis=0))
    book.apply_translation(bounds.mean(axis=0))
    # Draw the exact collision box, with a spine and cover markings, so users
    # can see its orientation without confusing visual mesh and collision mesh.
    pad = trimesh.creation.icosphere(subdivisions=2, radius=report['physical_parameters']['pad_radius_m'])
    colors = [(239, 139, 55), (62, 124, 235), (142, 82, 205)]*2
    nodes = []
    for side in range(2):
        prefix = f'/trial{side}'
        group = server.scene.add_frame(prefix, show_axes=False)
        item = dict(group=group)
        item['book'] = server.scene.add_mesh_simple(prefix+'/book', book.vertices, book.faces,
                                                   color=(93, 174, 209), flat_shading=True)
        item['reference'] = server.scene.add_mesh_simple(prefix+'/desired', book.vertices, book.faces,
                                                        wireframe=True, color=(150, 160, 164), opacity=.5)
        item['book_frame'] = server.scene.add_frame(prefix+'/book_axes', axes_length=.055, axes_radius=.0015)
        item['label'] = server.scene.add_label(prefix+'/label', '')
        item['pads'] = [server.scene.add_mesh_simple(prefix+f'/pad{i}', pad.vertices, pad.faces, color=colors[i]) for i in range(6)]
        item['targets'] = server.scene.add_point_cloud(prefix+'/targets', np.zeros((6, 3), np.float32),
                                                      colors=(255, 100, 80), point_size=.004)
        item['forces'] = server.scene.add_line_segments(prefix+'/contact_forces', np.zeros((6, 2, 3), np.float32),
                                                       colors=(36, 165, 100), line_width=2.)
        table = trimesh.creation.cylinder(radius=float(ref['table_radius']), height=float(ref['table_height']))
        server.scene.add_mesh_simple(prefix+'/table', table.vertices, table.faces,
                                     position=ref['table_center'], color=(165, 151, 130), opacity=.65)
        trace = np.stack((ref['object_positions'][:-1], ref['object_positions'][1:]), axis=1).astype(np.float32)
        server.scene.add_line_segments(prefix+'/reference_path', trace, colors=(155, 165, 175), line_width=1.)
        nodes.append(item)

    valid = report['accepted_demonstration_local_contact_reference']
    server.gui.add_markdown('## Supported EEF manipulation\n**'+('PASS' if valid else 'FAIL')+
                            ' · free-book physics**\n\nSix force-limited spherical finger pads. No walking or arm model.')
    comparison = server.gui.add_dropdown('View', options=['Corrected only', 'Before / corrected', 'Geometry only / corrected', 'Wrench seed / corrected'])
    frame = server.gui.add_slider('Source frame', min=int(ref['frame_ids'][0]), max=int(ref['frame_ids'][-1]),
                                  step=1, initial_value=int(np.clip(500, ref['frame_ids'][0], ref['frame_ids'][-1])))
    play = server.gui.add_checkbox('Play', initial_value=False)
    speed = server.gui.add_dropdown('Speed', options=['1×', '0.5×', '0.25×'], initial_value='1×')
    show_reference = server.gui.add_checkbox('Desired pose outline', initial_value=True)
    show_forces = server.gui.add_checkbox('Measured contact forces', initial_value=True)
    show_targets = server.gui.add_checkbox('Command markers', initial_value=False)
    focus = server.gui.add_button('Focus on manipulation')
    status = server.gui.add_markdown('')
    with server.gui.add_folder('Measured result and scope', expand_by_default=False):
        m = report['corrected']
        server.gui.add_markdown(
            f"Worst error over all simulation steps: **{m['max_position_error_mm']:.2f} mm / {m['max_rotation_error_deg']:.2f}°**.\n\n"
            f"Worst penetration: **{m['max_penetration_mm']:.3f} mm**. Peak pad actuator force: **{m['max_motor_force_n']:.1f} N**.\n\n"
            f"Correction and comparison: **{report['method_seconds']:.1f} s CPU**, {report['rollout_evaluations']} rollouts, no RL.\n\n"
            f"Common vertical reference correction: **{1000*report['common_object_registration_xyz_m'][2]:.1f} mm**, to clear the recorded table. "
            'All trials start from the same established grasp; the first 0.25 s shares a command handover.\n\n'
            'Object pose is set once at reset. Gravity and measured contact forces produce every later object pose. '
            'The outline is the desired pose; the solid book is simulated.\n\n'
            f"Maximum contact-command change from the human fingertips: **{report['max_contact_command_shift_from_human_mm']:.1f} mm**.\n\n"
            'This validates a reduced contact model. G1 finger geometry, arm IK, grasp acquisition, and walking are not validated.')

    offsets = [np.zeros(3), np.zeros(3)]
    def update(_=None):
        j = int(frame.value)-int(ref['frame_ids'][0])
        split = comparison.value != 'Corrected only'
        baseline = {'Before / corrected': 'baseline', 'Geometry only / corrected': 'geometry_only',
                    'Wrench seed / corrected': 'wrench_seed'}.get(comparison.value, 'corrected')
        offsets[:] = [np.array([0., .38 if split else 0., 0.]), np.array([0., -.38 if split else 0., 0.])]
        with server.atomic():
            for i, (item, label) in enumerate(zip(nodes, (baseline, 'corrected'))):
                item['group'].visible = split or i == 1
                item['group'].position = offsets[i]
                data = records[label]
                item['book'].position = data['object_positions'][j]
                item['book'].wxyz = data['object_wxyz'][j]
                item['book_frame'].position = data['object_positions'][j]
                item['book_frame'].wxyz = data['object_wxyz'][j]
                item['reference'].position = ref['object_positions'][j]
                item['reference'].wxyz = ref['object_wxyz'][j]
                item['reference'].visible = show_reference.value
                for pad_node, position in zip(item['pads'], data['pad_positions'][j]):
                    pad_node.position = position
                item['targets'].points = data['targets'][j].astype(np.float32)
                item['targets'].visible = show_targets.value
                item['forces'].points = np.stack((data['pad_positions'][j], data['pad_positions'][j]+.003*data['contact_forces'][j]), axis=1).astype(np.float32)
                item['forces'].visible = show_forces.value
                item['label'].text = label.replace('_', ' ').upper()+f" · {1000*data['position_error'][j]:.1f} mm / {np.rad2deg(data['rotation_error'][j]):.1f}°"
                item['label'].position = data['object_positions'][j]+np.array([0., 0., .2])
            m = records['corrected']
            status.content = (f"**Frame {frame.value}** · {ref['times'][j]-ref['times'][0]:.2f} / {report['duration_s']:.2f} s\n\n"
                              f"Corrected error: **{1000*m['position_error'][j]:.2f} mm / {np.rad2deg(m['rotation_error'][j]):.2f}°**")

    def camera(client):
        center = np.array([-.58, 0., .78])
        client.camera.position = center+np.array([-1.15, -.12, .55])
        client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)
        client.camera.fov = .9

    @server.on_client_connect
    def connect(client):
        camera(client)

    @focus.on_click
    def focus_camera(event):
        if event.client:
            camera(event.client)

    for handle in (comparison, frame, show_reference, show_forces, show_targets):
        handle.on_update(update)
    update()
    print(f'Supported EEF comparison: http://127.0.0.1:{port}/', flush=True)
    while True:
        rate = {'1×': 1., '0.5×': .5, '0.25×': .25}[speed.value]
        time.sleep(1/30/rate)
        if play.value:
            frame.value = int(ref['frame_ids'][0]) if frame.value >= int(ref['frame_ids'][-1]) else frame.value+1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('correction/outputs/supported_book_local_20261019'))
    parser.add_argument('--port', type=int, default=8083)
    args = parser.parse_args()
    serve(args.directory, args.port)
