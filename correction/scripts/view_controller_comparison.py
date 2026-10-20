"""One full-episode viewer: original retargeting versus the current method.

Intended object motion is always shown alongside actual simulated motion.
Historical controller ablations remain saved on disk, outside this viewer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import trimesh
import viser
from correction.scripts.view_book import NamedVisualizer
from correction.scripts.contact_overlay import ContactOverlay, FORCE_SCALE
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def serve(directory, port, support_directory):
    report = json.loads((directory/'report.json').read_text())
    ref = read(directory/'reference.npz')
    current='mechanics' if (directory/'mechanics.npz').exists() else 'position_feedback'
    selections = ('original', current)
    labels = ('ORIGINAL RETARGETING', 'OUR METHOD · current candidate')
    records = {name: read(directory/f'{name}.npz') for name in selections}
    support_report = json.loads((support_directory/'report.json').read_text())
    support_ref = read(support_directory/'reference.npz')
    report.setdefault('frames',len(ref['timestamps']))
    report.setdefault('duration_s',float(ref['timestamps'][-1]-ref['timestamps'][0]))
    report.setdefault('episode',support_report['episode'])
    if support_report['source_reference_sha256'] != hashlib.sha256((directory/'reference.npz').read_bytes()).hexdigest():
        raise ValueError('Independent pads must use the same episode and saved reference as the robot channel')
    support_records = {name: read(support_directory/f'{file}.npz')
                       for name, file in zip(selections, ('original', 'corrected'))}
    support_cases = dict(zip(selections, [support_report['cases'][k] for k in ('original', 'corrected')]))
    if any(len(data['q']) != len(ref['timestamps']) for data in records.values()):
        raise ValueError('Comparison requires both complete physical episodes')
    for data in support_records.values():
        np.testing.assert_array_equal(data['timestamps'], ref['timestamps'])
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.configure_theme(control_layout='collapsible', control_width='small', show_logo=False,
                               show_share_button=False, brand_color=(34, 135, 158))
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor', width=8, height=8, plane='xy', position=(0., 0., float(ref['floor_z'])))
    server.scene.add_frame('/world_origin', axes_length=.2, axes_radius=.003)
    kin = G1Kinematics()
    mesh = trimesh.load(str(ref['object_mesh']), force='mesh')
    hull = mesh.convex_hull
    target_path = np.stack((ref['object_positions'][:-1], ref['object_positions'][1:]), axis=1).astype(np.float32)
    nodes = []
    overlays, robot_groups, support_objects = [], [], []
    for i in range(2):
        prefix = f'/trial{i}'
        group = server.scene.add_frame(prefix, show_axes=False)
        robot_groups.append(server.scene.add_frame(prefix+'/robot', show_axes=False))
        robot = NamedVisualizer(server, model=kin.robot.model, visual_model=kin.robot.visual_model,
                                collision_model=kin.robot.collision_model, name=prefix+'/robot')
        robot.displayFrames(False)
        actual = server.scene.add_mesh_trimesh(prefix+'/object', mesh)
        support_objects.append(server.scene.add_mesh_simple(prefix+'/pad_object', mesh.vertices, mesh.faces,
            color=(68, 124, 173), opacity=.3, visible=False, cast_shadow=False, receive_shadow=False))
        target = server.scene.add_mesh_simple(prefix+'/intended_object', hull.vertices, hull.faces,
                                              color=(255, 143, 25), opacity=.22, cast_shadow=False, receive_shadow=False)
        target_edges = server.scene.add_mesh_simple(prefix+'/intended_outline', hull.vertices, hull.faces,
                                                    color=(255, 143, 25), wireframe=True, opacity=.9,
                                                    cast_shadow=False, receive_shadow=False)
        server.scene.add_line_segments(prefix+'/intended_path', target_path, colors=(255, 143, 25), line_width=2.)
        error_line = server.scene.add_line_segments(prefix+'/object_error', np.zeros((1, 2, 3), np.float32),
                                                     colors=(220, 65, 65), line_width=2.)
        label = server.scene.add_label(prefix+'/label', '')
        nodes.append((group, robot, actual, target, target_edges, error_line, label))
        overlays.append(ContactOverlay(server.scene, prefix+'/contacts',
            support_ref['original_tip_targets'] if i == 0 else support_ref.get('corrected_pad_targets',ref.get('tip_targets')), ref['contact_active'],
            planned_forces=None if i == 0 else ref['force_ff'],
            planned_origins=support_ref.get('intended_object_contact_points'),
            radius=support_report['physical_parameters']['pad_radius_m']))
    server.gui.add_markdown('**Original vs our method - physical acceptance failed**')
    with server.gui.add_folder('Playback and comparison', expand_by_default=False):
        server.gui.add_markdown('## Original retargeting vs our method')
        channel = server.gui.add_dropdown('Channel', options=['Robot simulation', 'Independent support-point simulation'],
                                          initial_value='Robot simulation')
        legend = server.gui.add_markdown(
            '**Solid robot and object:** physical simulation.\n\n'
            '**Orange ghost + path:** intended object motion, visible in both views. '
            '**Red line:** actual-to-intended object error.')
        frame = server.gui.add_slider('Frame', min=0, max=len(ref['timestamps'])-1, step=1, initial_value=0)
        play = server.gui.add_checkbox('Play', initial_value=False)
        speed = server.gui.add_dropdown('Speed', options=['0.25', '0.5', '1'], initial_value='0.5')
        arrow_scale = server.gui.add_number('Force scale (mm/N)', initial_value=1000*FORCE_SCALE,
                                            min=.1, max=30., step=1., visible=False)
        focus = server.gui.add_button('Focus robot')
        hands = server.gui.add_button('Focus hands')
        object_focus = server.gui.add_button('Focus intended + actual objects')
        status = server.gui.add_markdown('')
        force_status = server.gui.add_markdown('', visible=False)
        with server.gui.add_folder('What is being compared', expand_by_default=False):
            server.gui.add_markdown(f'**{report["episode"]["task"]}**\n\n'
                f'{report["frames"]} source frames, {report["duration_s"]:.2f} s, original timing. '
                'No hidden resets or forces applied directly to the object. The pelvis follows an external fixture; legs do not balance.\n\n'
                '**Left:** original retargeted motion executed in physics. '
                '**Right:** corrected upper-limb reference with the saved controller and bounded contact-force guidance. '
                'The current candidate has not passed physical acceptance.\n\n'
                'Both orange objects follow the same intended source trajectory. Their translucent convex hulls are lightweight visual guides, '
                'not simulated bodies. Solid objects follow their actual physical rollouts.\n\n'
                'Comparison copies have constant world-Y offsets of −0.8 m and +0.8 m. '
                'The world axes do not move. Camera controls only move the camera. '
                'This viewer shows saved rollouts and does not run a simulator or optimizer. '
                'Playback pauses automatically when no viewer is connected.')
            server.gui.add_markdown('**Independent support-point simulation:** two additional full physical rollouts of this same episode, '
                f'with {len(support_ref["initial_pads"])} finite-mass spherical pads and no robot. Blue spheres show actual pad positions and6mm radii. '
                'Cyan marks original/corrected pad-center targets. For surface-based runs, centers are offset behind the selected '
                'robot surface normals so a sphere touches at that surface instead of starting embedded. '
                'Cyan trails show intended contact phases. The simulated object is translucent blue in this channel '
                'so contacts on the far side remain visible; orange still marks its intended pose.\n\n'
                'Both start before the approach with identical states and zero velocity. PD controls pad motion; only the corrected '
                'trial adds the recorded force plan with bounded feedback, gated to matched physical contact. '
                'Object mass, COM/inertia (when explicit), collision pieces, friction and floor match the robot experiment. '
                'Pad contact solver settings and Cartesian actuators differ from the robot. Pad force limit is25N each. '
                'There is no artificial initial grasp, object assistance or reset during the episode.\n\n'
                'Purple arrows are saved planned pad-on-object forces at intended points. They are not measured forces or '
                'the gated commands sent to the pads. The original has no explicit force plan. '
                'Green arrows show measured per-pad resultant forces at force-weighted contact centroids. These aggregates do not '
                'preserve every contact point or the full contact moment. No green marker is drawn for a pad with no contact. '
                f'Both force colors start at {1000*FORCE_SCALE:g} mm/N, adjustable together without magnitude normalization or clipping.')
    lock = threading.RLock()
    playing = threading.Event()

    def camera(client, kind='robot'):
        if channel.value != 'Robot simulation' and kind == 'robot':
            kind = 'objects'
        j = int(frame.value)
        points = []
        for i, selected in enumerate(selections):
            support_view = channel.value != 'Robot simulation'
            data = (support_records if support_view else records)[selected]
            offset = np.array([0., -.8 if i == 0 else .8, 0.])
            if kind == 'objects':
                points.extend([data['object_positions'][j]+offset, ref['object_positions'][j]+offset])
            else:
                point = (data['pad_positions' if support_view else 'fk_tip_positions'][j].mean(0)
                         if kind == 'hands' else data['q'][j, :3].copy())
                points.append(point+offset)
        points = np.asarray(points)
        center = np.mean(points, axis=0)
        half_width = np.ptp(points[:, 1])/2+.5
        distance = max(1.6, half_width/(np.tan(.85/2)*max(.3, client.camera.aspect)))
        client.camera.position = center+[distance, -.3, .4*distance]
        client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)
        client.camera.fov = .85

    def update(_=None):
        with lock, server.atomic():
            j = int(frame.value)
            support_view = channel.value != 'Robot simulation'
            force_status.visible = support_view
            arrow_scale.visible = support_view
            focus.visible = not support_view
            if support_view:
                legend.content = ('**Full episode · independent pad physics**\n\n'
                    '**Orange:** intended object. **Translucent blue:** simulated object.\n\n'
                    '**Cyan points + trails:** intended pad motion. **Blue spheres:** simulated pads.\n\n'
                    '**Purple arrows:** planned forces (our method). **Green arrows:** measured contact forces (both). '
                    f'**Arrow scale: 10 N = {arrow_scale.value:g} cm.**')
            else:
                legend.content = ('**Solid robot and object:** physical simulation.\n\n'
                    '**Orange ghost + path:** intended object motion, visible in both views. '
                    '**Red line:** actual-to-intended object error.')
            lines = [f'**{j+1}/{len(ref["timestamps"])}** frames · {ref["timestamps"][j]:.2f}/{report["duration_s"]:.2f} s']
            for i, (node, selected) in enumerate(zip(nodes, selections)):
                group, robot, actual, target, target_edges, error_line, label = node
                data = (support_records if support_view else records)[selected]
                summary = (support_cases if support_view else report['cases'])[selected]
                group.position = (0., -.8 if i == 0 else .8, 0.)
                robot_groups[i].visible = not support_view
                if not support_view:
                    robot.display(data['q'][j])
                overlays[i].update(j, data, support_view, arrow_scale.value/1000)
                actual.visible = not support_view
                support_objects[i].visible = support_view
                if support_view:
                    support_objects[i].position, support_objects[i].wxyz = data['object_positions'][j], data['object_wxyz'][j]
                actual.position, actual.wxyz = data['object_positions'][j], data['object_wxyz'][j]
                for intended in (target, target_edges):
                    intended.position, intended.wxyz = ref['object_positions'][j], ref['object_wxyz'][j]
                error_line.points = np.array([[data['object_positions'][j], ref['object_positions'][j]]], np.float32)
                label.position = (ref['object_positions'][j]+[0., 0., .6] if support_view
                                  else data['q'][j, :3]+[0., 0., .85])
                label.text = labels[i]
                lines.append(f'**{"Original" if i == 0 else "Our method"}**: object error '
                             f'**{1000*data["position_error"][j]:.1f} mm** · '
                             f'whole-episode RMS {summary["rms_position_error_mm"]:.1f} mm.')
            if support_view:
                accepted = support_cases[current]['accepted_tracking']
                lines.append('Independent-pad tracking check: **'+('passed' if accepted else 'failed')+
                    '**. This channel tests contact execution without robot IK, arm collisions or joint limits.')
            else:
                lines.append('Current method: **physical acceptance failed**. The intended motion stays visible so the discrepancy can be inspected.')
            status.content = '\n\n'.join(lines)
            if support_view:
                measured = [np.linalg.norm(support_records[name]['contact_forces'][j], axis=-1)
                            for name in selections]
                planned = np.linalg.norm(ref['force_ff'][j], axis=-1)
                commanded = np.linalg.norm(support_records[current]['commanded_forces'][j], axis=-1)
                force_status.content = ('**Force magnitudes (N)**\n\n'
                    '| Pad | Original measured | Our measured | Our plan | Gated command |\n'
                    '|---|---:|---:|---:|---:|\n'+
                    '\n'.join(f'| {name} | {measured[0][i]:.2f} | {measured[1][i]:.2f} | {planned[i]:.2f} | {commanded[i]:.2f} |'
                        for i, name in enumerate(support_ref.get('channel_names',
                            ['L thumb', 'L index', 'L middle', 'R thumb', 'R index', 'R middle'])))+
                    '\n\nMeasured contact force differs from total pad actuator force, which also accelerates the pad and compensates gravity.')

    @server.on_client_connect
    def connect(client): camera(client)
    @focus.on_click
    def focus_robot(_):
        for client in server.get_clients().values(): camera(client)
    @hands.on_click
    def focus_hands(_):
        for client in server.get_clients().values(): camera(client, 'hands')
    @object_focus.on_click
    def focus_objects(_):
        for client in server.get_clients().values(): camera(client, 'objects')
    @play.on_update
    def playback_changed(_):
        if play.value: playing.set()
        else: playing.clear()
    @server.on_client_disconnect
    def disconnect(_):
        if not server.get_clients(): play.value = False
    frame.on_update(update)
    arrow_scale.on_update(update)
    @channel.on_update
    def channel_changed(_):
        play.value = False
        update()
        for client in server.get_clients().values(): camera(client)
    update()
    while True:
        playing.wait()  # No frame-update polling while paused.
        time.sleep(float(np.median(np.diff(ref['timestamps'])))/float(speed.value))
        if play.value:
            if not server.get_clients() or frame.value == frame.max:
                play.value = False  # no hidden looping/reset of a failed episode
            else:
                frame.value += 1
                # The slider setter invokes its callback; avoid a duplicate
                # robot/mesh update and duplicate websocket traffic per frame.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('correction/outputs/g1_controller_full_20261019_v2'))
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--support-directory', type=Path, default=Path('correction/outputs/support_points_full_20261019'))
    args = parser.parse_args()
    serve(args.directory, args.port, args.support_directory)
