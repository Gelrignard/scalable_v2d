"""Inspect saved coupled-optimizer physics and explicitly rejected G1 poses.

This viewer never generates object motion: it reads saved MuJoCo samples for
the controlled experiment. G1 mode is labeled kinematic optimizer diagnostics.
All data retain their original world coordinates, with no display offsets.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
import trimesh
import viser


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_g1(directory):
    from correction.src.preprocessing import retarget_bridge
    from v2d_retarget.core.robot import G1Kinematics
    from correction.src.preprocessing.horizon_reference import ContactPatch, HorizonReference
    from correction.src.dynamics.supported_robot import SupportedRobotScene
    from correction.src.dynamics.symbolic_mechanics import BodyPoint, SymbolicRobot
    from correction.src.optimization.coupled_horizon import sample_solution
    problem = json.loads((directory/'problem.json').read_text())
    problem['contacts'] = [ContactPatch(**value) for value in problem['contacts']]
    ref = HorizonReference(**problem)
    raw = read(Path(ref.metadata['source']))
    mesh = trimesh.load(str(raw['object_mesh']), force='mesh')
    kin = G1Kinematics()
    support = json.loads(str(raw['metadata_json']))['source']['support_surface']
    scene = SupportedRobotScene(kin, mesh.bounds, support, raw['q'][0],
                               mass=ref.mass, friction=.6, move_waist=True, moving_fixture=True)
    points = [BodyPoint(scene.model.body(p['body']).id, np.asarray(p['local']))
              for p in ref.metadata['contact_surface_points']]
    robot = SymbolicRobot(scene, points)
    ref.validate(robot)
    plan = read(directory/'plan.npz')
    times = np.linspace(ref.times[0], ref.times[-1], 121)
    sample = sample_solution(ref, plan, times)
    root_position = CubicSpline(ref.times, ref.base[:, :3])(times)
    root_rotation = RotationSpline(ref.times, Rotation.from_quat(ref.base[:, 3:7], scalar_first=True))(times)
    q = np.tile(raw['q'][0], (len(times), 1))
    q[:, :3], q[:, 3:7], q[:, 7+scene.movable] = root_position, root_rotation.as_quat(), sample['q']
    com = scene.model.body_ipos[scene.book_body]
    rotations = Rotation.from_quat(sample['wxyz'], scalar_first=True)
    origin = sample['position']-rotations.apply(np.tile(com, (len(times), 1)))
    tips, targets, penetration, pairs = [], [], [], []
    for i, t in enumerate(times):
        base = np.zeros(25); base[:3] = root_position[i]; base[3:7] = root_rotation[i].as_quat(scalar_first=True)
        tips.append(np.asarray(robot.kinematics(sample['q'][i], base)).T)
        targets.append(sample['position'][i]+rotations[i].apply(plan['anchors'][:robot.count]))
        scene.set_state(q[i], origin[i], sample['wxyz'][i])
        worst = min(scene.data.contact, key=lambda c: c.dist, default=None)
        penetration.append(max(0., -float(worst.dist)) if worst is not None else 0.)
        pairs.append(' / '.join(scene.model.geom(g).name for g in (worst.geom1, worst.geom2)) if worst is not None else '')
    return dict(ref=ref, scene=scene, kin=kin, q=q, sample=sample, origin=origin,
                tips=np.asarray(tips), targets=np.asarray(targets), penetration=np.asarray(penetration),
                pairs=pairs, times=times, report=json.loads((directory/'report.json').read_text()))


def serve(controlled, g1_directory, port):
    report = json.loads((controlled/'report.json').read_text())
    problem = json.loads((controlled/'problem.json').read_text())
    truth = read(controlled/'truth.npz')
    observations = read(controlled/'observations.npz')
    records = {key: read(controlled/f'{key}.npz') for key in ('baseline', 'trajectory_only', 'corrected')}
    g1 = load_g1(g1_directory)
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.configure_theme(control_layout='collapsible', control_width='medium', show_logo=False,
                               show_share_button=False, brand_color=(32, 137, 151))
    server.scene.set_up_direction('+z')
    server.scene.add_frame('/world', axes_length=.15, axes_radius=.002)
    server.scene.add_grid('/coordinate_grid', width=3., height=3., plane='xy')
    pad_group = server.scene.add_frame('/controlled', show_axes=False)
    book_mesh = trimesh.creation.box(extents=2*np.asarray(problem['half_size']))
    book = server.scene.add_mesh_simple('/controlled/object', book_mesh.vertices, book_mesh.faces, color=(52, 163, 198))
    outline = server.scene.add_mesh_simple('/controlled/clean_reference', book_mesh.vertices, book_mesh.faces,
                                           color=(235, 155, 55), wireframe=True)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=.006)
    pads = [server.scene.add_mesh_simple(f'/controlled/pad{i}', sphere.vertices, sphere.faces,
                                         color=(204, 95, 130)) for i in range(2)]
    obs = server.scene.add_point_cloud('/controlled/observations', observations['positions'].astype(np.float32),
                                       colors=(155, 145, 180), point_size=.004)
    trace = np.stack((truth['positions'][:-1], truth['positions'][1:]), axis=1).astype(np.float32)
    server.scene.add_line_segments('/controlled/reference_path', trace, colors=(235, 155, 55), line_width=2.)
    forces = server.scene.add_line_segments('/controlled/measured_forces', np.zeros((2, 2, 3), np.float32),
                                            colors=(40, 170, 105), line_width=3.)
    g1_group = server.scene.add_frame('/g1', show_axes=False)
    from correction.scripts.view_book import NamedVisualizer
    kin, scene = g1['kin'], g1['scene']
    visual = NamedVisualizer(server, model=kin.robot.model, visual_model=kin.robot.visual_model,
                             collision_model=kin.robot.collision_model, name='/g1/robot')
    visual.displayFrames(False); visual.displayCollisions(False)
    collision_box = trimesh.creation.box(extents=np.ptp(scene.bounds, axis=0))
    collision_box.apply_translation(scene.bounds.mean(axis=0))
    g1_book = server.scene.add_mesh_simple('/g1/object_estimate', collision_box.vertices, collision_box.faces, color=(220, 139, 60))
    table = trimesh.creation.cylinder(radius=scene.table_radius, height=2*(scene.table_top-scene.table_center[2]))
    server.scene.add_mesh_simple('/g1/table', table.vertices, table.faces, position=scene.table_center,
                                 color=(160, 150, 135), opacity=.7)
    gaps = server.scene.add_line_segments('/g1/contact_gaps', np.zeros((6, 2, 3), np.float32), colors=(240, 70, 70), line_width=4.)
    actual_points = server.scene.add_point_cloud('/g1/robot_surface_points', g1['tips'][0].astype(np.float32),
                                                 colors=(220, 75, 150), point_size=.008)
    anchor_points = server.scene.add_point_cloud('/g1/object_anchors', g1['targets'][0].astype(np.float32),
                                                 colors=(40, 190, 110), point_size=.008)
    server.gui.add_markdown('## Coupled correction · inspection\nWorld coordinates are unchanged. The camera moves; the world frame does not.')
    labels = {'Controlled: corrected + force guidance': 'corrected',
              'Controlled: corrected trajectory only': 'trajectory_only',
              'Controlled: original commands': 'baseline', 'G1: rejected optimizer poses': None}
    mode = server.gui.add_dropdown('Result', options=list(labels), initial_value=list(labels)[0])
    frame = server.gui.add_slider('Playback frame', min=0, max=120, step=1, initial_value=0)
    play = server.gui.add_checkbox('Play', initial_value=True)
    speed = server.gui.add_dropdown('Speed', options=['0.25x', '0.5x', '1x'], initial_value='0.5x')
    show_observations = server.gui.add_checkbox('Noisy object observations', initial_value=False)
    show_forces = server.gui.add_checkbox('Measured force vectors', initial_value=True)
    show_collision = server.gui.add_checkbox('G1 collision meshes', initial_value=False)
    focus = server.gui.add_button('Focus on manipulation')
    whole = server.gui.add_button('Show whole G1')
    jump = server.gui.add_button('Inspect worst G1 collision')
    title = server.gui.add_markdown('')
    status = server.gui.add_markdown('')
    with server.gui.add_folder('Legend and scope', expand_by_default=False):
        server.gui.add_markdown('**Controlled:** solid blue object and pink pads are saved MuJoCo physics. '
            'Orange outline is the clean target. Green vectors are measured forces on the object (5 mm per newton). '
            'No walking, arm model, or catching floor; the grid is only a coordinate reference.\n\n'
            '**G1:** orange object and robot are failed optimizer poses, not simulated execution. '
            'Pink points are robot surface points; green points are object anchors; red segments show their gaps. '
            'Gaps before 2.1 s are approach distances, not active-contact errors. '
            'Walking is excluded. The prescribed pelvis path is retained.\n\n'
            'No display offsets, trajectory centering, hidden resets, or object assistance are added by this viewer.')

    def camera(client, wide=False):
        if labels[mode.value] is None:
            center = g1['sample']['position'][int(frame.value)] if not wide else g1['q'][int(frame.value), :3]+[0, 0, .12]
            offset = np.array([-.75, -.85, .5]) if not wide else np.array([-1.5, -1.6, .85])
        else:
            center = np.array([.02, 0., .46]); offset = np.array([.5, -.65, .36])
        client.camera.position = center+offset
        client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)
        client.camera.fov = .75

    @server.on_client_connect
    def connect(client): camera(client)

    def update(_=None):
        j = int(frame.value); key = labels[mode.value]
        is_g1 = key is None
        with server.atomic():
            pad_group.visible, g1_group.visible = not is_g1, is_g1
            if is_g1:
                visual.display(g1['q'][j]); visual.displayCollisions(show_collision.value)
                g1_book.position, g1_book.wxyz = g1['origin'][j], g1['sample']['wxyz'][j]
                gaps.points = np.stack((g1['tips'][j], g1['targets'][j]), axis=1).astype(np.float32)
                actual_points.points = g1['tips'][j].astype(np.float32)
                anchor_points.points = g1['targets'][j].astype(np.float32)
                distance = 1000*np.max(np.linalg.norm(g1['tips'][j]-g1['targets'][j], axis=1))
                active = g1['times'][j] >= 2.1-1e-6
                title.content = '**REJECTED · kinematic optimizer diagnostic**\n\nNo successful G1 physical rollout exists for this plan.'
                status.content = (f"Time **{g1['times'][j]:.2f} / 4.00 s** · source frame **{j}**\n\n"
                    f"{'Active contact gap' if active else 'Approach distance'}: **{distance:.1f} mm**\n\n"
                    f"Mesh penetration: **{1000*g1['penetration'][j]:.1f} mm**\n\n{g1['pairs'][j]}")
            else:
                record = records[key]
                index = round(j*(len(record['times'])-1)/120)
                t = record['times'][index]
                p = np.array([np.interp(t, truth['times'], truth['positions'][:, d]) for d in range(3)])
                book.position, book.wxyz = record['object_positions'][index], record['object_wxyz'][index]
                outline.position, outline.wxyz = p, (1., 0., 0., 0.)
                centers = record['q'][index].reshape(2, 3)
                for node, center in zip(pads, centers): node.position = center
                forces.points = np.stack((centers, centers+.005*record['contact_forces'][index]), axis=1).astype(np.float32)
                forces.visible, obs.visible = show_forces.value, show_observations.value
                ok = report[key]['accepted_physics']
                title.content = f"**{'PASS' if ok else 'FAIL'} · saved MuJoCo physics**\n\n2 s controlled two-pad carry; not a G1 or challenge episode."
                error = 1000*np.linalg.norm(record['object_positions'][index]-p)
                outside = abs(record['object_positions'][index, 2]-.46) > .65
                status.content = (f"Time **{t:.2f} / 2.00 s**\n\nCurrent object error: **{error:.2f} mm**\n\n"
                    f"Whole replay maximum: **{report[key]['max_position_error_mm']:.2f} mm**"+
                    ('\n\nObject has fallen below the view. Its position is not reset or clamped.' if outside else ''))

    @mode.on_update
    def mode_change(_):
        frame.value = 0
        update()
        for client in server.get_clients().values(): camera(client)

    @focus.on_click
    def focus_click(_):
        for client in server.get_clients().values(): camera(client)

    @whole.on_click
    def wide_click(_):
        for client in server.get_clients().values(): camera(client, True)

    @jump.on_click
    def jump_click(_):
        play.value = False
        mode.value = 'G1: rejected optimizer poses'
        frame.value = int(np.argmax(g1['penetration']))
        update()
        for client in server.get_clients().values(): camera(client)

    for control in (frame, show_observations, show_forces, show_collision): control.on_update(update)
    update()
    print(f'Coupled correction inspection: http://127.0.0.1:{port}/', flush=True)
    while True:
        duration = 4. if labels[mode.value] is None else 2.
        time.sleep(duration/120/float(speed.value[:-1]))
        if play.value: frame.value = (int(frame.value)+1) % 121


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controlled', type=Path, default=Path('correction/outputs/coupled_controlled'))
    parser.add_argument('--g1', type=Path, default=Path('correction/outputs/coupled_book_facing_surfaces'))
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    serve(args.controlled, args.g1, args.port)
