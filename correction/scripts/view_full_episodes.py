"""Three complete, randomly selected episodes: actual physics before/after."""
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
import trimesh
import viser
from correction.scripts.view_book import NamedVisualizer
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics


def read(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def serve(directory,port):
    selection = json.loads((directory/'selection.json').read_text())
    episodes = {}
    for info in selection['selected']:
        path = directory/f'episode_{info["episode_index"]:06d}'
        label = f'{info["episode_index"]:02d} · {info["object"].replace("_"," ")}'
        episodes[label] = dict(info=info,ref=read(path/'reference.npz'),
            baseline=read(path/'baseline.npz'),corrected=read(path/'corrected.npz'),
            report=json.loads((path/'report.json').read_text()))
    server = viser.ViserServer(host='127.0.0.1',port=port)
    server.gui.configure_theme(control_layout='collapsible',control_width='small',show_logo=False,
                               show_share_button=False,brand_color=(34,135,158))
    server.scene.set_up_direction('+z')
    grid = server.scene.add_grid('/floor',width=8,height=8,plane='xy')
    server.scene.add_frame('/world_origin',axes_length=.25,axes_radius=.004)
    kin = G1Kinematics()
    nodes = []
    for i,name in enumerate(('baseline','corrected')):
        prefix = f'/trial{i}'
        group = server.scene.add_frame(prefix,show_axes=False)
        robot = NamedVisualizer(server,model=kin.robot.model,visual_model=kin.robot.visual_model,
            collision_model=kin.robot.collision_model,name=prefix+'/robot')
        robot.displayFrames(False); robot.displayCollisions(False)
        meshes = {}
        for label,episode in episodes.items():
            mesh = trimesh.load(str(episode['ref']['object_mesh']),force='mesh')
            object_path = prefix+f'/object{episode["info"]["episode_index"]}'
            actual = server.scene.add_mesh_trimesh(object_path,mesh)
            # Sparse edge overlay is optional and separate from the free object.
            hull = mesh.convex_hull
            outline = server.scene.add_mesh_simple(object_path+'_reference',hull.vertices,hull.faces,
                color=(240,170,50),wireframe=True,opacity=.55)
            meshes[label] = (actual,outline)
        label_handle = server.scene.add_label(prefix+'/label',name.upper())
        fixture = server.scene.add_line_segments(prefix+'/fixture',np.zeros((1,2,3),np.float32),
                                                colors=(85,110,130),line_width=4.)
        nodes.append(dict(group=group,robot=robot,meshes=meshes,label=label_handle,fixture=fixture,name=name))
    server.gui.add_markdown('## Three full episodes\n\n**Fixed world frame.** Solid robot and object: simulated motion. Moving pelvis fixture; no walking.')
    select = server.gui.add_dropdown('Episode',options=list(episodes))
    mode = server.gui.add_dropdown('View',options=['Before / after','Corrected only','Before only'])
    frame = server.gui.add_slider('Frame',min=0,max=857,step=1,initial_value=0)
    play = server.gui.add_checkbox('Play',initial_value=False)
    speed = server.gui.add_dropdown('Speed',options=['1×','0.5×','0.25×'],initial_value='0.5×')
    whole = server.gui.add_button('Focus whole robot')
    with server.gui.add_folder('Camera and overlays',expand_by_default=False):
        desired = server.gui.add_checkbox('Reference object outline',initial_value=False)
        collisions = server.gui.add_checkbox('Robot collision geometry',initial_value=False)
        hands = server.gui.add_button('Focus hands')
        object_button = server.gui.add_button('Focus actual object')
        episode_button = server.gui.add_button('Fit whole episode')
    status = server.gui.add_markdown('')
    with server.gui.add_folder('Whole-episode result and assumptions',expand_by_default=False):
        overview = server.gui.add_markdown('')
        server.gui.add_markdown(f'Random seed **{selection["seed"]}**, drawn from all 30 Tier 1 episodes before evaluation. All frames retained at source timing.\n\n'
            'The solid object is simulated. The optional orange outline is the recorded target. Mass and friction are assumed; '
            'the stroller is one rigid body without articulated wheels. No object pose resets, object welds or externally applied object forces during replay.\n\n'
            '**Left:** physical replay driven by the original contact-aware retargeted commands. '
            '**Right:** physical replay driven by corrected waist/arm/hand commands plus hand-force feedforward. '
            'Both use the same supported pelvis path and frozen legs. These are simulated states, not direct command-pose playback.\n\n'
            'World axes stay fixed. The two copies have constant display offsets of −0.8 m / +0.8 m along world Y; '
            'single-view modes use the original saved world coordinates. Camera focus changes only the camera.')
    lock = threading.RLock()

    def current_frame():
        episode = episodes[select.value]; ref = episode['ref']; j = min(int(frame.value),len(ref['timestamps'])-1)
        return episode,j

    def camera(client,kind='whole'):
        episode,j = current_frame()
        compare = mode.value == 'Before / after'
        chosen = 'baseline' if mode.value == 'Before only' else 'corrected'
        centers = []
        for name in (('baseline','corrected') if compare else (chosen,)):
            if kind == 'episode':
                points = np.vstack((episode[name]['q'][:,:3],episode[name]['object_positions'],episode['ref']['object_positions'])).copy()
                if compare: points[:,1] += -.8 if name == 'baseline' else .8
                centers.extend(points)
                continue
            if kind == 'hands': point = episode[name]['fk_tip_positions'][j].mean(0)
            elif kind == 'object': point = episode[name]['object_positions'][j]
            else:
                point = episode[name]['q'][j,:3].copy()
                point[2] = (point[2]+.7)/2
            center = point.copy()
            if compare: center[1] += -.8 if name == 'baseline' else .8
            centers.append(center)
        centers = np.asarray(centers)
        center = (centers.min(0)+centers.max(0))/2
        aspect = max(.25,client.camera.aspect)
        # Only the camera moves. Saved world trajectories and world orientation
        # are never recentered, yaw-aligned, or transformed as time advances.
        half_width = np.ptp(centers[:,1])/2+(.5 if kind == 'whole' else .35)
        distance = max(2.5 if kind == 'whole' else 1.8, .4+half_width/(np.tan(.85/2)*aspect))
        offset = np.array([distance,0.,.35*distance] if compare else ([1.,-.9,.5] if kind != 'whole' else [2.,-1.5,.8]))
        if kind == 'episode':
            # Bound the complete episode in a sphere, including robot extent.
            radius = float(np.max(np.linalg.norm(centers-center,axis=1)))+.8
            half_fov = min(.85/2,np.arctan(np.tan(.85/2)*aspect))
            offset = np.array([1.,0.,.35])*radius/np.sin(half_fov)
        client.camera.position = center+offset
        client.camera.look_at = center  # LAST: Viser position setter moves look_at.
        client.camera.up_direction = (0.,0.,1.)
        client.camera.fov = .85

    def update(_=None):
        with lock:
            episode,j = current_frame()
            ref = episode['ref']; compare = mode.value == 'Before / after'
            with server.atomic():
                grid.position = (0.,0.,float(ref['floor_z']))
                for i,node in enumerate(nodes):
                    name = node['name']; data = episode[name]
                    visible = compare or (name == ('baseline' if mode.value == 'Before only' else 'corrected'))
                    node['group'].visible = visible
                    offset = np.array([0.,-.8 if i == 0 else .8,0.]) if compare else np.zeros(3)
                    node['group'].wxyz = (1.,0.,0.,0.)
                    node['group'].position = offset
                    node['robot'].display(data['q'][j]); node['robot'].displayCollisions(collisions.value)
                    for label,(actual,outline) in node['meshes'].items():
                        actual.visible = label == select.value
                        outline.visible = label == select.value and desired.value
                        if label == select.value:
                            actual.position = data['object_positions'][j]; actual.wxyz = data['object_wxyz'][j]
                            outline.position = ref['object_positions'][j]; outline.wxyz = ref['object_wxyz'][j]
                    root = data['q'][j,:3]
                    node['label'].position = root+[0.,0.,.9]
                    node['label'].text = ('ORIGINAL COMMANDS' if i == 0 else 'CORRECTED + FORCE GUIDANCE')+' · simulated'
                    floor = root.copy(); floor[2] = float(ref['floor_z'])
                    node['fixture'].points = np.array([[floor,root]],np.float32)
                before,after = episode['baseline'],episode['corrected']
                verdict = 'PASS' if episode['report']['accepted_tracking'] else 'FAIL'
                status.content = (f'Full-episode tracking: **{verdict}**\n\n'
                    f'**{j+1} / {len(ref["timestamps"])} frames** · {ref["timestamps"][j]:.2f} / {ref["timestamps"][-1]:.2f} s\n\n'
                    f'Object position error, before → after: **{1000*before["position_error"][j]:.0f} → {1000*after["position_error"][j]:.0f} mm**.\n\n'
                    f'Rotation error: **{np.rad2deg(before["rotation_error"][j]):.1f} → {np.rad2deg(after["rotation_error"][j]):.1f}°**.\n\n'
                    f'After: L/R contact patches **{after["contacts_per_hand"][j,0]} / {after["contacts_per_hand"][j,1]}**.')

    def episode_changed(_=None):
        with lock:
            play.value = False
            episode = episodes[select.value]; report = episode['report']; ref = episode['ref']
            frame.value = 0; frame.max = len(ref['timestamps'])-1
            a,b = report['baseline'],report['corrected']
            overview.content = (f'**{episode["info"]["task"]}**\n\n'
                f'Full-episode tracking gate: **{"PASS" if report["accepted_tracking"] else "FAIL"}**. Not validated for training.\n\n'
                f'RMS object position error: **{a["rms_position_error_mm"]:.0f} → {b["rms_position_error_mm"]:.0f} mm**. '
                f'Worst: **{a["max_position_error_mm"]:.0f} → {b["max_position_error_mm"]:.0f} mm**.\n\n'
                f'Worst rotation error: **{a["max_rotation_error_deg"]:.1f} → {b["max_rotation_error_deg"]:.1f}°**. '
                f'Worst penetration: **{a["max_penetration_mm"]:.1f} → {b["max_penetration_mm"]:.1f} mm**.\n\n'
                f'Assumed mass **{report["mass_kg"]:g} kg**, friction **{report["friction"]}**, '
                f'{report["convex_parts"]} convex collision pieces. Physics step 1 ms. '
                f'Common initial floor alignment: **{1000*report["floor_z_m"]:.1f} mm**. '
                f'Method + three replays: **{report["method_seconds"]:.1f} s CPU wall time**.')
            update()
            for client in server.get_clients().values(): camera(client)

    @server.on_client_connect
    def connect(client): camera(client)
    @whole.on_click
    def focus_whole(_):
        for client in server.get_clients().values(): camera(client)
    @hands.on_click
    def focus_hands(_):
        for client in server.get_clients().values(): camera(client,'hands')
    @object_button.on_click
    def focus_object(_):
        for client in server.get_clients().values(): camera(client,'object')
    @episode_button.on_click
    def fit_episode(_):
        for client in server.get_clients().values(): camera(client,'episode')
    @mode.on_update
    def change_mode(_):
        update()
        for client in server.get_clients().values(): camera(client)
    select.on_update(episode_changed)
    for control in (frame,desired,collisions): control.on_update(update)
    episode_changed()
    print(f'Full episodes: http://127.0.0.1:{port}',flush=True)
    while True:
        if play.value:
            with lock:
                frame.value = 0 if frame.value >= frame.max else frame.value+1
        time.sleep((1/30)/{'1×':1.,'0.5×':.5,'0.25×':.25}[speed.value])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('correction/outputs/random_three_full_20261019'))
    parser.add_argument('--port',type=int,default=8084)
    args = parser.parse_args(); serve(args.directory,args.port)
