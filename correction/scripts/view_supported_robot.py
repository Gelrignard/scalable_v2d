"""Inspect actual supported-G1 physics, not a pose-assigned robot animation."""
import argparse
import json
from pathlib import Path
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
    report=json.loads((directory/'report.json').read_text())
    ref=read(directory/'reference.npz')
    records={name:read(directory/f'{name}.npz') for name in ('baseline','geometry_only','wrench_seed','corrected')}
    server=viser.ViserServer(host='127.0.0.1',port=port)
    server.gui.configure_theme(control_layout='collapsible',control_width='small',show_logo=False,
                               show_share_button=False,brand_color=(34,135,158))
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor',width=5,height=5,plane='xy')
    box=trimesh.creation.box(extents=np.ptp(ref['bounds'],axis=0));box.apply_translation(ref['bounds'].mean(0))
    nodes=[]
    for i in range(2):
        prefix=f'/trial{i}'
        group=server.scene.add_frame(prefix,show_axes=False)
        kin=G1Kinematics()
        robot=NamedVisualizer(server,model=kin.robot.model,visual_model=kin.robot.visual_model,
                               collision_model=kin.robot.collision_model,name=prefix+'/robot')
        robot.displayFrames(False);robot.displayCollisions(False)
        book=server.scene.add_mesh_simple(prefix+'/book',box.vertices,box.faces,color=(58,160,194))
        outline=server.scene.add_mesh_simple(prefix+'/desired',box.vertices,box.faces,wireframe=True,color=(210,150,60),opacity=.6)
        table=trimesh.creation.cylinder(radius=float(ref['table_radius']),height=float(ref['table_height']))
        server.scene.add_mesh_simple(prefix+'/table',table.vertices,table.faces,position=ref['table_center'],color=(170,156,135),opacity=.6)
        root=ref['fixed_q'][:3]
        mount=trimesh.creation.cylinder(radius=.025,height=float(root[2]))
        server.scene.add_mesh_simple(prefix+'/fixture_symbol',mount.vertices,mount.faces,
                                     position=(float(root[0]),float(root[1]),float(root[2]/2)),color=(70,90,105),opacity=.6)
        label=server.scene.add_label(prefix+'/label','')
        forces=server.scene.add_line_segments(prefix+'/forces',np.zeros((2,2,3),np.float32),colors=(240,90,65),line_width=3.)
        nodes.append(dict(group=group,robot=robot,book=book,outline=outline,label=label,forces=forces))
    server.gui.add_markdown('## G1 + Dex3 · supported physics\n\n**Pelvis and legs fixed. Arms, fingers, waist and book dynamic.**')
    if not report['accepted_tracking']:
        server.gui.add_markdown('Book-carry integration demo. **Tracking gate not met; not ready for training.**')
    mode=server.gui.add_dropdown('Compare',options=['Corrected only','Raw retarget / corrected','Geometry / corrected','Wrench seed / corrected'])
    frame=server.gui.add_slider('Source frame',min=int(ref['frame_ids'][0]),max=int(ref['frame_ids'][-1]),step=1,initial_value=int(ref['frame_ids'][0]))
    play=server.gui.add_checkbox('Play',initial_value=True)
    speed=server.gui.add_dropdown('Playback speed',options=['1×','0.5×','0.25×'],initial_value='0.5×')
    desired=server.gui.add_checkbox('Desired book outline',initial_value=True)
    arrows=server.gui.add_checkbox('Measured hand contact forces',initial_value=True)
    collision=server.gui.add_checkbox('Collision geometry',initial_value=False)
    focus=server.gui.add_button('Focus on hands')
    whole=server.gui.add_button('Show whole robot')
    status=server.gui.add_markdown('')
    m=report['corrected']
    with server.gui.add_folder('Measured result and assumptions',expand_by_default=False):
        server.gui.add_markdown(f'Worst book error: **{m["max_position_error_mm"]:.1f} mm / {m["max_rotation_error_deg"]:.1f}°**. '
            f'Max overlap: **{m["max_penetration_mm"]:.3f} mm**.\n\n'
            f'Source duration {report["source_duration_s"]:.1f} s; executed over **{report["rollout_duration_s"]:.1f} s**. '
            f'Same object pose path; {report["time_scale"]:g}× time scale.\n\n'
            '1 kg rigid-box book, friction 0.6, imported G1 collision geometry/inertias. '
            'Motor torques are clipped; finger cap 0.76 Nm. No applied book wrench, object weld, or pose reset during rollout.\n\n'
            'Starts in a prepared two-hand grasp; grasp acquisition, release and walking are excluded. '
            'The grey post symbolizes the fixed-base fixture. No SONIC and no RL. '
            'These are local MuJoCo assumptions, not official challenge validation.')

    def camera(client,close=False):
        j=int(frame.value)-int(ref['frame_ids'][0])
        center=ref['object_positions'][j] if close else ref['fixed_q'][:3]+[0.,0.,.1]
        compare=mode.value!='Corrected only'
        offset=np.array([-1.5,0.,.45] if compare else ([-.65,-.6,.3] if close else [-1.25,-1.25,.7]))
        # Viser's position setter translates look_at too. Set the target LAST.
        client.camera.position=center+offset
        client.camera.look_at=center
        client.camera.up_direction=(0.,0.,1.)
        client.camera.fov=.8

    @server.on_client_connect
    def connect(client):camera(client)

    @focus.on_click
    def close(_):
        for client in server.get_clients().values():camera(client,True)

    @whole.on_click
    def wide(_):
        for client in server.get_clients().values():camera(client)

    def update(_=None):
        j=int(frame.value)-int(ref['frame_ids'][0])
        names={'Corrected only':('corrected','corrected'),'Raw retarget / corrected':('baseline','corrected'),
               'Geometry / corrected':('geometry_only','corrected'),'Wrench seed / corrected':('wrench_seed','corrected')}[mode.value]
        compare=mode.value!='Corrected only'
        with server.atomic():
            for i,node in enumerate(nodes):
                node['group'].visible=compare or i==0
                node['group'].position=(0.,(.7 if i==0 else -.7) if compare else 0.,0.)
                data=records[names[i]]
                node['robot'].display(data['q'][j]);node['robot'].displayCollisions(collision.value)
                node['book'].position=data['object_positions'][j];node['book'].wxyz=data['object_wxyz'][j]
                node['outline'].position=ref['object_positions'][j];node['outline'].wxyz=ref['object_wxyz'][j]
                node['outline'].visible=desired.value
                node['label'].position=data['object_positions'][j]+[0.,0.,.28]
                node['label'].text=names[i].replace('_',' ').upper()+' · simulated'
                points=data['contact_points_world'][j];force=data['contact_forces_world'][j]
                node['forces'].points=np.stack([points,points+.003*force],axis=1).astype(np.float32)
                node['forces'].visible=arrows.value
            actual=records['corrected']
            status.content=(f'Frame **{frame.value}** · simulation **{ref["timestamps"][j]:.2f} s**\n\n'
                f'Book error **{1000*actual["position_error"][j]:.1f} mm / {np.rad2deg(actual["rotation_error"][j]):.1f}°**. '
                f'Contact patches L/R: **{actual["contacts_per_hand"][j,0]} / {actual["contacts_per_hand"][j,1]}**.')
    @mode.on_update
    def change_mode(_):
        update()
        for client in server.get_clients().values():camera(client,True)
    for control in (frame,desired,arrows,collision):control.on_update(update)
    update()
    print(f'Supported G1 physics: http://127.0.0.1:{port}',flush=True)
    while True:
        if play.value:frame.value=int(ref['frame_ids'][0]) if frame.value>=int(ref['frame_ids'][-1]) else frame.value+1
        multiplier={'1×':1.,'0.5×':.5,'0.25×':.25}[speed.value]
        time.sleep(float(np.median(np.diff(ref['timestamps'])))/multiplier)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('correction/outputs/supported_g1_20261019'))
    parser.add_argument('--port',type=int,default=8084)
    args=parser.parse_args();serve(args.directory,args.port)

if __name__=='__main__':main()
