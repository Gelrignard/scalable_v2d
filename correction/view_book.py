"""Side-by-side replay of a real Tier 1 reference and local contact correction."""
from pathlib import Path
import json
import re
import time
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation
from .retarget_bridge import WORKSPACE
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.output import load_result
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer


class NamedVisualizer(ViserVisualizer):
    def __init__(self, *args, name, **kwargs):
        self.namespace=name
        super().__init__(*args, **kwargs)

    def loadViewerModel(self, **kwargs):
        super().loadViewerModel(rootNodeName=self.namespace, **kwargs)


def serve(directory,port=8082):
    records=[load_result(directory/'retargeted/contact_aware.npz')[0],
             load_result(directory/'dynamics_guided.npz')[0]]
    plan=np.load(directory/'wrench_plan.npz')
    metrics=json.loads((directory/'robot_repair_metrics.json').read_text())
    audit=json.loads((directory/'fk_force_audit.json').read_text())
    ids=plan['ids']
    server=viser.ViserServer(host='127.0.0.1',port=port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor',width=6,height=6,plane='xy')
    offsets=[np.array([0.,.5,0.]),np.array([0.,-.5,0.])]
    tables=[]
    visuals=[]; objects=[]; tips=[]; targets=[]; errors=[]; force_lines=[]; labels=[]
    mesh=trimesh.load(str(records[0]['object_mesh']),force='mesh',process=False)
    for k,(label,data,offset) in enumerate(zip(['BEFORE: retargeted','AFTER: contact correction'],records,offsets)):
        name=f'/comparison/{k}'
        kin=G1Kinematics()
        visual=NamedVisualizer(server,model=kin.robot.model,visual_model=kin.robot.visual_model,
            collision_model=kin.robot.collision_model,name=name+'/robot')
        visual.displayFrames(False); visual.displayCollisions(False)
        visuals.append(visual)
        objects.append(server.scene.add_mesh_trimesh(name+'/book',mesh))
        tips.append(server.scene.add_point_cloud(name+'/tips',points=np.zeros((6,3),np.float32),colors=(215,55,180),point_size=.008))
        targets.append(server.scene.add_point_cloud(name+'/targets',points=np.zeros((6,3),np.float32),colors=(25,185,100),point_size=.009))
        errors.append(server.scene.add_line_segments(name+'/gaps',points=np.zeros((6,2,3),np.float32),colors=(235,85,45),line_width=2.))
        force_lines.append(server.scene.add_line_segments(name+'/forces',points=np.zeros((6,2,3),np.float32),colors=(25,145,225),line_width=3.))
        labels.append(server.scene.add_label(name+'/label',label,position=(-.1,float(offset[1]),1.6)))
        support=Path(json.loads(str(data['metadata_json']))['source']['support_surface']).read_text()
        radius=float(re.search(r'double radius = ([\d.eE+-]+)',support).group(1))
        height=float(re.search(r'double height = ([\d.eE+-]+)',support).group(1))
        position=np.array(list(map(float,re.search(r'xformOp:translate = \(([^)]+)\)',support).group(1).split(','))))
        cylinder=trimesh.creation.cylinder(radius=radius,height=height)
        cylinder.visual.face_colors=[175,157,139,255]
        tables.append(server.scene.add_mesh_trimesh(name+'/table',cylinder,position=position+offset))
    server.gui.add_markdown('## Tier 1 · Book · Episode 11\nSame object motion and timing in both replays.\n\n**Kinematic replay — no physical rollout.**')
    frame=server.gui.add_slider('Frame',min=0,max=len(records[0]['q'])-1,step=1,initial_value=155)
    play=server.gui.add_checkbox('Play',initial_value=False)
    phase_only=server.gui.add_checkbox('Loop corrected segment',initial_value=True)
    marks=server.gui.add_checkbox('Contact markers',initial_value=True)
    forces=server.gui.add_checkbox('Planned force vectors',initial_value=False)
    focus=server.gui.add_button('Focus on hands')
    overview=server.gui.add_button('Whole body')
    status=server.gui.add_markdown('')
    server.gui.add_markdown(f"**Segment: {records[0]['timestamps'][ids[0]]:.2f}–{records[0]['timestamps'][ids[-1]]:.2f} s**\n\nMean surface gap: **{audit['before']['contact_gap_absolute_mean_mm']:.1f} → {audit['after']['contact_gap_absolute_mean_mm']:.1f} mm**.\n\nFrames passing instantaneous wrench + probe checks: **{audit['before']['frames_passing_wrench_and_probe_checks']} → {audit['after']['frames_passing_wrench_and_probe_checks']} / {len(ids)}**.\n\nGreen: targets. Magenta: robot tips. Orange: target error (includes tangential movement).\n\nRemaining object-relative tip speed: **{audit['after']['mean_object_relative_tip_speed_m_s']*1000:.0f} mm/s** mean.\n\n**Sticking and full robot dynamics remain unverified.** Assumed 1 kg, friction 0.6; no physical rollout.")

    def update(_=None):
        t=int(frame.value)
        with server.atomic():
            for k,(data,offset) in enumerate(zip(records,offsets)):
                q=data['q'][t].copy(); q[:3]+=offset
                visuals[k].display(q)
                objects[k].position=data['object_positions'][t]+offset
                objects[k].wxyz=data['object_wxyz'][t]
                labels[k].position=data['object_positions'][t]+offset+np.array([0.,0.,.25])
                actual=data['fk_tip_positions'][t]+offset
                target=data['tip_targets'][t]+offset
                tips[k].points=actual.astype(np.float32); tips[k].visible=marks.value
                targets[k].points=target.astype(np.float32); targets[k].visible=marks.value
                errors[k].points=np.stack((actual,target),axis=1).astype(np.float32); errors[k].visible=marks.value
                force_lines[k].visible=forces.value and t in ids
                if t in ids:
                    f=plan['before_forces' if k==0 else 'after_forces'][t-ids[0]]
                    rotation=Rotation.from_quat(data['object_wxyz'][t],scalar_first=True).as_matrix()
                    force_lines[k].points=np.stack((target,target+.014*(f@rotation.T)),axis=1).astype(np.float32)
            status.content=f"Time **{records[0]['timestamps'][t]:.3f} s** · Frame **{t}**\n\nMean tip target gap: **{records[0]['tip_error_m'][t].mean()*1000:.1f} / {records[1]['tip_error_m'][t].mean()*1000:.1f} mm** (before / after)."

    def camera(client,close=False):
        t=int(frame.value)
        center=records[0]['object_positions'][t].copy() if close else np.array([-.35,0.,.75])
        client.camera.up_direction=(0.,0.,1.)
        client.camera.position=center+np.array([-1.35,-.01,.16] if close else [-2.8,-1.,1.4])
        client.camera.look_at=center

    @server.on_client_connect
    def connect(client):
        camera(client)

    @focus.on_click
    def focus_camera(_):
        for table in tables: table.visible=False
        for client in server.get_clients().values(): camera(client,True)

    @overview.on_click
    def overview_camera(_):
        for table in tables: table.visible=True
        for client in server.get_clients().values(): camera(client)

    for control in (frame,marks,forces): control.on_update(update)
    update()
    print(f'Comparison viewer: http://127.0.0.1:{port}',flush=True)
    while True:
        if play.value:
            lo,hi=(int(ids[0])-15,int(ids[-1])+15) if phase_only.value else (0,len(records[0]['q'])-1)
            frame.value=lo if int(frame.value)>=hi or int(frame.value)<lo else int(frame.value)+1
        time.sleep(1/30)


if __name__=='__main__':
    serve(WORKSPACE/'correction/outputs/tier1_book')
