"""Saved full-episode robot comparisons, with optional matching pad simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np
import trimesh
import viser
from correction.scripts.view_controller_comparison import read, NamedVisualizer
from correction.scripts.contact_overlay import ContactOverlay, FORCE_SCALE
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics


def load_episode(entry):
    directory = Path(entry['directory'])
    original_directory = Path(entry.get('original_directory', directory))
    report = json.loads((directory/'report.json').read_text())
    original_report = json.loads((original_directory/'report.json').read_text())
    ref = read(directory/'reference.npz')
    original_ref = read(original_directory/'reference.npz')
    for key in ('timestamps', 'object_positions', 'object_wxyz'):
        np.testing.assert_array_equal(ref[key], original_ref[key], err_msg='Comparison targets must match')
    records = [read(original_directory/'original.npz'), read(directory/'mechanics.npz')]
    pad_report, pad_ref, pads = None, None, []
    if entry.get('support_directory'):
        support = Path(entry['support_directory'])
        pad_report = json.loads((support/'report.json').read_text())
        pad_ref = read(support/'reference.npz')
        if pad_report['source_reference_sha256'] != hashlib.sha256((directory/'reference.npz').read_bytes()).hexdigest():
            raise ValueError('Robot and pad channels must share the identical saved reference')
        pads = [read(support/f'{name}.npz') for name in ['original','corrected']]
    count = len(ref['timestamps'])
    for record in records:
        if len(record['q']) != count or float(record['max_object_applied_force']) != 0:
            raise ValueError('Complete unassisted robot episode required')
        np.testing.assert_array_equal(record['timestamps'], ref['timestamps'])
    for key in ('q', 'object_positions', 'object_wxyz'):
        np.testing.assert_array_equal(records[0][key][0], records[1][key][0],
                                      err_msg='Comparisons must start from identical physical poses')
    for record in pads:
        np.testing.assert_array_equal(record['timestamps'],ref['timestamps'])
        if float(record['max_object_applied_force']) != 0:
            raise ValueError('Unassisted pad episode required')
    return dict(entry=entry,report=report,pad_report=pad_report,ref=ref,pad_ref=pad_ref,
                records=records,pads=pads,count=count,
                summaries=[original_report['cases']['original'], report['cases']['mechanics']],
                guard_rows=json.loads(str(records[1]['guard_log_json']))
                    if 'guard_log_json' in records[1] else [],
                recovery_rows=json.loads(str(records[1]['recovery_log_json']))
                    if 'recovery_log_json' in records[1] else [])


class CatalogViewer:
    def __init__(self, entries, port):
        self.entries={item['name']:item for item in entries}
        if len(self.entries) != len(entries):
            raise ValueError('Episode names must be unique')
        self.lock=threading.RLock(); self.playing=threading.Event(); self.loading=True
        self.state=None
        self.server=server=viser.ViserServer(host='127.0.0.1',port=port)
        server.gui.configure_theme(control_layout='collapsible',control_width='small',
            show_logo=False,show_share_button=False,brand_color=(34,135,158))
        self.kin=G1Kinematics()
        self.episode=server.gui.add_dropdown('Episode',options=list(self.entries))
        self.channel=server.gui.add_dropdown('Channel',
            options=['Robot simulation','Independent support-point simulation'])
        self.legend=server.gui.add_markdown('')
        self.frame=server.gui.add_slider('Frame',min=0,max=1,step=1,initial_value=0)
        self.play=server.gui.add_checkbox('Play',initial_value=False)
        self.speed=server.gui.add_dropdown('Speed',options=['0.25','0.5','1'],initial_value='0.5')
        self.force_scale=server.gui.add_number('Force scale (mm/N)',initial_value=1000*FORCE_SCALE,
            min=.1,max=30.,step=1.,visible=False)
        self.focus_robot=server.gui.add_button('Focus robot')
        self.focus_objects=server.gui.add_button('Focus objects')
        self.status=server.gui.add_markdown('')
        self.forces=server.gui.add_markdown('',visible=False)
        with server.gui.add_folder('Controller diagnosis',expand_by_default=False):
            self.controller_status=server.gui.add_markdown('')
        with server.gui.add_folder('Model and result details',expand_by_default=False):
            self.details=server.gui.add_markdown('')
        self.episode.on_update(lambda _:self.load())
        self.channel.on_update(self.change_channel)
        self.frame.on_update(self.update)
        self.force_scale.on_update(self.update)
        self.play.on_update(self.play_changed)
        self.focus_robot.on_click(lambda _:self.focus())
        self.focus_objects.on_click(lambda _:self.focus(True))
        server.on_client_connect(lambda client:self.camera(client))
        server.on_client_disconnect(lambda _:self.pause_without_clients())
        self.load()

    def load(self):
        with self.lock:
            self.loading=True; self.play.value=False; self.playing.clear()
            state=load_episode(self.entries[self.episode.value])
            ref,pad_ref=state['ref'],state['pad_ref']
            self.state=state; server=self.server
            self.channel.options=(['Robot simulation','Independent support-point simulation']
                                  if state['pads'] else ['Robot simulation'])
            self.channel.value='Robot simulation'
            self.channel.visible=bool(state['pads'])
            with server.atomic():
                server.scene.reset(); server.scene.set_up_direction('+z')
                server.scene.add_grid('/floor',width=8,height=8,plane='xy',
                    position=(0.,0.,float(ref['floor_z'])))
                server.scene.add_frame('/world_origin',axes_length=.2,axes_radius=.003)
                mesh=trimesh.load(str(ref['object_mesh']),force='mesh')
                target_path=np.stack((ref['object_positions'][:-1],ref['object_positions'][1:]),axis=1).astype(np.float32)
                state['nodes']=[]; state['overlays']=[]
                for i in range(2):
                    prefix=f'/trial{i}'
                    group=server.scene.add_frame(prefix,show_axes=False,position=(0.,-.8 if i==0 else .8,0.))
                    robot_group=server.scene.add_frame(prefix+'/robot',show_axes=False)
                    robot=NamedVisualizer(server,model=self.kin.robot.model,visual_model=self.kin.robot.visual_model,
                        collision_model=self.kin.robot.collision_model,name=prefix+'/robot')
                    robot.displayFrames(False)
                    actual=server.scene.add_mesh_trimesh(prefix+'/object',mesh)
                    pad_object=server.scene.add_mesh_simple(prefix+'/pad_object',mesh.vertices,mesh.faces,
                        color=(68,124,173),opacity=.3,visible=False,cast_shadow=False,receive_shadow=False)
                    target=server.scene.add_mesh_simple(prefix+'/intended',mesh.vertices,mesh.faces,
                        color=(255,143,25),opacity=.22,cast_shadow=False,receive_shadow=False)
                    outline=server.scene.add_mesh_simple(prefix+'/outline',mesh.vertices,mesh.faces,
                        color=(255,143,25),wireframe=True,opacity=.9,cast_shadow=False,receive_shadow=False)
                    server.scene.add_line_segments(prefix+'/intended_path',target_path,colors=(255,143,25),line_width=2.)
                    error=server.scene.add_line_segments(prefix+'/error',np.zeros((1,2,3),np.float32),colors=(220,65,65))
                    label=server.scene.add_label(prefix+'/label','')
                    state['nodes'].append((group,robot_group,robot,actual,pad_object,target,outline,error,label))
                    state['overlays'].append(ContactOverlay(server.scene,prefix+'/contacts',
                        pad_ref['original_tip_targets'] if i==0 else pad_ref['corrected_pad_targets'],
                        ref['contact_active'],planned_forces=None if i==0 else ref['force_ff'],
                        planned_origins=pad_ref.get('intended_object_contact_points'),
                        radius=state['pad_report']['physical_parameters']['pad_radius_m']) if pad_ref is not None else None)
                self.frame.max=state['count']-1; self.frame.value=0
                diagnostic=state['report'].get('diagnostic_fallback')
                selection=state['report'].get('selected_candidate')
                entry=state['entry']
                limitations=entry.get('display_limitations',state['report'].get('limitations',[]))
                self.details.content=('**Entire recorded episode.** Original and method start from the same pre-approach state. '
                    'Object and upper robot joints have physical dynamics and finite actuators; pelvis is externally supported. '
                    'No object assistance or phase resets. Walking and balance are not simulated.\\n\\n'
                    'World trajectories remain fixed. Display panels use constant Y offsets of -0.8/+0.8 m.\\n\\n'
                    'Orange and solid objects use the source mesh. Physics uses convex collision pieces; the orange ghost is not a collision-surface certificate.\\n\\n'+
                    ('Independent pads are a separate Cartesian-actuator experiment; they are not articulated robot execution.\\n\\n'
                     if state['pads'] else 'Robot replay only: no matching independent-pad replay was generated for this controller.\\n\\n')+
                    f'**Run provenance:** {entry.get("model_notes", "See saved experiment report.")}\\n\\n'
                    f'**Stage status:** {entry.get("stage_status", diagnostic or "Saved staged reference; physical acceptance remains separate.")}\\n\\n'
                    f'**Contact selection:** {selection or "Saved per-episode contact sequence."}\\n\\n'+
                    '\\n\\n'.join(limitations)).replace('\\n','\n')
            self.loading=False
            self.update(); self.focus()

    def camera(self, client, objects=False):
        if self.state is None or self.loading: return
        s=self.state; j=int(self.frame.value)
        pad=self.channel.value!='Robot simulation'
        points=[]
        for i,record in enumerate(s['pads'] if pad else s['records']):
            offset=np.array([0.,-.8 if i==0 else .8,0.])
            points.append(record['object_positions'][j]+offset if objects or pad else record['q'][j,:3]+offset)
            if objects: points.append(s['ref']['object_positions'][j]+offset)
        points=np.asarray(points); center=points.mean(0)
        half_width=np.ptp(points[:,1])/2+.5
        # Reserve the right-hand control panel when fitting both comparison
        # subjects. Only the camera changes; all world trajectories stay fixed.
        aspect=max(.3,client.camera.aspect)
        panel_fraction=min(.5,.34/aspect)
        distance=max(1.6,half_width/(np.tan(.85/2)*aspect*(1-panel_fraction)))
        view_offset=np.array([distance,-.3,.4*distance])
        right=np.cross(-view_offset,[0.,0.,1.]); right/=np.linalg.norm(right)
        center=center+right*(distance*np.tan(.85/2)*aspect*panel_fraction)
        client.camera.position=center+view_offset
        client.camera.look_at=center; client.camera.up_direction=(0.,0.,1.); client.camera.fov=.85

    def focus(self, objects=False):
        for client in self.server.get_clients().values(): self.camera(client,objects)

    def change_channel(self, _):
        self.play.value=False; self.update(); self.focus()

    def play_changed(self, _):
        if self.play.value: self.playing.set()
        else: self.playing.clear()

    def pause_without_clients(self):
        if not self.server.get_clients(): self.play.value=False

    def update(self, _=None):
        if self.loading or self.state is None: return
        with self.lock,self.server.atomic():
            s=self.state; ref=s['ref']; j=int(self.frame.value)
            pad=self.channel.value!='Robot simulation'
            self.force_scale.visible=pad; self.forces.visible=pad; self.focus_robot.visible=not pad
            self.legend.content=('**Orange ghost/path: intended object motion.** Blue translucent object and spheres: physical pad simulation. '
                'Cyan: intended pad paths. Purple: planned force. Green: measured force.'
                if pad else '**Orange ghost/path: intended object motion.** Solid robot/object: physical simulation. Red: tracking error.')
            lines=[f'**{self.episode.value}** · {j+1}/{s["count"]} frames · {ref["timestamps"][j]:.2f}/{ref["timestamps"][-1]:.2f} s']
            for i,node in enumerate(s['nodes']):
                _,robot_group,robot,actual,pad_object,target,outline,error,label=node
                data=(s['pads'] if pad else s['records'])[i]
                summary=(s['pad_report']['cases'][['original','corrected'][i]] if pad else
                         s['summaries'][i])
                robot_group.visible=not pad; actual.visible=not pad; pad_object.visible=pad
                if not pad: robot.display(data['q'][j])
                if s['overlays'][i] is not None:
                    s['overlays'][i].update(j,data,pad,self.force_scale.value/1000)
                for handle in [actual,pad_object]:
                    handle.position,handle.wxyz=data['object_positions'][j],data['object_wxyz'][j]
                for handle in [target,outline]:
                    handle.position,handle.wxyz=ref['object_positions'][j],ref['object_wxyz'][j]
                error.points=np.array([[data['object_positions'][j],ref['object_positions'][j]]],np.float32)
                label.position=(ref['object_positions'][j]+[0,0,.6] if pad else data['q'][j,:3]+[0,0,.85])
                label.text=('ORIGINAL RETARGETING' if i==0 else
                            s['entry'].get('method_label','OUR METHOD · diagnostic candidate'))
                lines.append(f'**{"Original" if i==0 else "Our method"}**: error {1000*data["position_error"][j]:.1f} mm; '
                             f'whole-episode RMS **{summary["rms_position_error_mm"]:.1f} mm**.')
            accepted=(s['pad_report']['cases']['corrected']['accepted_tracking'] if pad else
                      s['report']['cases']['mechanics']['accepted_physics'])
            lines.append('Physical '+('pad tracking' if pad else 'robot acceptance')+': **'+('passed' if accepted else 'failed')+'**.')
            if s['report'].get('diagnostic_fallback') or s['entry'].get('rejected'):
                lines.append('**Rejected optimization: this is a diagnostic candidate, not an accepted correction.**')
            self.status.content='\n\n'.join(lines)
            rows=s['recovery_rows']
            self.controller_status.visible=bool(rows) and not pad
            if rows and not pad:
                k=max(0,np.searchsorted([r['time_s'] for r in rows],float(ref['timestamps'][j]),side='right')-1)
                row=rows[k]; force=row['optimization'].get('force')
                force_text=('No force allocation: command QP failed.' if force is None else
                    f"Proposed force solve: **{force['status']}**; vertical wrench residual "
                    f"**{force['wrench_residual'][2]:.2f} N**. This is a proposal diagnostic, not measured force.")
                self.controller_status.content=(
                    f"Last replan: **{row['time_s']:.2f} s** · {row['mode']}\n\n"
                    f"Decision: **{row['selected']}**. Command QP: {row['optimization']['command']['status']}.\n\n"
                    f"Matching active patches at this saved frame: "
                    f"**{int(np.count_nonzero(s['records'][1]['recovery_touching'][j] & s['records'][1]['recovery_active'][j]))}**.\n\n"
                    +force_text+"\n\nRecorded replay of a closed-loop simulation; no optimizer is running in this viewer.")
            guards=s['guard_rows']
            if guards and not pad:
                self.controller_status.visible=True
                k=np.searchsorted([r['time_s'] for r in guards],float(ref['timestamps'][j]),side='right')-1
                if k < 0:
                    self.controller_status.content='Nominal execution before the first feedback decision.'
                else:
                    row=guards[k]
                    decision='Enable bounded feedback' if row['enabled'] else 'Smooth return toward nominal'
                    self.controller_status.content=(
                        f"Last decision: **{row['time_s']:.2f} s — {decision}**.\n\n"
                        f"Reasons: {', '.join(row['reasons']) or 'All relative adoption checks passed'}.\n\n"
                        f"Predicted task cost, return / feedback: **{row['nominal']['task_cost']:.3f} / "
                        f"{row['proposal'].get('task_cost',float('nan')):.3f}**.\n\n"
                        f"Predicted impulse cost, return / feedback: **{row['nominal']['impulse_cost']:.3f} / "
                        f"{row['proposal'].get('impulse_cost',float('nan')):.3f}**.\n\n"
                        "Forecast costs are dimensionless diagnostics, not current measured force. "
                        "Relative adoption is not physical acceptance. Saved closed-loop replay; no optimizer is running.")
            observed=s['records'][1]
            if 'controller_observed_goal_wrench' in observed and not pad:
                self.controller_status.visible=True
                goal=observed['controller_observed_goal_wrench'][j]
                unmet=observed['controller_observed_unmet_delta_wrench'][j]
                matched=int(np.count_nonzero(observed['controller_observed_matched_contact'][j]))
                delta=float(np.max(abs(observed['controller_observed_force_delta'][j])))
                self.controller_status.content=(
                    f'**Staged force feedback; no MPC.** Matching active patches: **{matched}**.\n\n'
                    f'Max force increment: **{delta:.3f} N per axis**. '
                    f'Requested additional vertical wrench: **{goal[2]:.3f} N**; '
                    f'unallocated: **{unmet[2]:.3f} N**.\n\n'
                    'These are commanded increments, not measured object forces. '
                    'Saved physical replay; no optimizer runs in this viewer.')
            if pad:
                measured=[np.linalg.norm(data['contact_forces'][j],axis=-1) for data in s['pads']]
                plan=np.linalg.norm(ref['force_ff'][j],axis=-1)
                names=s['pad_ref']['channel_names']
                self.forces.content=('| Patch | Original measured N | Our measured N | Plan N |\n|---|---:|---:|---:|\n'+
                    '\n'.join(f'| {name} | {measured[0][k]:.2f} | {measured[1][k]:.2f} | {plan[k]:.2f} |'
                        for k,name in enumerate(names)))

    def run(self):
        while True:
            self.playing.wait()
            with self.lock:
                dt=float(np.median(np.diff(self.state['ref']['timestamps'])))/float(self.speed.value)
            time.sleep(dt)
            with self.lock:
                if self.play.value and not self.loading:
                    if not self.server.get_clients() or self.frame.value==self.frame.max: self.play.value=False
                    else: self.frame.value+=1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8087)
    args=parser.parse_args()
    CatalogViewer(json.loads(args.catalog.read_text())['episodes'],args.port).run()
