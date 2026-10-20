"""Locate acquisition/contact failures without substituting commanded for executed motion."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint, contact_kinematics


def run(directory, frames):
    report = json.loads((directory/'report.json').read_text())
    _, _, _, scene = load_strategy_scene(report['input_directory'])
    ref = read_npz(directory/'reference.npz')
    points = [SurfacePoint(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in p.items()})
              for p in report['contacts']]
    out = dict(contacts=[dict(body=scene.model.body(p.body).name, object_point=p.object_local.tolist(),
                             object_outward=p.object_outward.tolist(), source=p.source_channel) for p in points],
               cases={})
    for name in report['cases']:
        result = read_npz(directory/f'{name}.npz')
        rows = []
        for frame in frames:
            scene.set_state(result['q'][frame], result['object_positions'][frame],result['object_wxyz'][frame])
            position, _, _, _ = contact_kinematics(scene,points)
            target = result['object_positions'][frame]+Rotation.from_quat(result['object_wxyz'][frame],scalar_first=True).apply(
                np.array([p.object_local for p in points]))
            row = dict(frame=frame, object_error_mm=float(result['position_error'][frame]*1000),
                actual_patch_gaps_mm=(1000*np.linalg.norm(position-target,axis=1)).tolist(),
                contacts=result['contacts_per_hand'][frame].tolist(),
                all_hand_forces_n=result['contact_forces_world'][frame].tolist() if 'contact_forces_world' in result else None)
            planned = ref['baseline_command_q'] if name=='original' else ref['q']
            error = result['q'][frame,7+scene.movable]-planned[frame,7+scene.movable]
            row['joint_error_rad'] = {scene.kin.joint_names[i]:float(error[j]) for j,i in enumerate(scene.movable) if abs(error[j])>.03}
            row['actual_collision_pairs'] = [dict(pair=scene.model.geom(c.geom1).name+' / '+scene.model.geom(c.geom2).name,
                distance_mm=float(1000*c.dist)) for c in scene.data.contact if scene.object_geoms.intersection((c.geom1,c.geom2))]
            for key in ('touching','force_enabled','measured_contact_force','commanded_contact_force'):
                full='controller_'+key
                if full in result: row[key]=result[full][frame].tolist()
            rows.append(row)
        out['cases'][name]=rows
    (directory/'acquisition_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--frames',type=int,nargs='+',default=[210,225,235,250,270,280,290,300])
    args=parser.parse_args(); run(args.directory,args.frames)
