"""Capacity of contact geometry at measured robot/object poses (not inferred force)."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.wrench_screen import screen_wrench
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(directory):
    start=perf_counter();cpu=process_cost()['cpu_seconds']
    report=json.loads((directory/'report.json').read_text())
    actual=read_npz(directory/'mechanics.npz')
    _,_,_,scene=load_strategy_scene(report['input_directory'])
    frames=[]
    for frame in (645,650,660,665,670,675,680):
        scene.set_state(actual['q'][frame],actual['object_positions'][frame],actual['object_wxyz'][frame])
        contacts={}
        for c in scene.data.contact:
            if not scene.object_geoms.intersection((c.geom1,c.geom2)): continue
            first=c.geom1 in scene.object_geoms
            other=int(c.geom2 if first else c.geom1)
            body=int(scene.model.geom_bodyid[other])
            if 'hand' not in scene.model.body(body).name: continue
            key=(body,int(c.geom1 if first else c.geom2))
            if key in contacts and contacts[key]['distance_m']<c.dist: continue
            jac=np.zeros((3,scene.model.nv))
            mujoco.mj_jac(scene.model,scene.data,jac,None,c.pos,body)
            contacts[key]=dict(body=scene.model.body(body).name,position=c.pos.copy(),
                inward=(-1. if first else 1.)*np.asarray(c.frame)[:3],
                jac=jac[:,scene.active_v],distance_m=float(c.dist))
        rows=list(contacts.values())
        screen=None
        if rows:
            screen=screen_wrench([r['position'] for r in rows],[r['inward'] for r in rows],
                scene.data.xipos[scene.book_body],np.r_[-scene.model.body_mass[scene.book_body]*scene.model.opt.gravity,np.zeros(3)],
                [r['jac'] for r in rows],scene.data.qfrc_bias[scene.active_v],scene.caps,
                friction=report['controller']['friction'],normal_caps=15.)
        frames.append(dict(frame=frame,contacts=[{k:v for k,v in r.items() if k!='jac'} for r in rows],
            static_capacity=screen))
    audit=dict(scope='Geometry recomputed at actual saved poses; no measured forces inferred, no environment support in LP',
        frames=frames,wall_seconds=perf_counter()-start,cost=process_cost(),
        provenance=snapshot_sources(directory/'executed_geometry_sources',[Path(__file__)]))
    audit['cost']['cpu_seconds']-=cpu
    (directory/'executed_geometry_screen.json').write_text(json.dumps(audit,indent=2,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)+'\n')
    print(json.dumps([dict(frame=r['frame'],bodies=[c['body'] for c in r['contacts']],screen=r['static_capacity']) for r in frames],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
