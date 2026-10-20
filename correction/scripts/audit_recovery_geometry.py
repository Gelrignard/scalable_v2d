"""Check saved contact charts against source mesh and simulator collision pieces."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.observed_contact import ContactCharts
from correction.src.pass_traj.experiment_record import snapshot_sources,process_cost


def run(source,output):
    if output.exists():raise ValueError('Preserve previous audits')
    start=perf_counter();cpu=process_cost()['cpu_seconds']
    report=json.loads((source/'report.json').read_text());ref=read_npz(source/'reference.npz')
    _,_,_,scene=load_strategy_scene(report['input_directory'])
    points=[SurfacePoint(**{k:np.array(v) if isinstance(v,list) else v for k,v in p.items()}) for p in report['contacts']]
    library=ContactCharts(scene,points);mesh=trimesh.load(str(ref['object_mesh']),force='mesh')
    frames=np.flatnonzero(np.any(ref['contact_active'],axis=1));rows=[];seen=set()
    for frame in frames:
        key=np.r_[ref['robot_local_points'][frame].ravel(),ref['object_local_points'][frame].ravel()].round(6).tobytes()
        if key in seen:continue
        seen.add(key)
        chart=library.get(ref['robot_local_points'][frame],ref['object_local_points'][frame],ref['inward_normals_local'][frame])
        distances=[]
        for anchor in chart['object_local']:
            near=trimesh.triangles.closest_point(mesh.triangles,np.tile(anchor,(len(mesh.faces),1)))
            distances.append(float(np.linalg.norm(near-anchor,axis=1).min()))
        rows.append(dict(first_frame=int(frame),source_mesh_distance_m=distances,
                         chart_projection=library.audit[-1]))
    output.mkdir(parents=True)
    result=dict(source=str(source),charts=rows,seconds=perf_counter()-start,
        scope='Selected contact patches only; no global mesh equivalence or manipulation-success claim',
        intended_overlay='Viewer source now uses source mesh rather than global convex hull',
        provenance=snapshot_sources(output,[Path(__file__),Path('correction/src/dynamics/observed_contact.py'),
                                            Path('correction/src/optimization/surface_hypotheses.py')]),cost=process_cost())
    result['cost']['cpu_seconds']-=cpu
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['charts'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
