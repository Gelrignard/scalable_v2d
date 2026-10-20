"""Audit inactive finger paths, freeze one bounded repair, then replay separately."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.finger_transition import candidates, audit_path, refine_free_path
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists(): raise ValueError('Use a fresh output directory')
    start=perf_counter(); cpu=process_cost()['cpu_seconds']
    prior=json.loads((args.source/'report.json').read_text())
    ref=read_npz(args.source/'reference.npz')
    observed_positions=ref['object_positions'].copy()
    observed_rotations=ref['object_wxyz'].copy()
    if args.collision_prediction:
        prediction=read_npz(args.collision_prediction/'mechanics.npz')
        if prediction['max_object_applied_force']!=0 or len(prediction['q'])!=len(ref['q']):
            raise ValueError('Collision prediction needs an unassisted full episode')
        ref['object_positions']=prediction['object_positions'].copy()
        ref['object_wxyz']=prediction['object_wxyz'].copy()
    _,_,_,scene=load_strategy_scene(prior['input_directory'])
    points=[SurfacePoint(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in p.items()}) for p in prior['contacts']]
    args.output.mkdir(parents=True)
    report=dict(source=str(args.source.resolve()), input_directory=prior['input_directory'],
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        contacts=prior['contacts'], phase_contacts=prior.get('phase_contacts'), controller=prior['controller'],
        physics=prior['physics'], kp_hand=prior['kp_hand'], cases={}, path_candidates={},
        collision_prediction=str(args.collision_prediction) if args.collision_prediction else None,
        collision_prediction_sha256=(hashlib.sha256((args.collision_prediction/'mechanics.npz').read_bytes()).hexdigest()
                                     if args.collision_prediction else None),
        status='screening', ready_for_training=False, physical_success=False,
        provenance=snapshot_sources(args.output,[Path(__file__),Path('correction/src/optimization/finger_transition.py'),
                                                Path('correction/src/dynamics/collision_distance.py')]))
    selected=None
    for name,command in candidates(scene,dict(ref,_points=points)):
        if args.refine and name!='all_open': continue
        optimization=None
        if args.refine:
            command,optimization=refine_free_path(scene,command,dense_pairs=args.dense_pairs)
        audit=audit_path(scene,command)
        audit['optimization']=optimization
        report['path_candidates'][name]=audit
        if not audit['accepted']:
            filename=f'diagnostic_rejected_{name}.npz'
            np.savez_compressed(args.output/filename,**command)
            audit['rejected_debug_artifact']=filename
        if audit['accepted'] and selected is None:
            selected=name
            command['object_positions']=observed_positions
            command['object_wxyz']=observed_rotations
            np.savez_compressed(args.output/'reference.npz',**command)
    report.update(status='command_only' if selected else 'rejected_before_replay', selected=selected,
                  wall_seconds=perf_counter()-start,cost=process_cost())
    report['cost']['cpu_seconds']-=cpu
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--collision-prediction',type=Path)
    parser.add_argument('--dense-pairs',action='store_true')
    run(parser.parse_args())
