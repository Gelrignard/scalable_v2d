"""Refine an inactive withdrawal against a preceding unassisted object rollout."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.finger_transition import refine_free_path, audit_path
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    start=perf_counter(); cpu=process_cost()['cpu_seconds']
    prior=json.loads((args.source/'report.json').read_text())
    ref=read_npz(args.source/'reference.npz')
    prediction=read_npz(args.prediction/'mechanics.npz')
    if prediction['max_object_applied_force']!=0 or len(prediction['q'])!=len(ref['q']):
        raise ValueError('A full unassisted prediction is required')
    if np.any(ref['contact_active'][args.first:args.last+1]):
        raise ValueError('This repair changes only unloaded withdrawal')
    np.testing.assert_array_equal(ref['initial_q'],prediction['q'][0])
    _,_,_,scene=load_strategy_scene(prior['input_directory'])
    args.output.mkdir(parents=True)
    report={key:prior[key] for key in ('input_directory','contacts','phase_contacts','physics','controller','kp_hand')}
    report.update(source=str(args.source.resolve()), prediction=str(args.prediction.resolve()),
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        prediction_sha256=hashlib.sha256((args.prediction/'mechanics.npz').read_bytes()).hexdigest(),
        provenance=snapshot_sources(args.output,[Path(__file__),Path('correction/src/optimization/finger_transition.py'),
                                               Path('correction/src/dynamics/collision_distance.py')]),
        status='running',frames=[args.first,args.last],ready_for_training=False,physical_success=False)
    observed_p,observed_r=ref['object_positions'].copy(),ref['object_wxyz'].copy()
    ref['object_positions']=prediction['object_positions'].copy()
    ref['object_wxyz']=prediction['object_wxyz'].copy()
    command,optimization=refine_free_path(scene,ref,start=args.first,end=args.last,iterations=30,seconds=20.)
    report['optimization']=optimization
    report['geometry']=audit_path(scene,command,start=args.first,end=args.last)
    command['object_positions']=observed_p
    command['object_wxyz']=observed_r
    report['status']='command_only' if report['geometry']['accepted'] else 'rejected_before_replay'
    if report['geometry']['accepted']:
        np.savez_compressed(args.output/'reference.npz',**command)
    report['wall_seconds']=perf_counter()-start
    report['cost']=process_cost();report['cost']['cpu_seconds']-=cpu
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','geometry','optimization','wall_seconds')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prediction',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first',type=int,default=746)
    parser.add_argument('--last',type=int,default=857)
    run(parser.parse_args())
