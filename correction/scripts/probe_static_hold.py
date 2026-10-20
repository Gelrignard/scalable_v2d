"""Counterfactual diagnostic: does a screened grasp hold if already acquired?

This deliberately starts at the optimized grasp, NOT the episode start. It
cannot count as successful acquisition or an accepted manipulation demo.
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene,read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources,process_cost


def run(candidates,output,motion_from=None):
    if output.exists():raise ValueError('Fresh output required')
    start=perf_counter();saved=json.loads((candidates/'report.json').read_text())
    candidate=next(c for c in saved['candidates'] if c['candidate_for_acquisition'] and not c['static_hold_screen']['floor_contacts_included'])
    _,raw,_,scene=load_strategy_scene(saved['source'])
    q0=read_npz(candidates/'geometry_candidates.npz')['q'][candidate['solution_index']]
    frame=candidate['frame'];times=np.arange(91)/30
    q=np.tile(q0,(len(times),1));position=np.tile(raw['object_positions'][frame],(len(times),1))
    wxyz=np.tile(raw['object_wxyz'][frame],(len(times),1))
    points=[SurfacePoint(**{k:np.array(v) if isinstance(v,list) else v for k,v in p.items()}) for p in candidate['contacts']]
    forces=np.tile(np.array(candidate['static_hold_screen']['forces']),(len(times),1,1))
    normals=np.tile(-np.array([p.object_outward for p in points]),(len(times),1,1))
    velocity=None
    if motion_from is not None:
        reference=read_npz(motion_from/'reference.npz');window=slice(frame,481)
        q=reference['q'][window];q0=q[0];times=reference['timestamps'][window];times=times-times[0]
        position=reference['object_positions'][window];wxyz=reference['object_wxyz'][window]
        forces=reference['force_ff'][window];normals=reference['inward_normals_local'][window]
        velocity=reference['command_velocity'][window]
    output.mkdir(parents=True); report=dict(kind='counterfactual_already_acquired_hold_not_episode_success',
        candidates=str(candidates.resolve()),initialization='optimized static grasp; artificial for diagnosis only',
        motion_from=str(motion_from.resolve()) if motion_from else None,
        physical_object=True,object_assistance=False,results={},provenance=snapshot_sources(output,[Path(__file__),
            Path('correction/src/dynamics/supported_robot.py'),Path('correction/src/dynamics/contact_controller.py')]))
    for kp in (2.,5.,20.):
        config=ContactControllerConfig(interface='torque',feedback_gain=.25,friction=saved['contract']['contact']['friction'])
        result=scene.rollout(q,position,wxyz,times,force_ff=forces,force_points=points,contact_controller=config,
            contact_normals_local=normals,contact_active=np.ones((len(times),len(points)),bool),
            initial_q=q0,initial_velocity=np.zeros(len(scene.movable)) if velocity is None else velocity[0],
            command_velocity=velocity,kp_arm=300.,kp_hand=kp,inertia_ff=True)
        np.savez_compressed(output/f'hand_kp_{kp:g}.npz',**result)
        report['results'][str(kp)]=dict(max_position_mm=float(1000*result['position_error'].max()),
            final_position_mm=float(1000*result['position_error'][-1]),
            final_rotation_deg=float(np.rad2deg(result['rotation_error'][-1])),
            final_contacts=result['contacts_per_hand'][-1].tolist(),
            actual_object_force=float(result['max_object_applied_force']))
        print(kp,report['results'][str(kp)],flush=True)
    report.update(wall_seconds=perf_counter()-start,cost=process_cost())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--motion-from',type=Path)
    args=parser.parse_args();run(args.candidates,args.output,args.motion_from)
