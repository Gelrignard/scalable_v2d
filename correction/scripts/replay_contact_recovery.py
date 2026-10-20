"""Complete, uninterrupted observed-state recovery experiment with provenance."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.contact_recovery import RecoverySettings
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def run(args):
    if args.output.exists(): raise ValueError('Use a fresh output directory')
    start=perf_counter();cpu=process_cost()['cpu_seconds']
    prior=json.loads((args.source/'report.json').read_text())
    ref=read_npz(args.source/'reference.npz')
    _,_,_,scene=load_strategy_scene(prior['input_directory'])
    points=[SurfacePoint(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in p.items()}) for p in prior['contacts']]
    args.output.mkdir(parents=True)
    cfg=RecoverySettings(progress_path=str(args.output/'progress.json'))
    # Force guidance is enabled only by actual contact. Geometric approach is
    # the recovery planner's responsibility, not a force applied through air.
    control=ContactControllerConfig(**dict(prior['controller'],near_surface_m=0.,engagement_seconds=.02))
    report={k:prior[k] for k in ('input_directory','physics','contacts','kp_hand','inertia_ff_scale')}
    report.update(status='running',source=str(args.source.resolve()),controller=asdict(control),
        recovery=asdict(cfg),cases={},ready_for_training=False,object_assistance=False,resets_during_episode=0,
        source_reference_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        initialization_protocol='same full episode pre-approach rest state; declared initial object offset if nonzero',
        initial_object_offset_m=args.object_offset,scope='Observed-state staged recovery; free object and dynamic robot with externally supported pelvis',
        provenance=snapshot_sources(args.output,[Path(__file__),*[Path('correction/src')/p for p in (
            'optimization/contact_recovery.py','optimization/delta_command.py','dynamics/observed_contact.py',
            'dynamics/contact_controller.py','dynamics/supported_robot.py','dynamics/execution_step.py',
            'optimization/contact_mpc.py','optimization/surface_hypotheses.py')]]))
    def write(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    write();np.savez_compressed(args.output/'reference.npz',**ref)
    try:
        state=None
        if any(args.object_offset):
            state=dict(position=ref['object_positions'][0]+args.object_offset,wxyz=ref['object_wxyz'][0],
                       linear_velocity=np.zeros(3),angular_velocity_world=np.zeros(3))
        result=scene.rollout(ref['q'],ref['object_positions'],ref['object_wxyz'],ref['timestamps'],
            force_ff=ref['force_ff'],force_points=points,contact_controller=control,
            contact_normals_local=ref['inward_normals_local'],contact_active=ref['contact_active'],
            point_local_trajectory=ref['robot_local_points'],object_local_trajectory=ref['object_local_points'],
            command_velocity=ref['command_velocity'],kp_arm=300.,kp_hand=prior['kp_hand'],
            inertia_ff=True,inertia_ff_scale=prior['inertia_ff_scale'],smooth_fixture=True,
            initial_q=ref['initial_q'],initial_velocity=ref['initial_velocity'],initial_object_state=state,
            contact_body_audit=True,recovery_config=None if args.baseline else cfg)
        if len(result['q'])!=len(ref['q']) or result['max_object_applied_force']!=0:
            raise AssertionError('Complete unassisted episode required')
        np.savez_compressed(args.output/'mechanics.npz',**result)
        report.update(status='complete',frames=len(ref['q']),duration_s=float(ref['timestamps'][-1]),
                      cases=dict(mechanics=assessed(result,ref['timestamps'])))
        if not args.baseline:
            report['prediction_error']=float(result['recovery_max_prediction_state_error'])
            report['updates']=len(json.loads(str(result['recovery_log_json'])))
        print(json.dumps(report['cases'],indent=2),flush=True)
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds']=perf_counter()-start
        report['cost']=process_cost();report['cost']['cpu_seconds']-=cpu;write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--object-offset',type=float,nargs=3,default=[0.,0.,0.])
    parser.add_argument('--baseline',action='store_true',help='Matched controller settings without recovery')
    run(parser.parse_args())
