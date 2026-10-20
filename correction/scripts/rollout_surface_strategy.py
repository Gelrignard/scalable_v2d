"""Full-episode physics for a newly selected real-surface contact strategy.

Original / geometry / geometry+force use one controller and initial state.
Only first-phase correction is implemented here; later failures remain visible.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.surface_reference import build_reference
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources,process_cost
from correction.scripts.compare_g1_controllers import assessed


def run(candidates, output, cases, approach=False, object_feedback=False, kp_hand=20.,rigid_grasp=False,follow_only=False,reference_from=None,smooth_fixture=False,normal_follow=False,fixture_window=0,candidate_family=None,force_profile=False,preload=2.,release=False,wrench_residual=False):
    if output.exists(): raise ValueError('Use a fresh output directory')
    start=perf_counter()
    initial_cpu=process_cost()['cpu_seconds']
    saved=json.loads((candidates/'report.json').read_text())
    accepted=[c for c in saved['candidates'] if c['candidate_for_acquisition'] and
              not c['static_hold_screen']['floor_contacts_included'] and
              c['geometry'].get('evidence_region_satisfied')]
    if candidate_family:accepted=[c for c in accepted if c['family']==candidate_family]
    if not accepted: raise ValueError('No accepted unsupported, evidence-bounded static grasp')
    candidate=min(accepted,key=lambda c:c['static_hold_screen']['max_torque_fraction'])
    ref, raw, source_report, scene=load_strategy_scene(saved['source'])
    original_root=raw['q'][:,:7].copy(); fixture_report=None
    if fixture_window:
        from correction.src.preprocessing.supported_fixture import regularize_fixture
        raw['q'],fixture_report=regularize_fixture(raw['q'],fixture_window)
    points=[SurfacePoint(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in p.items()})
            for p in candidate['contacts']]
    grasp_q=read_npz(candidates/'geometry_candidates.npz')['q'][candidate['solution_index']]
    output.mkdir(parents=True)
    provenance=snapshot_sources(output,[Path(__file__), *[Path('correction/src')/p for p in
        ('optimization/surface_reference.py','optimization/surface_hypotheses.py','optimization/rigid_grasp.py','optimization/actuator_force_profile.py',
         'dynamics/supported_robot.py','dynamics/robot_scene.py','dynamics/contact_controller.py','dynamics/object_relative_controller.py','dynamics/wrench_residual_controller.py',
         'preprocessing/strategy_bundle.py','preprocessing/supported_fixture.py')]])
    cached_profile=None
    if reference_from is None:
        command, command_report=build_reference(scene,raw,points,grasp_q,
            np.asarray(candidate['static_hold_screen']['forces']),candidate['frame'],approach=approach,rigid_grasp=rigid_grasp,release=release)
    else:
        prior=json.loads((reference_from/'report.json').read_text())
        if (Path(prior['candidates']).resolve()!=candidates.resolve() or prior['command']['approach_enabled']!=approach or
            prior['command']['rigid_grasp']!=rigid_grasp or prior.get('fixture_regularization')!=fixture_report or
            prior['contacts']!=candidate['contacts'] or prior['command'].get('release_enabled',False)!=release):
            raise ValueError('Cached command must use identical contact candidate and path mode')
        archive=read_npz(reference_from/'reference.npz')
        command={k:archive[k] for k in ('q','command_velocity','force_ff','contact_active','inward_normals_local','timestamps')}
        command_report=prior['command']
        cached_profile=prior.get('force_profile')
    profile_report=cached_profile
    if force_profile:
        from correction.src.optimization.actuator_force_profile import allocate_profile
        magnitude=np.linalg.norm(command['force_ff'].reshape(len(raw['q']),-1),axis=1)
        blend=magnitude/max(magnitude.max(),1e-12)
        command['force_ff'],profile_report=allocate_profile(scene,points,command['q'],raw['object_positions'],raw['object_wxyz'],
            command['timestamps'],blend,friction=saved['contract']['contact']['friction'],preload=preload)
    config=ContactControllerConfig(interface='torque', feedback_gain=.25,
                                    friction=saved['contract']['contact']['friction'])
    from correction.src.dynamics.object_relative_controller import ObjectFeedbackConfig
    object_config=(ObjectFeedbackConfig(allocate_wrench=not (follow_only or normal_follow),
        normal_follow_only=normal_follow,preserve_finger_shape=normal_follow) if object_feedback or follow_only or normal_follow else None)
    from correction.src.dynamics.wrench_residual_controller import WrenchResidualConfig
    residual_config=WrenchResidualConfig() if wrench_residual else None
    report=dict(scope='Full physical replay, first-phase fixed-contact candidate; not final horizon solver',
        status='running', candidates=str(candidates.resolve()), input_directory=saved['source'],
        candidate_family=candidate['family'],candidate_solution_index=candidate['solution_index'],candidate_frame=candidate['frame'],
        reference_cache=str(reference_from.resolve()) if reference_from else None,
        reference_cache_sha256=hashlib.sha256((reference_from/'reference.npz').read_bytes()).hexdigest() if reference_from else None,
        source_reference_sha256=hashlib.sha256((Path(saved['source'])/'reference.npz').read_bytes()).hexdigest(),
        frames=len(raw['q']), duration_s=float(command['timestamps'][-1]),
        intended_object_reference='unchanged input observations', command=command_report,
        controller=asdict(config), object_feedback=asdict(object_config) if object_config else None,
        gains=dict(kp_arm=300.,kp_hand=kp_hand,inertia_ff=True),smooth_fixture=smooth_fixture,
        fixture_regularization=fixture_report,
        force_profile=profile_report,
        wrench_residual=asdict(residual_config) if residual_config else None,
        physics=source_report['physics'], contacts=[asdict(p) for p in points],
        resets_during_episode=0, object_assistance=False, cases={}, provenance=provenance, ready_for_training=False)

    def write():
        (output/'report.json').write_text(json.dumps(report,indent=2,
            default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)+'\n')
    write()
    np.savez_compressed(output/'reference.npz',**command, object_positions=raw['object_positions'],
        object_wxyz=raw['object_wxyz'], baseline_command_q=raw['q'], initial_q=ref['initial_q'],
        initial_velocity=ref['initial_velocity'], object_mesh=ref['object_mesh'], bounds=ref['bounds'],
        floor_z=ref['floor_z'], frame_ids=np.arange(len(raw['q'])),original_input_root=original_root)
    initial_state=None
    for name in cases:
        begin=perf_counter(); print(f'Running full episode: {name}',flush=True)
        q=raw['q'] if name=='original' else command['q']
        forces=command['force_ff'] if name=='mechanics' else np.zeros_like(command['force_ff'])
        active=command['contact_active'] if name=='mechanics' else np.zeros_like(command['contact_active'])
        result=scene.rollout(q,raw['object_positions'],raw['object_wxyz'],command['timestamps'],
            force_ff=forces,force_points=points,contact_controller=config,
            contact_normals_local=command['inward_normals_local'],contact_active=active,
            command_velocity=None if name=='original' else command['command_velocity'],
            kp_arm=300.,kp_hand=kp_hand,inertia_ff=True,object_feedback=object_config,smooth_fixture=smooth_fixture,wrench_residual=residual_config,
            initial_q=ref['initial_q'],initial_velocity=ref['initial_velocity'])
        state=np.r_[result['q'][0],result['joint_velocities'][0],result['object_positions'][0],result['object_wxyz'][0]]
        if initial_state is None: initial_state=state
        np.testing.assert_array_equal(state,initial_state)
        if len(result['q'])!=len(raw['q']) or result['max_object_applied_force']!=0:
            raise AssertionError('Uninterrupted full episode and no object assistance required')
        np.savez_compressed(output/f'{name}.npz',**result)
        summary=assessed(result,command['timestamps'])
        if 'controller_touching' in result:
            left=np.array(['left_' in scene.model.body(p.body).name for p in points])
            touching=result['controller_touching']
            summary['frames_with_bilateral_selected_contacts']=int((np.any(touching[:,left],axis=1)&np.any(touching[:,~left],axis=1)).sum())
        summary['legacy_contact_count_scope']='contacts_per_hand includes any left/right robot link, including elbows; not verified fingertip contact'
        first,last=command_report['first_frame'],command_report['last_frame']
        summary.update(first_phase_rms_position_mm=float(1000*np.sqrt(np.mean(result['position_error'][first:last+1]**2))),
            first_phase_max_position_mm=float(1000*result['position_error'][first:last+1].max()),
            sampled_frames_with_contact=int(np.any(result['contacts_per_hand']>0,axis=1).sum()),
            wall_seconds=perf_counter()-begin)
        report['cases'][name]=summary;write();print(json.dumps(summary),flush=True)
    report.update(status='complete',initial_state_identical=True if len(cases)>1 else None,wall_seconds=perf_counter()-start)
    report['cost']=process_cost();report['cost']['cpu_seconds']-=initial_cpu
    write();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=['original','geometry','mechanics'],default=['original','geometry','mechanics'])
    parser.add_argument('--approach',action='store_true')
    parser.add_argument('--object-feedback',action='store_true')
    parser.add_argument('--kp-hand',type=float,default=20.)
    parser.add_argument('--rigid-grasp',action='store_true')
    parser.add_argument('--follow-only',action='store_true')
    parser.add_argument('--reference-from',type=Path)
    parser.add_argument('--smooth-fixture',action='store_true')
    parser.add_argument('--normal-follow',action='store_true')
    parser.add_argument('--fixture-window',type=int,default=0)
    parser.add_argument('--candidate-family')
    parser.add_argument('--force-profile',action='store_true');parser.add_argument('--preload',type=float,default=2.)
    parser.add_argument('--release',action='store_true')
    parser.add_argument('--wrench-residual',action='store_true')
    args=parser.parse_args();run(args.candidates,args.output,args.cases,args.approach,args.object_feedback,args.kp_hand,args.rigid_grasp,args.follow_only,args.reference_from,args.smooth_fixture,args.normal_follow,args.fixture_window,args.candidate_family,args.force_profile,args.preload,args.release,args.wrench_residual)
