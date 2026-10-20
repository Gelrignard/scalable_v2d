"""Apply the staged surface/force/spline prototype to one new full episode.

Timing and candidates are chosen from input evidence before physical replay.
Rejected stages remain explicitly diagnostic; no fitted outcome selection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.preprocessing.supported_fixture import regularize_fixture
from correction.src.preprocessing.episode_phase import single_phase
from correction.src.preprocessing.interaction_evidence import regions
from correction.src.optimization.surface_hypotheses import SurfaceLibrary
from correction.src.optimization.patch_geometry import solve_patch_geometry
from correction.src.optimization.strategy_screen import static_hold_screen
from correction.src.optimization.surface_reference import build_reference, smoothstep
from correction.src.optimization.actuator_force_profile import allocate_profile
from correction.src.optimization.command_smoothing import refine_commands
from correction.src.dynamics.execution_contract import ContactExecutionContract, scene_contract
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost
from correction.scripts.compare_g1_controllers import assessed


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v:
        v.tolist() if isinstance(v, np.ndarray) else v.item())+'\n', encoding='utf-8')


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh experiment directory')
    started = perf_counter(); cpu = process_cost()['cpu_seconds']
    old = json.loads((args.source/'report.json').read_text())
    old_ref = read_npz(args.source/'reference.npz')
    args.output.mkdir(parents=True)
    profile = args.output/'profile'; profile.mkdir()
    physics = dict(mass_kg=1., friction=1., timestep_s=.001,
                   object_inertial=dict(com_local_m=[0., 0., 0.], diagonal_kg_m2=[.01]*3))
    write(profile/'report.json', dict(input_directory=str(args.source.resolve()),
          episode=old['episode'], physics=physics,
          scope='Same source-derived object defaults as preserved bin; not official-runtime parity'))
    np.savez_compressed(profile/'reference.npz', **old_ref)
    config = ContactControllerConfig(interface='torque', feedback_gain=.25, friction=.8,
                                    patch_distance_gate=True, near_surface_m=.025)
    report = dict(status='loading', episode=old['episode'], input_directory=str(profile.resolve()),
        physics=physics, controller=asdict(config), kp_hand=2., inertia_ff_scale=1.,
        source=str(args.source.resolve()), scope='Single-interval transfer of staged bin prototype',
        object_assistance=False, resets_during_episode=0, ready_for_training=False, cases={},
        configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        limitations=['Single rigid object; stroller wheels are not articulated',
          'One fixed grasp; no outcome-fitted phase/contact selection',
          'Quasistatic hand-force profile does not optimize sliding/rolling environment forces',
          'Supported pelvis; perfect Tier1 object reference; no learned whole-body controller'],
        source_sha256=hashlib.sha256((args.source/'retargeted/contact_aware.npz').read_bytes()).hexdigest(),
        provenance=snapshot_sources(args.output, [Path(__file__), *[
            Path('correction/src')/f for f in ['preprocessing/episode_phase.py',
            'optimization/surface_hypotheses.py','optimization/patch_geometry.py',
            'optimization/strategy_screen.py','optimization/surface_reference.py',
            'optimization/rigid_grasp.py','optimization/actuator_force_profile.py',
            'optimization/command_smoothing.py','dynamics/supported_robot.py',
            'dynamics/contact_controller.py']]]))
    def save():
        report['wall_seconds'] = perf_counter()-started
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        write(args.output/'report.json', report)
    save()
    try:
        ref, raw, _, scene = load_strategy_scene(profile)
        raw['q'], fixture = regularize_fixture(raw['q'], 21)
        phase, frames = single_phase(raw['contact_active'], raw['timestamps'])
        report.update(status='screening', frames=len(raw['q']),
            duration_s=float(raw['timestamps'][-1]-raw['timestamps'][0]),
            timing=asdict(phase), candidate_frames=frames, fixture_regularization=fixture,
            initialization_protocol='Original pre-approach posture and object state; robot at rest',
            contract=scene_contract(scene, ContactExecutionContract(friction=.8, physics_friction=1.)))
        save()
        library = SurfaceLibrary(scene)
        contract = ContactExecutionContract(friction=.8, physics_friction=1.)
        candidates = []; solutions = []; started_screen = perf_counter()
        for frame in frames:
            normal = {r.side:np.asarray(r.outward_object)
                      for r in regions(raw, max(0, frame-20), min(len(raw['q'])-1, frame+20))}
            for family in ['observed_six', 'coherent_four', 'support_any']:
                if perf_counter()-started_screen >= args.screen_seconds:
                    break
                before = perf_counter()
                row = dict(frame=frame, family=family)
                try:
                    points = library.hypothesis(raw, frame, family, normal)
                    q, points, geometry = solve_patch_geometry(scene, points, raw['q'][frame],
                        raw['object_positions'][frame], raw['object_wxyz'][frame],
                        seconds=min(8., max(.05,args.screen_seconds-(perf_counter()-started_screen))))
                    force, gap = static_hold_screen(scene, library, points, float(ref['floor_z']), contract)
                    valid = bool(geometry['max_gap_m'] <= .003 and
                        geometry['max_robot_penetration_m'] <= .002 and
                        geometry['max_normal_cone_gap_m'] <= .001 and
                        geometry['patch_neighborhood_satisfied'] and geometry['evidence_region_satisfied'])
                    row.update(geometry=geometry, static_hold_screen=force, support_gap_m=gap,
                        geometry_accepted=valid, candidate_for_acquisition=valid and force['feasible'],
                        contacts=[asdict(p) for p in points], solution_index=len(solutions))
                    # Select from input-only evidence, never physical outcome.
                    score = (not row['candidate_for_acquisition'], not valid,
                        1000*(geometry['max_gap_m']+5*geometry['max_robot_penetration_m'])+
                        100*geometry['max_normal_cone_gap_m'])
                    solutions.append((score,q,points,row))
                except (ValueError, KeyError) as error:
                    row.update(error=str(error),candidate_for_acquisition=False)
                row['wall_seconds'] = perf_counter()-before; candidates.append(row)
                report['candidates'] = candidates; save()
                print('screen',frame,family,row.get('candidate_for_acquisition'),flush=True)
        if not solutions:
            raise ValueError('No finite surface hypothesis; cannot construct a contact diagnostic')
        _, grasp_q, points, selected = min(solutions, key=lambda x:x[0])
        report.update(status='constructing',selected_candidate=dict(frame=selected['frame'],
            family=selected['family'],candidate_for_acquisition=selected['candidate_for_acquisition']),
            contacts=[asdict(p) for p in points],
            phase_contacts=[dict(frame=selected['frame'],timing=asdict(phase),contacts=[asdict(p) for p in points])])
        save()
        force = np.asarray(selected['static_hold_screen']['forces'])[:len(points)]
        command, geometry = build_reference(scene, raw, points, grasp_q, force, selected['frame'],
            approach=True, rigid_grasp=True, release=True, phase=phase)
        report['construction'] = geometry
        blend = smoothstep((np.arange(len(raw['q']))-phase.contact_start)/(phase.force_full-phase.contact_start))
        blend *= smoothstep((phase.force_end-np.arange(len(raw['q'])))/(phase.force_end-phase.contact_end))
        try:
            command['force_ff'], report['force_profile'] = allocate_profile(scene,points,command['q'],
                raw['object_positions'],raw['object_wxyz'],command['timestamps'],blend,friction=.8,preload=2.)
        except ValueError as error:
            command['force_ff'][:] = 0.
            report['force_profile'] = dict(solved=False,error=str(error),
                fallback='Zero explicit force guidance; geometry-only diagnostic')
        count = len(raw['q'])
        command.update(robot_local_points=np.tile([p.local for p in points],(count,1,1)),
            object_local_points=np.tile([p.object_local for p in points],(count,1,1)),
            object_positions=raw['object_positions'],object_wxyz=raw['object_wxyz'],
            baseline_command_q=raw['q'],initial_q=raw['q'][0].copy(),
            initial_velocity=np.zeros(len(scene.movable)),object_mesh=ref['object_mesh'],
            bounds=ref['bounds'],floor_z=ref['floor_z'],frame_ids=np.arange(count))
        np.savez_compressed(args.output/'before_spline.npz',**command)
        report['status']='spline_refinement'; save()
        fitted, audit = refine_commands(scene,command,points,seconds=args.spline_seconds,
            free_trust_rad=.12,continuous_bounds=True)
        report['optimization']=audit
        if fitted is not None:
            command=fitted
        else:
            report['diagnostic_fallback']='Spline rejected; replaying finite pre-spline geometric candidate'
        report['planning_accepted']=bool(selected['candidate_for_acquisition'] and fitted is not None and
                                        report['force_profile'].get('task_wrench_satisfied',False))
        if not report['planning_accepted'] and not args.replay_rejected:
            np.savez_compressed(args.output/'rejected_reference.npz',**command)
            report.update(status='rejected_before_replay',accepted=False,
                diagnostic_fallback='Planning gates failed; explicit --replay-rejected needed for diagnostic physics')
            return
        # Same physical initialization for both branches, with exact visible frame0.
        command['q'][0]=command['initial_q']
        command['command_velocity'][0]=0.
        np.savez_compressed(args.output/'reference.npz',**command)
        report['status']='replaying'; save()
        initial=None
        for case in ['original','mechanics']:
            print('full replay',case,count,flush=True)
            before=perf_counter()
            result=scene.rollout(command['baseline_command_q'] if case=='original' else command['q'],
                command['object_positions'],command['object_wxyz'],command['timestamps'],
                force_ff=command['force_ff'] if case=='mechanics' else np.zeros_like(command['force_ff']),
                force_points=points,contact_controller=config,contact_normals_local=command['inward_normals_local'],
                contact_active=command['contact_active'] if case=='mechanics' else np.zeros_like(command['contact_active']),
                point_local_trajectory=command['robot_local_points'],object_local_trajectory=command['object_local_points'],
                command_velocity=command['command_velocity'] if case=='mechanics' else None,
                kp_arm=300.,kp_hand=2.,inertia_ff=True,inertia_ff_scale=1.,smooth_fixture=True,
                initial_q=command['initial_q'],initial_velocity=command['initial_velocity'])
            state=np.r_[result['q'][0],result['joint_velocities'][0],result['object_positions'][0],result['object_wxyz'][0]]
            if initial is None: initial=state
            np.testing.assert_array_equal(initial,state)
            if len(result['q'])!=count or result['max_object_applied_force']!=0:
                raise AssertionError('Expected a complete unassisted replay')
            np.savez_compressed(args.output/f'{case}.npz',**result)
            report['cases'][case]=dict(assessed(result,command['timestamps']),wall_seconds=perf_counter()-before)
            save()
            print(case,json.dumps(report['cases'][case]),flush=True)
        report.update(status='complete',initial_state_identical=True,
            accepted=bool(report['planning_accepted'] and report['cases']['mechanics']['accepted_physics']))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--screen-seconds',type=float,default=75.)
    parser.add_argument('--spline-seconds',type=float,default=30.)
    parser.add_argument('--replay-rejected',action='store_true',
        help='Record full diagnostic physics even if planning gates reject; never enables training export')
    run(parser.parse_args())
