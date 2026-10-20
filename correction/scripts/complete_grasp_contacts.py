"""Bounded surface/force screening of an omitted opposing finger contact."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.dynamics.execution_contract import ContactExecutionContract, scene_contract
from correction.src.optimization.surface_hypotheses import SurfacePoint, SurfaceLibrary
from correction.src.optimization.contact_completion import extract_contact_patch
from correction.src.optimization.patch_geometry import solve_patch_geometry
from correction.src.optimization.strategy_screen import static_hold_screen
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    started = perf_counter(); cpu = process_cost()['cpu_seconds']
    prior = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    ref = read_npz(args.source/'reference.npz')
    executed = read_npz(args.execution/'mechanics.npz')
    executed_ref = read_npz(args.execution/'reference.npz')
    for key in ('q','object_positions','object_wxyz','force_ff','initial_q','initial_velocity'):
        np.testing.assert_array_equal(ref[key],executed_ref[key])
    if len(executed['q']) != len(ref['q']) or executed['max_object_applied_force'] != 0:
        raise ValueError('Candidate discovery needs a full unassisted source rollout')
    _, raw, _, scene = load_strategy_scene(prior['input_directory'])
    library = SurfaceLibrary(scene)
    contract = ContactExecutionContract(friction=prior['controller']['friction'],
                                        physics_friction=prior['physics']['friction'])
    base = [SurfacePoint(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in p.items()})
            for p in prior['phase_contacts'][0]['contacts']]
    if args.drop_unrealized_left_thumb:
        base = [p for p in base if scene.model.body(p.body).name != 'left_hand_thumb_2_link']
    args.output.mkdir(parents=True)
    report = dict(status='screening',source=prior['input_directory'],
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        command_source=str(args.source.resolve()), execution_source=str(args.execution.resolve()),
        command_sha256=hashlib.sha256((args.source/'reference.npz').read_bytes()).hexdigest(),
        execution_sha256=hashlib.sha256((args.execution/'mechanics.npz').read_bytes()).hexdigest(),
        contract=scene_contract(scene,contract), candidates=[], physical_success=False,
        provenance=snapshot_sources(args.output,[Path(__file__),
            Path('correction/src/optimization/contact_completion.py'),
            Path('correction/src/optimization/patch_geometry.py'),
            Path('correction/src/dynamics/collision_distance.py'),
            Path('correction/src/optimization/surface_hypotheses.py'),
            Path('correction/src/optimization/strategy_screen.py'),
            Path('correction/src/optimization/wrench_screen.py')]))
    solutions = []
    def write():
        report['wall_seconds'] = perf_counter()-started
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        (args.output/'report.json').write_text(json.dumps(report,indent=2,
            default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)+'\n',encoding='utf-8')
        np.savez_compressed(args.output/'geometry_candidates.npz',q=np.asarray(solutions))
    write()
    for frame in args.frames:
        before = perf_counter()
        family = 'completed_without_thumb' if args.drop_unrealized_left_thumb else 'completed_right_index'
        row = dict(frame=frame,family=family,candidate_for_acquisition=False)
        try:
            scene.set_state(executed['q'][frame],executed['object_positions'][frame],executed['object_wxyz'][frame])
            missing, extraction = extract_contact_patch(scene,library,'right_hand_index_1_link',
                raw['contact_anchors_local'][frame,3:6],source_channels=[3,4,5])
            row['extraction'] = extraction
            if missing.body in [p.body for p in base]:
                raise ValueError('Proposed contact already has a channel in the first phase')
            points = deepcopy(base)+[missing]
            initial = executed['q'][frame].copy(); initial[:7] = ref['q'][frame,:7]
            q,points,geometry = solve_patch_geometry(scene,points,ref['q'][frame],
                ref['object_positions'][frame],ref['object_wxyz'][frame],
                initial_q=initial,seconds=args.per_solve_seconds,non_contact_clearance_m=args.clearance,
                allow_patch_neighbours=args.patch_neighbours)
            screen,support_gap = static_hold_screen(scene,library,points,float(ref['floor_z']),contract,allow_floor=False)
            accepted = (geometry['max_gap_m']<=contract.contact_gap_tolerance_m
                and geometry['max_robot_penetration_m']<=contract.penetration_tolerance_m
                and geometry['max_normal_cone_gap_m']<=.001 and geometry['patch_neighborhood_satisfied']
                and geometry['evidence_region_satisfied']
                and all(p['distance_m']>=args.clearance-.0001 for p in geometry['unwanted_near_pairs']))
            row.update(geometry=geometry,geometry_accepted=bool(accepted),
                static_hold_screen=screen,support_gap_m=support_gap,
                contacts=[asdict(p) for p in points],solution_index=len(solutions),
                candidate_for_acquisition=bool(accepted and screen['feasible']))
            solutions.append(q)
        except ValueError as error:
            row['error'] = str(error)
        row['wall_seconds'] = perf_counter()-before
        report['candidates'].append(row);write()
        print(json.dumps({k:v for k,v in row.items() if k!='contacts'},
            default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v),flush=True)
    report.update(status='complete',accepted_static_candidates=sum(
        c['candidate_for_acquisition'] for c in report['candidates']))
    write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[300,350,400,430])
    parser.add_argument('--per-solve-seconds',type=float,default=3.)
    parser.add_argument('--clearance',type=float,default=.002)
    parser.add_argument('--patch-neighbours',action='store_true')
    parser.add_argument('--drop-unrealized-left-thumb',action='store_true')
    run(parser.parse_args())
