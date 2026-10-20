"""M0-M3: inspect saved failure, extract evidence and screen real G1 surfaces.

Writes fresh, reproducible outputs. Static candidates are NOT replay successes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from correction.src.preprocessing.convex_object import decompose
from correction.src.preprocessing.interaction_evidence import evidence_report, regions
from correction.src.dynamics.execution_contract import ContactExecutionContract, scene_contract
from correction.src.dynamics.supported_robot import SupportedRobotScene
from correction.src.optimization.surface_hypotheses import SurfaceLibrary, solve_geometry, contact_kinematics
from correction.src.optimization.wrench_screen import screen_wrench
from correction.src.optimization.strategy_screen import static_hold_screen
from correction.src.pass_traj.experiment_record import snapshot_sources


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def run(source, output, frames, per_solve_seconds, patch_sliding=False, planning_friction=.45, families=None,
        recharts=0, prune_unloaded=False, non_contact_clearance=0.):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    started = perf_counter()
    ref = read(source/'reference.npz')
    source_report = json.loads((source/'report.json').read_text())
    raw_path = Path(source_report['input_directory'])/'retargeted/contact_aware.npz'
    raw = read(raw_path)
    meta = json.loads(str(raw['metadata_json']))
    contract = ContactExecutionContract(friction=planning_friction,physics_friction=source_report['physics']['friction'])
    rotation = Rotation.from_quat(ref['object_wxyz'], scalar_first=True).as_matrix()
    inward_world = np.einsum('tij,tkj->tki', rotation, -raw['contact_normals_local'])
    force_audit = ContactExecutionContract().force_audit(ref['force_ff'], inward_world, ref['contact_active'])
    kin = G1Kinematics()
    parts = decompose(str(ref['object_mesh']), Path(source_report['input_directory'])/'collision')
    scene = SupportedRobotScene(kin, ref['bounds'], meta['source']['support_surface'], ref['initial_q'],
        mass=source_report['physics']['mass_kg'], friction=contract.physics_friction,
        dt=source_report['physics']['timestep_s'], move_waist=True, moving_fixture=True, object_parts=parts, floor_z=float(ref['floor_z']),
        object_inertial=source_report['physics'].get('object_inertial'))
    library = SurfaceLibrary(scene)
    output.mkdir(parents=True)
    paths = [Path(__file__), *[Path('correction/src')/p for p in
        ('dynamics/execution_contract.py', 'dynamics/supported_robot.py', 'dynamics/robot_scene.py',
         'preprocessing/interaction_evidence.py', 'preprocessing/convex_object.py',
         'optimization/surface_hypotheses.py', 'optimization/patch_geometry.py', 'optimization/wrench_screen.py',
         'dynamics/collision_distance.py',
         'optimization/strategy_screen.py')]]
    provenance = snapshot_sources(output, paths)
    report = dict(scope='M0-M3 static geometry and capacity only; no dynamic success claim',
        source=str(source.resolve()), input_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        source_reference_sha256=hashlib.sha256((source/'reference.npz').read_bytes()).hexdigest(),
        contract=scene_contract(scene, contract), saved_force_audit=force_audit,
        evidence=evidence_report(raw), frames=frames, per_solve_seconds=per_solve_seconds, patch_sliding=patch_sliding,
        candidates=[], ready_for_training=False, physical_replay_run=False, provenance=provenance,
        families=families or ['observed_six', 'coherent_four', 'support_six'], recharts=recharts,
        prune_unloaded=prune_unloaded, non_contact_clearance_m=non_contact_clearance, status='running')
    write(output/'report.json', report)
    print('Model built; force audit: '+json.dumps(force_audit), flush=True)
    q_solutions = []
    families = families or ['observed_six', 'coherent_four', 'support_six']
    for frame in frames:
        region = {r.side: np.asarray(r.outward_object) for r in regions(raw, max(0, frame-20), min(len(raw['q'])-1, frame+20))}
        for family in families:
            before = perf_counter()
            result = dict(frame=frame, family=family, time_s=float(ref['timestamps'][frame]))
            try:
                points = library.hypothesis(raw, frame, family, region)
                if patch_sliding:
                    from correction.src.optimization.patch_geometry import solve_patch_geometry
                    q, points, geometry = solve_patch_geometry(scene, points, raw['q'][frame],
                        raw['object_positions'][frame], raw['object_wxyz'][frame], seconds=per_solve_seconds,
                        non_contact_clearance_m=non_contact_clearance)
                    attempts = [dict(geometry)]
                    for iteration in range(recharts):
                        if perf_counter()-before >= per_solve_seconds: break
                        points = library.rechart(points, raw['object_positions'][frame], raw['object_wxyz'][frame])
                        q, points, geometry = solve_patch_geometry(scene, points, raw['q'][frame],
                            raw['object_positions'][frame], raw['object_wxyz'][frame], initial_q=q,
                            non_contact_clearance_m=non_contact_clearance,
                            seconds=max(.01, per_solve_seconds-(perf_counter()-before)))
                        attempts.append(dict(geometry))
                    result['chart_attempts'] = attempts
                else:
                    q, geometry = solve_geometry(scene, points, raw['q'][frame], raw['object_positions'][frame],
                        raw['object_wxyz'][frame], seconds=per_solve_seconds)
                screen, support_gap = static_hold_screen(scene, library, points, float(ref['floor_z']), contract)
                if prune_unloaded and patch_sliding and screen['feasible'] and not screen['floor_contacts_included']:
                    keep = np.flatnonzero(np.linalg.norm(screen['forces'], axis=1) >= .2)
                    if 2 <= len(keep) < len(points) and perf_counter()-before < per_solve_seconds:
                        result['force_selected_contact_indices'] = keep.tolist()
                        result['before_force_selection'] = dict(geometry=geometry, screen=screen)
                        points = [points[i] for i in keep]
                        points = library.rechart(points, raw['object_positions'][frame], raw['object_wxyz'][frame])
                        q, points, geometry = solve_patch_geometry(scene, points, raw['q'][frame],
                            raw['object_positions'][frame], raw['object_wxyz'][frame], initial_q=q,
                            non_contact_clearance_m=non_contact_clearance,
                            seconds=max(.01, per_solve_seconds-(perf_counter()-before)))
                        screen, support_gap = static_hold_screen(scene, library, points, float(ref['floor_z']), contract)
                geometry_ok = (geometry['max_gap_m'] <= contract.contact_gap_tolerance_m and
                    geometry['max_robot_penetration_m'] <= contract.penetration_tolerance_m and
                    (geometry.get('max_normal_cone_gap_m', np.inf) <= .001 if patch_sliding else
                     geometry['min_normal_cosine'] >= contract.normal_cosine_min) and
                    geometry.get('patch_neighborhood_satisfied', True) and
                    geometry.get('evidence_region_satisfied', True) and
                    all(pair['distance_m']>=non_contact_clearance-.0001 for pair in geometry.get('unwanted_near_pairs',[])))
                result.update(geometry=geometry, geometry_accepted=geometry_ok,
                              static_hold_screen=screen, support_gap_m=support_gap,
                              candidate_for_acquisition=bool(geometry_ok and screen['feasible']),
                              contacts=[{key: value.tolist() if isinstance(value, np.ndarray) else value
                                         for key, value in asdict(p).items()} for p in points])
                q_solutions.append(q)
                result['solution_index'] = len(q_solutions)-1
            except (ValueError, KeyError) as error:
                result.update(error=str(error), candidate_for_acquisition=False)
            result['wall_seconds'] = perf_counter()-before
            report['candidates'].append(result)
            report['wall_seconds'] = perf_counter()-started
            write(output/'report.json', report)
            np.savez_compressed(output/'geometry_candidates.npz', q=np.asarray(q_solutions))
            print(json.dumps({k: v for k, v in result.items() if k != 'contacts'}), flush=True)
    report['accepted_static_candidates'] = sum(r['candidate_for_acquisition'] for r in report['candidates'])
    report['status'] = 'complete'
    report['wall_seconds'] = perf_counter()-started
    write(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('correction/outputs/g1_controller_full_20261019_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[240, 270, 300])
    parser.add_argument('--per-solve-seconds', type=float, default=12.)
    parser.add_argument('--patch-sliding', action='store_true')
    parser.add_argument('--planning-friction', type=float, default=.45)
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--recharts', type=int, default=0)
    parser.add_argument('--prune-unloaded', action='store_true')
    parser.add_argument('--non-contact-clearance', type=float, default=0.)
    args = parser.parse_args()
    run(args.source, args.output, args.frames, args.per_solve_seconds, args.patch_sliding, args.planning_friction,
        args.families, args.recharts, args.prune_unloaded, args.non_contact_clearance)
