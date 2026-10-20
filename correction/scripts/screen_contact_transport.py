"""Reject static grasps that cannot maintain geometry across a contact phase."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.preprocessing.supported_fixture import regularize_fixture
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.optimization.rigid_grasp import RigidGrasp
from correction.src.optimization.surface_reference import smoothstep
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists(): raise ValueError('Use a fresh output directory')
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.candidates/'report.json').read_text())
    _, raw, _, scene = load_strategy_scene(saved['source'])
    raw['q'], fixture = regularize_fixture(raw['q'], 21)
    shift = np.zeros(3)
    if args.shift_from:
        result = read_npz(args.shift_from/'mechanics.npz')
        shift = result['object_positions'][600]-raw['object_positions'][600]
    args.output.mkdir(parents=True)
    report = dict(status='running', source=str(args.candidates.resolve()), fixture=fixture,
        frame_samples=args.frames, second_translation_m=shift.tolist(),
        arm_neighborhood_rad=args.arm_neighborhood,
        source_snapshot=snapshot_sources(args.output, [Path(__file__),
            Path('correction/src/optimization/rigid_grasp.py')]),
        candidates=[], scope='necessary sparse transport geometry check; no physical success claim')
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    write()
    configurations = read_npz(args.candidates/'geometry_candidates.npz')['q']
    for candidate in saved['candidates']:
        if not candidate['candidate_for_acquisition'] or candidate['static_hold_screen']['floor_contacts_included']:
            continue
        points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
                  for p in candidate['contacts']]
        grasp_q = configurations[candidate['solution_index']]
        solver = RigidGrasp(scene, points, grasp_q, raw['object_positions'][candidate['frame']],
                            raw['object_wxyz'][candidate['frame']])
        previous = grasp_q.copy(); rows = []
        for frame in args.frames:
            coefficient = 1-smoothstep((frame-660)/(725-660))
            target = raw['object_positions'][frame]+coefficient*shift
            previous, audit = solver.solve(raw['q'][frame], target, raw['object_wxyz'][frame],
                previous, arm_neighborhood=args.arm_neighborhood, seconds=.5)
            rows.append(dict(frame=frame, **audit))
        maximum = max(row['max_gap_m'] for row in rows)
        entry = dict(frame=candidate['frame'], family=candidate['family'],
            solution_index=candidate['solution_index'], rows=rows, max_contact_gap_m=maximum,
            transport_geometry_accepted=bool(maximum<=.003 and all(row['max_robot_penetration_m']<=.002 for row in rows)))
        report['candidates'].append(entry); write()
        print(json.dumps({k: v for k, v in entry.items() if k != 'rows'}), flush=True)
    report.update(status='complete', wall_seconds=perf_counter()-start, cost=process_cost())
    report['cost']['cpu_seconds'] -= cpu
    write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', nargs='+', type=int, default=[635, 650, 670, 690, 710, 725])
    parser.add_argument('--shift-from', type=Path)
    parser.add_argument('--arm-neighborhood', type=float, default=1.5)
    run(parser.parse_args())
