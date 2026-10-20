"""Sensitivity of saved contact hypotheses to force caps and support removal.

These are counterfactual capacity screens, not changes to the physical scene
or evidence that a greater assumed force is actually delivered by the robot.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint, contact_kinematics
from correction.src.optimization.wrench_screen import screen_wrench
from correction.src.pass_traj.experiment_record import snapshot_sources


def run(candidates, output):
    if output.exists(): raise ValueError('Use a fresh output directory')
    start = perf_counter()
    saved = json.loads((candidates/'report.json').read_text())
    ref, raw, _, scene = load_strategy_scene(saved['source'])
    solutions = read_npz(candidates/'geometry_candidates.npz')['q']
    output.mkdir(parents=True)
    provenance = snapshot_sources(output, [Path(__file__),
        Path('correction/src/optimization/wrench_screen.py'),
        Path('correction/src/optimization/surface_hypotheses.py'),
        Path('correction/src/preprocessing/strategy_bundle.py')])
    rows = []
    for candidate in saved['candidates']:
        if not candidate.get('geometry_accepted'): continue
        f = candidate['frame']; q = solutions[candidate['solution_index']]
        points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
                  for p in candidate['contacts']]
        scene.set_state(q, raw['object_positions'][f], raw['object_wxyz'][f])
        position, _, jac, _ = contact_kinematics(scene, points)
        inward = Rotation.from_quat(raw['object_wxyz'][f], scalar_first=True).apply(-np.array([p.object_outward for p in points]))
        wanted = np.r_[-scene.model.body_mass[scene.book_body]*scene.model.opt.gravity, np.zeros(3)]
        for friction in (.45, .6, 1.):
            for cap in (15., 25.):
                for actuator_bounded in (True, False):
                    result = screen_wrench(position, inward, scene.data.xipos[scene.book_body], wanted, jac,
                        scene.data.qfrc_bias[scene.active_v], scene.caps if actuator_bounded else np.full(len(scene.caps), 1e6),
                        friction=friction, normal_caps=cap)
                    row = dict(frame=f, family=candidate['family'], floor_support=False,
                        friction=friction, normal_cap_n=cap, actuator_bounded=actuator_bounded, result=result)
                    rows.append(row)
                    print(f"frame={f} mu={friction} cap={cap} bounded={actuator_bounded}: force error={result['force_residual_n']:.3f} N, moment={result['moment_residual_nm']:.3f} Nm", flush=True)
    report = dict(scope='unsupported static hold capacity; hypothetical parameter sensitivity only',
        source=str(candidates.resolve()), actual_scene_friction=scene.friction, rows=rows,
        provenance=provenance, wall_seconds=perf_counter()-start, physical_replay_run=False, ready_for_training=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args();run(args.candidates,args.output)
