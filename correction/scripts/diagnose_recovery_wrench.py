"""Frozen-state force feasibility diagnostics; no integration or trajectory edits.

Compare the implemented soft allocation with minimum wrench residual using
the same constraints, and then without the approximate actuator rows. States
come from saved video-rate samples, not exact unsaved MPC replan boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation, Slerp

from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint
from correction.src.dynamics.observed_contact import ContactCharts, observe_patches, allocate_observed
from correction.src.dynamics.environment_contact import environment_contacts
from correction.src.dynamics.sparse_forces import cross_matrix
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def minimum_residual(operator, wanted, constraints, low, high):
    scale = np.array([10., 10., 10., 1., 1., 1.])
    a, b = operator/scale[:, None], wanted/scale
    problem = osqp.OSQP()
    problem.setup(P=sparse.csc_matrix(2*(a.T@a+1e-10*np.eye(a.shape[1]))),
                  q=-2*a.T@b, A=constraints, l=low, u=high, verbose=False,
                  eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, polishing=True)
    result = problem.solve(raise_error=False)
    out = dict(status=result.info.status)
    if result.info.status_val not in (1, 2) or result.x is None:
        return out
    violation = float(max(0., np.max(low-constraints@result.x), np.max(constraints@result.x-high)))
    residual = operator@result.x-wanted
    out.update(constraint_violation=violation, residual=residual.tolist(),
               normalized_residual_norm=float(np.linalg.norm(residual/scale)),
               forces=result.x.reshape(-1,3).tolist())
    if violation > 1e-4:
        raise AssertionError('Diagnostic solver violated the frozen constraints')
    return out


def run(args):
    if args.output.exists():
        raise ValueError('Use a new diagnostic directory')
    started = perf_counter(); cpu = process_cost()['cpu_seconds']
    report = json.loads((args.source/'report.json').read_text())
    ref, data = read_npz(args.source/'reference.npz'), read_npz(args.source/'mechanics.npz')
    _, _, _, scene = load_strategy_scene(report['input_directory'])
    points = [SurfacePoint(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in p.items()})
              for p in report['contacts']]
    charts = ContactCharts(scene, points)
    kp = np.where(scene.hand, report['kp_hand'], 300.)
    kd = np.where(scene.hand, .4, 8.)
    results = []
    real_solver = osqp.OSQP
    for frame in args.frames:
        scene.set_state(data['q'][frame], data['object_positions'][frame], data['object_wxyz'][frame])
        scene.data.qvel[:] = data['generalized_velocities'][frame]
        mujoco.mj_forward(scene.model, scene.data)
        chart = charts.get(data['recovery_robot_local'][frame], data['recovery_object_local'][frame],
                           data['recovery_inward'][frame])
        obs = observe_patches(scene, points, chart)
        support = environment_contacts(scene, 8)
        contacts = np.vstack([obs['positions'], np.array([p['position'] for p in support]).reshape(-1,3)])
        operator = np.column_stack([np.vstack([np.eye(3), cross_matrix(p-scene.data.xipos[scene.book_body])])
                                    for p in contacts])
        time = float(ref['timestamps'][frame])
        future = min(time+report['recovery']['horizon_s'], float(ref['timestamps'][-1]))
        desired_p = np.array([np.interp(future, ref['timestamps'], col) for col in ref['object_positions'].T])
        desired_r = Slerp(ref['timestamps'], Rotation.from_quat(ref['object_wxyz'], scalar_first=True))(future).as_quat(scalar_first=True)
        captured = {}

        class Capture:
            def __init__(self):
                self.delegate = real_solver()
            def setup(self, **kwargs):
                captured.update(kwargs)
                self.delegate.setup(**kwargs)
            def solve(self, **kwargs):
                return self.delegate.solve(**kwargs)

        with patch('correction.src.dynamics.observed_contact.osqp.OSQP', Capture):
            forces, audit = allocate_observed(scene, obs, data['recovery_active'][frame], desired_p, desired_r,
                data['recovery_command'][frame], data['recovery_velocity'][frame], kp, kd,
                data['recovery_force'][frame], report['recovery']['period_s'],
                normal_cap=report['controller']['normal_limit_n'], friction=report['controller']['friction'])
        all_forces = np.r_[forces.ravel(), np.array(audit['environment_forces']).ravel()]
        wanted = operator@all_forces-np.array(audit['wrench_residual'])
        constrained = minimum_residual(operator, wanted, captured['A'], captured['l'], captured['u'])
        n = len(scene.active_v)
        no_torque = minimum_residual(operator, wanted, captured['A'][:-n], captured['l'][:-n], captured['u'][:-n])
        if 'normalized_residual_norm' in constrained and 'normalized_residual_norm' in no_torque:
            assert no_torque['normalized_residual_norm'] <= constrained['normalized_residual_norm']+1e-4
        results.append(dict(frame=frame, time_s=time, wanted_wrench=wanted.tolist(),
            eligible=audit['eligible'], touching_reconstructed=obs['touching'].tolist(),
            touching_recorded=data['recovery_touching'][frame].tolist(),
            body_names=[scene.model.body(p.body).name for p in points],
            implemented_soft=audit, minimum_residual=constrained, without_actuator_rows=no_torque))
    args.output.mkdir(parents=True)
    result = dict(source=str(args.source), frames=results,
        scope='Frozen saved sample geometry with recomputed contacts. Not exact replan state, a new rollout, or a real-robot feasibility certificate.',
        limitations=['Same conservative friction cones and selected bodies as the prototype.',
                     'Instantaneous actuator estimate omits inertia and controller history.',
                     'Environment support is a local allocation; contact complementarity is not enforced.'],
        mechanics_sha256=hashlib.sha256((args.source/'mechanics.npz').read_bytes()).hexdigest(),
        provenance=snapshot_sources(args.output,[Path(__file__),Path('correction/src/dynamics/observed_contact.py')]),
        wall_seconds=perf_counter()-started, cost=process_cost())
    result['cost']['cpu_seconds'] -= cpu
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in results:
        print(json.dumps(dict(frame=row['frame'], time_s=row['time_s'], eligible=row['eligible'],
            soft=row['implemented_soft']['wrench_residual'], minimum=row['minimum_residual'],
            no_torque=row['without_actuator_rows']),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[266,293,320])
    run(parser.parse_args())
