"""Read-only constraint isolation for a saved unsupported grasp.

Relaxed cases are counterfactual diagnostics. No trajectory, simulation,
controller configuration or accepted-candidate export is produced.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.optimization.surface_hypotheses import SurfacePoint, contact_kinematics
from correction.src.optimization.wrench_screen import screen_wrench
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def run(args):
    if args.output.exists():
        raise ValueError('Preserve existing diagnostics')
    started = perf_counter(); cpu = process_cost()['cpu_seconds']
    saved = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    if len(saved['candidates']) != 1:
        raise ValueError('Diagnostic expects one declared grasp candidate')
    candidate = saved['candidates'][0]; frame = candidate['frame']
    if candidate['static_hold_screen']['floor_contacts_included']:
        raise ValueError('This diagnostic isolates unsupported contact capacity')
    _, raw, _, scene = load_strategy_scene(saved['source'])
    q = read_npz(args.source/'geometry_candidates.npz')['q'][candidate['solution_index']]
    scene.set_state(q, raw['object_positions'][frame], raw['object_wxyz'][frame])
    points = [SurfacePoint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})
              for p in candidate['contacts']]
    position, _, jacobian, _ = contact_kinematics(scene, points)
    inward = Rotation.from_quat(raw['object_wxyz'][frame], scalar_first=True).apply(
        -np.array([p.object_outward for p in points]))
    wanted = np.r_[-scene.model.body_mass[scene.book_body]*scene.model.opt.gravity, np.zeros(3)]
    bias = scene.data.qfrc_bias[scene.active_v].copy()
    friction = saved['contract']['contact']['friction']
    normal_cap = saved['contract']['contact']['normal_limit_n']
    options = [
        ('declared', friction, normal_cap, 1.),
        ('friction_1_only', 1., normal_cap, 1.),
        ('normal_60_only', friction, 60., 1.),
        ('torque_100x_only', friction, normal_cap, 100.),
        ('friction_1_normal_60', 1., 60., 1.),
        ('normal_60_torque_100x', friction, 60., 100.),
        ('all_three_relaxed', 1., 60., 100.)]
    args.output.mkdir(parents=True)
    report = dict(status='running', scope='Counterfactual LP diagnosis only; no physical trial or trajectory export',
        source=str(args.source.resolve()), configuration=dict(frame=frame, cases=options, seconds_per_lp=1.),
        geometry_unchanged=True, environment_contacts=False, diagnostic_only=True,
        physical_success=False, ready_for_training=False, cases={},
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            args.source/'report.json', args.source/'geometry_candidates.npz']},
        provenance=snapshot_sources(args.output, [Path(__file__),
            Path('correction/src/optimization/wrench_screen.py'), Path('correction/src/optimization/surface_hypotheses.py')]))
    def write():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    write()
    try:
        for name, mu, force_cap, torque_scale in options:
            row = screen_wrench(position, inward, scene.data.xipos[scene.book_body], wanted,
                jacobian, bias, torque_scale*scene.caps, friction=mu, normal_caps=force_cap, seconds=1.)
            force = np.asarray(row['forces'])
            normal = np.sum(force*inward, axis=1)
            torque = bias+np.einsum('kij,ki->j', jacobian, force)
            row.update(friction=mu, normal_cap_n=force_cap, actuator_multiplier=torque_scale,
                       max_normal_force_n=float(normal.max()),
                       torque_fraction_of_real_caps=float(np.max(abs(torque)/scene.caps)),
                       physical_success=False, diagnostic_only=True)
            # Numerical force solutions stay in the diagnostic report only.
            report['cases'][name] = row; write()
        np.testing.assert_allclose(report['cases']['declared']['force_residual_n'],
            candidate['static_hold_screen']['force_residual_n'], atol=1e-4, rtol=0)
        report.update(status='complete', reproduced_declared_failure=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_seconds'] = perf_counter()-started
        report['cost'] = process_cost(); report['cost']['cpu_seconds'] -= cpu
        write()
        print(json.dumps({name: {key: row[key] for key in ['feasible', 'force_residual_n', 'moment_residual_nm',
            'max_normal_force_n', 'torque_fraction_of_real_caps', 'seconds']}
            for name, row in report['cases'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
