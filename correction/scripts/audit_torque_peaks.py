"""Explain saved actuator peaks at physics resolution without another rollout."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import load_strategy_scene, read_npz
from correction.src.pass_traj.experiment_record import process_cost, snapshot_sources


def run(args):
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    output = args.source/'torque_peak_audit.json'
    if output.exists():
        raise ValueError('Preserve the existing audit')
    report = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    data = read_npz(args.source/'mechanics.npz')
    reference = read_npz(args.source/'reference.npz')
    _, _, _, scene = load_strategy_scene(report['input_directory'])
    names = [scene.kin.joint_names[i] for i in data['movable_joint_indices']]
    terms = data['torque_peak_terms_nm']
    fraction = data['frame_peak_torque_fraction']
    valid = np.isfinite(data['torque_peak_time_seconds'])
    residual = np.abs(terms.sum(axis=-1))/data['torque_caps']-fraction
    np.testing.assert_allclose(residual[valid], 0., atol=1e-12)
    result = dict(source=str(args.source.resolve()),
        source_sha256=hashlib.sha256((args.source/'mechanics.npz').read_bytes()).hexdigest(),
        initialization_protocol=report.get('initialization_protocol'),
        max_decomposition_residual_fraction=float(np.max(abs(residual[valid]))),
        provenance=snapshot_sources(args.source/'peak_audit_provenance', [Path(__file__)]))
    if args.compare:
        old = read_npz(args.compare/'mechanics.npz')
        for key in ('q','object_positions','object_wxyz','torques','frame_peak_torque_fraction'):
            np.testing.assert_array_equal(old[key], data[key])
        result['instrumentation_unchanged_physics_vs'] = str(args.compare.resolve())

    def peak(a, b):
        flat = np.argmax(fraction[a:b+1])
        frame, joint = np.unravel_index(flat, fraction[a:b+1].shape)
        frame += a
        return dict(frame=int(frame), joint=names[joint],
            time_seconds=float(data['torque_peak_time_seconds'][frame, joint]),
            requested_torque_fraction=float(fraction[frame,joint]),
            cap_nm=float(data['torque_caps'][joint]),
            terms_nm=dict(zip(data['torque_peak_term_names'].tolist(), terms[frame,joint].tolist())),
            contact_step_nm=float(data['torque_peak_contact_torque_step_nm'][frame,joint]),
            force_enabled=data['torque_peak_force_enabled'][frame,joint].tolist(),
            touching=data['torque_peak_touching'][frame,joint].tolist(),
            gap_mm=(1000*data['torque_peak_contact_gap_m'][frame,joint]).tolist())

    result['peaks'] = {name: peak(a,b) for name,a,b in [
        ('full',0,len(fraction)-2), ('first',235,480), ('free',525,644),
        ('second',645,725), ('final',770,856)]}
    result['selected_frames'] = [peak(frame,frame) for frame in args.frames]
    result['joints'] = {}
    for label, values in [('actual',data['q']), ('command',reference['q'])]:
        movable = values[:, 7+data['movable_joint_indices']]
        step = np.diff(movable, axis=0)
        frame,joint = np.unravel_index(np.argmax(abs(step)), step.shape)
        result['joints'][label] = dict(from_frame=int(frame),to_frame=int(frame+1),
            joint=names[joint], delta_rad=float(step[frame,joint]),
            neighbouring_positions=movable[max(0,frame-2):frame+4,joint].tolist())
    result['wall_seconds'] = perf_counter()-start
    result['cost'] = process_cost(); result['cost']['cpu_seconds'] -= cpu
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--frames', type=int, nargs='+', default=[162,320,619,700])
    run(parser.parse_args())
