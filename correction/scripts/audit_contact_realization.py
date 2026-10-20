"""Partition actual object impulses and compare independent-pad acquisition."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from correction.src.preprocessing.strategy_bundle import read_npz
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def digest(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(args):
    start = perf_counter(); cpu = process_cost()['cpu_seconds']
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.json'):
        raise ValueError('Audit output must be a local JSON filename')
    output = args.robot/args.output_name
    if output.exists():
        raise ValueError('Preserve existing audit')
    robot = read_npz(args.robot/'mechanics.npz')
    reference = read_npz(args.robot/'reference.npz')
    result = dict(robot=digest(args.robot/'mechanics.npz'), pads={},
        scope=('Robot impulses integrate every1ms solved contact. Selected patches require the same link, '
               '30mm position radius and normal cosine>=0.5; active patches additionally require the planned mode. '
               'Pad forces below are video-rate means, not an equivalent impulse measurement. '
               'Attribution describes the executed rollout; it is not a causal ablation removing other contacts.'),
        provenance=snapshot_sources(args.robot/(output.stem+'_provenance'), [Path(__file__)]))
    if args.compare:
        old = read_npz(args.compare/'mechanics.npz')
        for key in ('q','object_positions','object_wxyz','torques','frame_peak_torque_fraction'):
            np.testing.assert_array_equal(robot[key],old[key])
        result['instrumentation_unchanged_physics_vs'] = str(args.compare.resolve())
    by_body = robot['interval_robot_body_wrench_impulse_world']
    all_robot = robot['interval_robot_contact_impulse_world']
    residual = by_body.sum(axis=1)-all_robot
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    result['max_body_partition_residual'] = float(np.max(abs(residual)))
    intervals = [('acquisition',168,235),('first',235,480),('release1',480,525),
                 ('carry1',300,440),('free',525,645),('second',645,725),
                 ('carry2',680,700),('release2',725,770),('final',770,857)]
    result['robot_intervals'] = {}
    for name,a,b in intervals:
        duration = float(robot['interval_timestamps'][b]-robot['interval_timestamps'][a])
        def mean(array):
            return (array[a:b].sum(axis=0)/duration).tolist()
        total = all_robot[a:b].sum(axis=0)
        selected = robot['interval_selected_patch_impulse_world'][a:b].sum(axis=0)
        active = robot['interval_active_selected_patch_impulse_world'][a:b].sum(axis=0)
        body = by_body[a:b].sum(axis=0)/duration
        result['robot_intervals'][name] = dict(first_frame=a,end_frame=b,duration_seconds=duration,
            rms_position_mm=float(1000*np.sqrt(np.mean(robot['position_error'][a:b+1]**2))),
            all_robot_mean_wrench=(total/duration).tolist(),
            selected_patch_mean_wrench=(selected/duration).tolist(),
            active_selected_patch_mean_wrench=(active/duration).tolist(),
            other_patch_mean_wrench=((total-selected)/duration).tolist(),
            environment_mean_wrench=mean(robot['interval_environment_contact_impulse_world']),
            bodies_mean_wrench={str(robot['robot_body_names'][i]):body[i].tolist()
                               for i in np.argsort(-np.linalg.norm(body[:,:3],axis=1))
                               if np.linalg.norm(body[i])>1e-6})
    for path in args.pads:
        data = read_npz(path/'corrected.npz')
        pad_reference = read_npz(path/'reference.npz')
        for key in ('q','object_positions','object_wxyz','force_ff','contact_active','object_local_points'):
            np.testing.assert_array_equal(reference[key],pad_reference[key])
        rows = {}
        for name,a,b in intervals:
            rows[name] = dict(
                rms_position_mm=float(1000*np.sqrt(np.mean(data['position_error'][a:b+1]**2))),
                contact_frames_per_channel=(data['contact_counts'][a:b+1]>0).sum(axis=0).tolist(),
                guidance_frames_per_channel=data['force_enabled'][a:b+1].sum(axis=0).tolist(),
                sampled_mean_force_n=data['contact_forces'][a:b+1].sum(axis=1).mean(axis=0).tolist(),
                max_pad_tracking_error_mm=float(1000*np.linalg.norm(
                    data['pad_positions'][a:b+1]-data['targets'][a:b+1],axis=-1).max()))
        result['pads'][path.name] = dict(source=digest(path/'corrected.npz'), intervals=rows)
    result['wall_seconds'] = perf_counter()-start
    result['cost'] = process_cost(); result['cost']['cpu_seconds'] -= cpu
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{key:row[key] for key in (
        'rms_position_mm','all_robot_mean_wrench','selected_patch_mean_wrench',
        'other_patch_mean_wrench','environment_mean_wrench')} for name,row in result['robot_intervals'].items()},
        indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,required=True)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--pads',type=Path,nargs='*',default=[])
    parser.add_argument('--output-name', default='contact_realization_audit.json')
    run(parser.parse_args())
