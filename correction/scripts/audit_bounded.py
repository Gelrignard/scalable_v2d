"""Recheck a saved bounded reference; optionally repeat the short physics trials."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from correction.src.preprocessing import retarget_bridge
from correction.src.dynamics.models import RigidBody
from correction.src.dynamics.object_wrench import object_wrench
from correction.src.dynamics.robot_scene import RobotScene
from correction.src.dynamics.window_audit import attained
from correction.src.optimization.rollout_refinement import compare_rollouts
from correction.src.pass_traj.bounded_reference import save_bounded
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.core.surface import ObjectSurface
from v2d_retarget.output import load_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--rollout', action='store_true')
    args = parser.parse_args()
    tick = perf_counter()
    report_path = args.directory/'bounded_report.json'
    report = json.loads(report_path.read_text())
    before, metadata = load_result(Path(report['input_reference']))
    after, _ = load_result(args.directory/'bounded_reference.npz')
    kin = G1Kinematics()
    surface = ObjectSurface(Path(str(before['object_mesh'])))
    parameters = report['physical_parameters']
    scene = RobotScene(kin, surface.mesh.bounds, metadata['source']['support_surface'],
                       mass=parameters['mass_kg'], friction=parameters['friction'])
    body = RigidBody(parameters['mass_kg'], parameters['com_m'], parameters['inertia_kg_m2'],
                     np.array([0., 0., -9.81]), parameters['provenance'])
    times = before['timestamps']
    pose = np.column_stack((before['object_positions'], before['object_wxyz']))
    required = object_wrench(times, pose, np.ones(len(times), bool), body, window=1)['required_wrench_object']
    lo, hi = report['correction_frames_inclusive']
    ids = np.arange(lo-2, hi+3)
    for label, data in [('before', before), ('after', after)]:
        report[label], arrays = attained(kin, scene, data['q'][ids], data['object_positions'][ids], data['object_wxyz'][ids],
                                    times[ids], required[ids], midpoints=True)
    if args.rollout:
        start, stop = report['requested_frames_inclusive']
        ids = np.arange(start, stop+1)
        records, report['rollouts'], report['control_refinement'] = compare_rollouts(scene, kin,
            before['q'][ids], after['q'][ids], after['object_positions'][ids], after['object_wxyz'][ids], times[ids],
            -after['contact_normals_local'][ids], after['contact_active'][ids])
        for label, result in records.items():
            np.savez_compressed(args.directory/f'rollout_{label}.npz', **result)
    report['revalidation_seconds'] = perf_counter()-tick
    ids = np.arange(lo-2, hi+3)
    save_bounded(args.directory, before, metadata, after['q'], ids, after['tip_targets'][ids],
                 after['contact_anchors_local'][ids], -after['contact_normals_local'][ids],
                 after['contact_active'][ids], arrays, report)
    print(json.dumps(dict(revalidation_seconds=report['revalidation_seconds'], rollouts=report['rollouts']), indent=2))


if __name__ == '__main__':
    main()
