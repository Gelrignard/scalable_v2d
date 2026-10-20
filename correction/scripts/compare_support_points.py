"""Run two complete independent-pad episodes using the saved G1 force plan."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.core.contacts import TIP_NAMES
from correction.src.dynamics.independent_contacts import IndependentContacts
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.preprocessing.convex_object import decompose
from correction.src.preprocessing.strategy_bundle import load_strategy_scene
from correction.src.preprocessing.surface_pad_reference import derive_pad_targets
from correction.src.pass_traj.experiment_record import snapshot_sources, process_cost


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def run(directory, output, matching_acquisition=False):
    if output.exists():
        raise ValueError('Choose a new output directory; preserve previous experiments')
    start = perf_counter()
    cpu=process_cost()['cpu_seconds']
    ref = read(directory/'reference.npz')
    source = json.loads((directory/'report.json').read_text())
    source_dir = Path(source['input_directory'])
    surface_contract='robot_local_points' in ref
    if matching_acquisition and not surface_contract:
        raise ValueError('Patch acquisition requires a real-surface reference')
    if surface_contract:
        _,raw,base,robot_scene=load_strategy_scene(source_dir)
        source_dir=Path(base['input_directory'])
    else:
        raw = read(source_dir/'retargeted/contact_aware.npz')
    meta = json.loads(str(raw['metadata_json']))
    support = Path(meta['source']['support_surface']).read_text()
    # The selected trash-can episode has floor only. Do not silently omit
    # environment geometry when this script is pointed at a different task.
    if any(f'def {kind}' in support for kind in ('Cylinder', 'Mesh', 'Cube', 'Sphere', 'Capsule')):
        raise ValueError('This full-episode runner currently supports floor-only episodes')
    frames=len(ref['timestamps'])
    duration=float(ref['timestamps'][-1]-ref['timestamps'][0])
    episode=source.get('episode',dict(task=Path(str(ref['object_mesh'])).stem))
    np.testing.assert_array_equal(ref['frame_ids'], np.arange(frames))
    np.testing.assert_allclose(ref['timestamps'], raw['timestamps']-raw['timestamps'][0], atol=1e-8)
    if surface_contract:
        pad_fields=derive_pad_targets(robot_scene,ref,source,.006)
        original=pad_fields['original_tip_targets']
        corrected=pad_fields['corrected_pad_targets']
        initial=pad_fields['initial_pads']
        inward=ref['inward_normals_local']
    else:
        kin = G1Kinematics()
        original = np.array([[pose.translation for pose in kin.poses(q, TIP_NAMES)] for q in ref['baseline_command_q']])
        initial = np.array([pose.translation for pose in kin.poses(ref['initial_q'], TIP_NAMES)])
        corrected=ref['tip_targets']
        inward=-raw['contact_normals_local']
        pad_fields=dict(original_tip_targets=original,corrected_pad_targets=corrected,initial_pads=initial,
            channel_names=np.array(['L thumb','L index','L middle','R thumb','R index','R middle']))
    parts = decompose(str(ref['object_mesh']), source_dir/'collision')
    physics = source['physics']
    scene = IndependentContacts(ref['bounds'], mass=physics['mass_kg'], friction=physics['friction'],
        dt=physics['timestep_s'], floor_z=float(ref['floor_z']), object_parts=parts,
        object_inertial=physics.get('object_inertial'),pad_count=original.shape[1])
    config = (ContactControllerConfig(**source['controller']) if surface_contract else
              ContactControllerConfig(feedback_gain=.25))
    output.mkdir(parents=True)
    cases = {}
    initial_state = None
    provenance=snapshot_sources(output,[Path(__file__),Path('correction/src/dynamics/independent_contacts.py'),
        Path('correction/src/dynamics/supported_contacts.py'),Path('correction/src/preprocessing/surface_pad_reference.py'),
        Path('correction/src/dynamics/contact_controller.py')])
    for name, targets in [('original', original), ('corrected', corrected)]:
        print(f'Independent pads: {name}, {frames} frames / {duration:.2f} s', flush=True)
        begin = perf_counter()
        options = {} if name == 'original' else dict(planned_forces=ref['force_ff'],
            inward_normals_local=inward, active=ref['contact_active'], config=config)
        if name != 'original' and matching_acquisition:
            options['object_local_points'] = ref['object_local_points']
        result = scene.rollout(ref['timestamps'], targets, ref['object_positions'], ref['object_wxyz'],
                               initial_pads=initial, **options)
        if len(result['object_positions']) != frames or result['max_object_applied_force'] != 0:
            raise AssertionError('Rollout must be complete and object must be unassisted')
        state = np.concatenate([result[key][0].ravel() for key in
            ('object_positions', 'object_wxyz', 'object_linear_velocity', 'pad_positions', 'pad_velocities')])
        if initial_state is None: initial_state = state
        np.testing.assert_array_equal(state, initial_state)
        if not all(np.isfinite(value).all() for key,value in result.items() if key != 'contact_gate_gap_m'):
            raise AssertionError('Nonfinite rollout export')
        if np.max(abs(result['simulation_timestamps']-ref['timestamps'])) > physics['timestep_s']/2+1e-9:
            raise AssertionError('Simulation samples must match the source clock within half a physics step')
        np.savez_compressed(output/f'{name}.npz', **result)
        cases[name] = dict(rms_position_error_mm=float(1000*np.sqrt(np.mean(result['position_error']**2))),
            max_position_error_mm=float(1000*result['max_position_error_m']),
            max_rotation_error_deg=float(np.rad2deg(result['max_rotation_error_rad'])),
            max_penetration_mm=float(1000*result['max_penetration_m']),
            max_motor_force_n=result['max_motor_force_n'], max_object_applied_force=result['max_object_applied_force'],
            frames_with_contact=int(np.any(result['contact_counts'] > 0, axis=1).sum()),
            frames_with_guidance=int(np.any(result['force_enabled'], axis=1).sum()),
            saturation_steps=result['saturation_steps'], wall_seconds=perf_counter()-begin,
            accepted_tracking=bool(result['max_position_error_m'] <= .05 and
                                   result['max_rotation_error_rad'] <= np.deg2rad(15) and result['max_penetration_m'] <= .005))
        print(json.dumps(cases[name]), flush=True)
    np.savez_compressed(output/'reference.npz', **dict(ref,**pad_fields))
    (output/'scene.xml').write_text(scene.xml, encoding='utf-8')
    report = dict(scope='Independent finite-mass pads and free object; complete episode, not G1 execution',
        episode=episode, frames=frames, duration_s=duration,
        input_directory=str(directory.resolve()), source_reference_sha256=hashlib.sha256((directory/'reference.npz').read_bytes()).hexdigest(),
        cases=cases, initial_state_identical=True, resets_during_episode=0, object_assistance=False,
        physical_parameters=dict(mass_kg=physics['mass_kg'], friction=physics['friction'], floor_z_m=float(ref['floor_z']),
            timestep_s=scene.model.opt.timestep, collision_pieces=len(parts), pad_radius_m=scene.radius,
            object_inertial=physics.get('object_inertial'), pad_count=scene.pad_count,
            pad_mass_kg=scene.pad_mass, pad_stiffness_n_m=scene.kp, pad_damping_ns_m=scene.kd, pad_force_limit_n=scene.force_limit),
        guidance=asdict(config),
        force_acquisition=('matched contact or sphere-surface-to-patch distance below capture gap'
                           if matching_acquisition else 'matched physical contact only; configured capture gap unused'),
        explanation=('Original/corrected targets use the same real robot surface channels, with centers offset6mm behind '
            'their FK surface normals. Saved forces and contact schedule are unchanged; force guidance is gated to '
            'matched physical contact. Both independent trials start before approach at rest, without artificial grasp. '
            'Mass, COM, inertia, collision pieces, friction and floor match the robot model. Contact solver settings and '
            'finite spherical pads differ: this tests a relaxation, not robot execution or identical controller dynamics.'
            if surface_contract else 'Historical FK/pre-IK fingertip-pad diagnostic, full unassisted episodes.'),
        surface_contract=surface_contract,channel_names=pad_fields['channel_names'].tolist(),
        provenance=provenance,ready_for_training=False, seconds=perf_counter()-start,cost=process_cost())
    report['cost']['cpu_seconds']-=cpu
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {output} ({report["seconds"]:.1f} s CPU wall time)', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('correction/outputs/g1_controller_full_20261019_v2'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/support_points_full_20261019'))
    parser.add_argument('--matching-acquisition', action='store_true')
    args = parser.parse_args()
    run(args.directory, args.output, args.matching_acquisition)
