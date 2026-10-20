"""Same-initial-state controller ablations for a coupled G1 plan."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from correction.src.optimization.coupled_horizon import sample_solution
from correction.src.optimization.robot_manipulation import metrics
from .contact_controller import ContactControllerConfig


def replay_g1(ref, robot, scene, raw, result, *, controller_variants=False):
    times = ref.observation_times
    sampled = sample_solution(ref, result, times)
    original = raw['q'][:len(times)].copy()
    root_p = CubicSpline(ref.times, ref.base[:, :3])(times)
    root_r = RotationSpline(ref.times, Rotation.from_quat(ref.base[:, 3:7], scalar_first=True))(times)
    # Both controllers receive exactly the same supported pelvis trajectory.
    original[:, :3], original[:, 3:7] = root_p, root_r.as_quat()
    corrected = original.copy()
    corrected[:, 7+scene.movable] = sampled['q']
    original_velocity = CubicSpline(times, original[:, 7+scene.movable])(times, 1)
    original_velocity[0] = ref.initial_joint_velocity
    com = scene.model.body_ipos[scene.book_body]
    observed_p = ref.observed_positions-Rotation.from_quat(ref.observed_wxyz, scalar_first=True).apply(np.tile(com, (len(times), 1)))
    initial_rotation = Rotation.from_quat(ref.initial_wxyz, scalar_first=True)
    initial_omega = initial_rotation.apply(np.linalg.solve(ref.inertia, initial_rotation.inv().apply(ref.initial_angular_momentum)))
    initial = dict(position=ref.initial_position-initial_rotation.apply(com), wxyz=ref.initial_wxyz,
                   linear_velocity=ref.initial_velocity-np.cross(initial_omega, initial_rotation.apply(com)),
                   angular_velocity_world=initial_omega)
    force = np.zeros((len(times), robot.count, 3))
    for i, patch in enumerate(ref.contacts):
        if patch.robot_index >= 0: force[:, patch.robot_index] += sampled['forces'][:, i]
    scene.model.opt.cone = 1  # elliptic, as in the documented controlled benchmark
    scene.model.opt.impratio = 20
    scene.model.opt.noslip_iterations = 10
    scene.model.opt.tolerance = 1e-10
    common = dict(kp_arm=300., kp_hand=20., inertia_ff=True, initial_q=original[0],
                  initial_velocity=ref.initial_joint_velocity, initial_object_state=initial,
                  smooth_fixture=True)
    records = {}
    for name, q, velocity, guidance in (
        ('baseline', original, original_velocity, None),
        ('trajectory_only', corrected, sampled['v'], None),
        ('corrected', corrected, sampled['v'], force)):
        print(f'Independent G1 physics: {name}', flush=True)
        records[name] = scene.rollout(q, observed_p, ref.observed_wxyz, times,
                                     force_ff=guidance, force_points=robot.contacts,
                                     command_velocity=velocity, **common)
    normals = np.zeros((len(times), robot.count, 3))
    active = np.zeros((len(times), robot.count), bool)
    intervals = np.clip(np.searchsorted(ref.times, times, side='right')-1, 0, len(ref.times)-2)
    for patch in ref.contacts:
        if patch.robot_index >= 0:
            normals[:, patch.robot_index] = patch.inward
            active[:, patch.robot_index] = patch.active[intervals] & patch.active[intervals+1]
    variants = (
        ('gated_feedforward', ContactControllerConfig()),
        ('position_equivalent', ContactControllerConfig(interface='position')),
        ('force_feedback', ContactControllerConfig(feedback_gain=.25)),
        ('position_feedback', ContactControllerConfig(interface='position', feedback_gain=.25)))
    for name, config in variants if controller_variants else ():
        print(f'Independent G1 physics: {name}', flush=True)
        records[name] = scene.rollout(corrected, observed_p, ref.observed_wxyz, times,
            force_ff=force, force_points=robot.contacts, command_velocity=sampled['v'],
            contact_controller=config, contact_normals_local=normals, contact_active=active, **common)
    report = {name: metrics(value) for name, value in records.items()}
    for name, value in records.items():
        actual_r = Rotation.from_quat(value['object_wxyz'], scalar_first=True)
        actual_com = value['object_positions']+actual_r.apply(np.tile(com, (len(times), 1)))
        report[name]['rms_error_to_estimated_object_mm'] = float(1000*np.sqrt(np.mean(np.sum((actual_com-sampled['position'])**2, axis=1))))
        report[name]['goal_position_error_mm'] = float(1000*np.linalg.norm(actual_com[-1]-ref.goal_position))
        report[name]['rms_actual_joint_acceleration'] = float(np.sqrt(np.mean((np.diff(value['joint_velocities'], axis=0)/np.diff(times)[:, None])**2)))
        report[name]['accepted_physics'] = bool(report[name]['rms_error_to_estimated_object_mm'] < 10
            and report[name]['goal_position_error_mm'] < 1000*ref.goal_position_tolerance
            and value['max_rotation_error_rad'] < np.deg2rad(10) and value['max_penetration_m'] < .001
            and value['max_object_applied_force'] == 0. and value['max_torque_fraction_requested'] <= 1.
            and value['max_joint_limit_violation_rad'] < .01)
    report['accepted_correction'] = bool(report['corrected']['accepted_physics']
        and report['corrected']['rms_position_error_mm'] < report['baseline']['rms_position_error_mm']
        and report['corrected']['rms_actual_joint_acceleration'] <= report['baseline']['rms_actual_joint_acceleration'])
    return records, report
