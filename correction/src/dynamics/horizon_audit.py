"""Dense checks independent of the optimizer's scalar objective and status."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from correction.src.optimization.coupled_horizon import sample_solution


def audit_horizon(ref, robot, result, *, samples_per_interval=5):
    times = np.unique(np.r_[np.concatenate([np.linspace(a, b, samples_per_interval+1)
                       for a, b in zip(ref.times[:-1], ref.times[1:])]), ref.times])
    sampled = sample_solution(ref, result, times)
    matrix = Rotation.from_quat(sampled['wxyz'], scalar_first=True).as_matrix()
    forces = sampled['forces']
    force_residual = ref.mass*(sampled['acceleration']-[0, 0, -9.81])-forces.sum(1)
    moment = np.cross(np.einsum('tij,cj->tci', matrix, result['anchors']), forces).sum(1)
    iw = matrix@ref.inertia@np.swapaxes(matrix, 1, 2)
    angular_momentum = np.einsum('tij,tj->ti', iw, sampled['angular_velocity'])
    moment_residual = (np.einsum('tij,tj->ti', iw, sampled['angular_acceleration'])
                       +np.cross(sampled['angular_velocity'], angular_momentum)-moment)
    max_gap = max_cone = max_inactive = max_torque = 0.
    servo = result.get('settings', {})
    servo_offsets, servo_limit_violation, servo_bound_violation = [], 0., 0.
    min_separation = min_clearance = min_support_clearance = np.inf
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])*ref.half_size
    support_heights = [p.support_z for p in ref.contacts if p.support_z is not None]
    base_position = CubicSpline(ref.times, ref.base[:, :3])
    base_rotation = RotationSpline(ref.times, Rotation.from_quat(ref.base[:, 3:7], scalar_first=True))
    inward = np.zeros((3, robot.count))
    for patch in ref.contacts:
        if patch.robot_index >= 0: inward[:, patch.robot_index] = patch.inward
    for index, time in enumerate(times):
        k = min(np.searchsorted(ref.times, time, side='right')-1, len(ref.times)-2)
        u = (time-ref.times[k])/(ref.times[k+1]-ref.times[k])
        base = np.zeros(25)
        base[:3], base[3:7] = base_position(time), base_rotation(time).as_quat(scalar_first=True)
        base[7:10], base[13:16] = base_position(time, 1), base_position(time, 2)
        base[10:13], base[16:19] = base_rotation(time).apply(base_rotation(time, 1)), base_rotation(time).apply(base_rotation(time, 2))
        rm = matrix[index]
        tips = np.asarray(robot.contact_positions(sampled['q'][index], base, rm, inward)).T
        centers = np.asarray(robot.kinematics(sampled['q'][index], base)).T
        if support_heights:
            lowest = np.min((sampled['position'][index]+corners@rm.T)[:, 2])
            min_support_clearance = min(min_support_clearance, lowest-max(support_heights))
        hand_force = np.zeros((robot.count, 3))
        for i, patch in enumerate(ref.contacts):
            active = patch.active[k] and patch.active[k+1]
            if abs(u) < 1e-9: active = patch.active[k]
            if abs(u-1) < 1e-9: active = patch.active[k+1]
            f = forces[index, i]
            normal = np.array([0, 0, 1]) if patch.support_z is not None else rm@patch.inward
            fn = normal@f
            max_cone = max(max_cone, -fn, fn-patch.force_limit, np.linalg.norm(f-normal*fn)-patch.friction*fn)
            if not active: max_inactive = max(max_inactive, np.linalg.norm(f))
            location = sampled['position'][index]+rm@result['anchors'][i]
            if patch.robot_index >= 0:
                hand_force[patch.robot_index] += f
                gap = np.linalg.norm(tips[patch.robot_index]-location)
                if active:
                    max_gap = max(max_gap, gap)
                    min_separation = min(min_separation, (-normal)@(tips[patch.robot_index]-location))
                else:
                    # An eventual contact-face half-space is not a collision
                    # distance during approach. Check the whole box instead.
                    delta = np.abs(rm.T@(centers[patch.robot_index]-sampled['position'][index]))-ref.half_size
                    clearance = np.linalg.norm(np.maximum(delta, 0))+min(np.max(delta), 0)-robot.radius
                    min_clearance = min(min_clearance, clearance)
            elif active:
                max_gap = max(max_gap, abs(location[2]-patch.support_z))
        tau = np.asarray(robot.inverse_dynamics(sampled['q'][index], sampled['v'][index], sampled['a'][index], base)).ravel()
        contact_tau = np.asarray(robot.force_torque(sampled['q'][index], base, hand_force.T)).ravel()
        tau += contact_tau
        if servo.get('servo_kp') is not None:
            offset = contact_tau/np.asarray(servo['servo_kp'])
            servo_offsets.append(offset)
            command = sampled['q'][index]+offset
            servo_limit_violation = max(servo_limit_violation, np.max(robot.lower-command), np.max(command-robot.upper))
            servo_bound_violation = max(servo_bound_violation, np.max(abs(offset)-np.asarray(servo['servo_max_offset'])))
        max_torque = max(max_torque, np.max(abs(tau)/robot.caps))
    checks = dict(max_contact_gap_m=float(max_gap), max_friction_violation_n=float(max_cone),
                  max_inactive_force_n=float(max_inactive),
                  min_active_normal_separation_m=float(min_separation) if np.isfinite(min_separation) else None,
                  min_inactive_box_clearance_m=float(min_clearance) if np.isfinite(min_clearance) else None,
                  min_object_support_clearance_m=float(min_support_clearance) if np.isfinite(min_support_clearance) else None,
                  max_dense_torque_fraction=float(max_torque),
                  max_velocity_fraction=float(np.max(abs(sampled['v'])/robot.speed)),
                  max_joint_limit_violation=float(max(0., np.max(robot.lower-sampled['q']), np.max(sampled['q']-robot.upper))),
                  max_acceleration_fraction=float(np.max(abs(sampled['a'])/robot.acceleration)),
                  rms_joint_acceleration=float(np.sqrt(np.mean(sampled['a']**2))),
                  rms_joint_jerk=float(np.sqrt(np.mean(sampled['jerk']**2))),
                  max_force_residual_n=float(np.max(np.linalg.norm(force_residual, axis=1))),
                  max_moment_residual_nm=float(np.max(np.linalg.norm(moment_residual, axis=1))),
                  samples=len(times), full_robot_collision_checked=False)
    checks['accepted_nominal'] = bool(result['success'] and max_gap < .001 and max_cone < .01 and max_inactive < 1e-5
        and min_clearance >= -.0001 and min_support_clearance >= -.0001
        and max_torque <= 1. and checks['max_joint_limit_violation'] < 1e-5 and checks['max_velocity_fraction'] <= 1.001
        and checks['max_acceleration_fraction'] <= 1.001 and checks['max_force_residual_n'] < .05
        and checks['max_moment_residual_nm'] < .02)
    if servo_offsets:
        rate = np.max(abs(np.diff(servo_offsets, axis=0))/np.diff(times)[:, None])
        checks.update(max_servo_joint_limit_violation=float(servo_limit_violation),
                      max_servo_offset_violation=float(servo_bound_violation), max_servo_offset_rate=float(rate))
        checks['accepted_nominal'] &= bool(servo_limit_violation < 1e-5 and servo_bound_violation < 1e-5
                                            and rate <= servo['servo_offset_rate']*1.001)
    return checks
