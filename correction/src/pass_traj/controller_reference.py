"""Export physical trajectories separately from nominal actuator setpoints."""
from pathlib import Path
import numpy as np
from correction.src.dynamics.contact_controller import position_target


def controller_reference(ref, robot, result, *, kp, max_offset):
    """Nominal knot commands; existing inverse-dynamics compensation retained.

    Output is a diagnostic contract, not an acceptance or training certificate.
    Online feedback and dense replay remain necessary. No observed pose is
    differentiated here; acceleration comes from the optimized robot spline.
    """
    from correction.src.optimization.coupled_horizon import sample_solution
    sample = sample_solution(ref, result, ref.times)
    compensation, contact = [], []
    for k in range(len(ref.times)):
        forces = np.zeros((robot.count, 3))
        for i, patch in enumerate(ref.contacts):
            if patch.robot_index >= 0:
                forces[patch.robot_index] += sample['forces'][k, i]
        compensation.append(np.asarray(robot.inverse_dynamics(sample['q'][k], sample['v'][k], sample['a'][k], ref.base[k])).ravel())
        contact.append(np.asarray(robot.force_torque(sample['q'][k], ref.base[k], forces.T)).ravel())
    compensation, contact = np.asarray(compensation), np.asarray(contact)
    target, requested, residual = position_target(sample['q'], contact, kp, robot.lower, robot.upper, max_offset)
    return dict(times=ref.times, physical_q=sample['q'], physical_velocity=sample['v'],
                servo_q=target, requested_servo_q=requested, servo_velocity=sample['v'],
                feedforward_torque=compensation, contact_torque=contact,
                desired_total_torque=compensation+contact, conversion_residual=residual,
                kp=np.asarray(kp), max_offset=np.asarray(max_offset), ready_for_training=False,
                scope='Nominal knot commands; contact torque encoded once in servo_q; compensation retained')


def save_controller_reference(directory, ref, robot, result, *, kp, max_offset):
    path = Path(directory)/'controller_reference.npz'
    record = controller_reference(ref, robot, result, kp=kp, max_offset=max_offset)
    np.savez_compressed(path, **record)
    return dict(file=path.name, max_conversion_residual=float(np.max(abs(record['conversion_residual']))),
                ready_for_training=False)
