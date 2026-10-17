from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, RotationSpline

from .models import RigidBody


def contiguous_runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def object_wrench(times_s, pose_xyz_wxyz, visible, body: RigidBody, window=11):
    """Estimate required NON-GRAVITATIONAL wrench about the COM in object axes.

    Poses describe world_T_object; object origin need not equal COM. Missing
    poses are never interpreted as rotations or interpolated across. Endpoints
    of each visible segment are excluded from valid_wrench to avoid presenting
    spline/filter boundary derivatives as reliable observations.

    Smoothing: Savitzky-Golay translation and sign-continuous quaternion
    coefficients, followed by quaternion normalization. This is an approximate
    local smoother, not a reconstruction estimator or a contact-aware filter.
    Timestamps and raw input poses are unchanged in the returned record.
    """
    times = np.asarray(times_s, dtype=float)
    poses = np.asarray(pose_xyz_wxyz, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    count = len(times)
    if times.shape != (count,) or poses.shape != (count, 7) or visible.shape != (count,):
        raise ValueError("expected timestamps [T], poses [T,7], visibility [T]")
    if count < 5 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("need at least five strictly increasing finite timestamps")
    dt = np.median(np.diff(times))
    if not np.allclose(np.diff(times), dt, rtol=2e-3, atol=1e-6):
        raise ValueError("nonuniform timestamps: resample explicitly before this prototype")
    if window != 1 and (window < 5 or window % 2 != 1):
        raise ValueError("window must be 1 (unsmoothed) or an odd integer >= 5")
    if not np.all(np.isfinite(poses[visible])):
        raise ValueError("visible poses must be finite")
    if np.any(np.linalg.norm(poses[visible, 3:], axis=1) < 1e-8):
        raise ValueError("a visible object has a zero quaternion")

    smooth = np.full((count, 7), np.nan)
    wrench = np.full((count, 6), np.nan)
    com_acceleration = np.full((count, 3), np.nan)
    omega_body = np.full((count, 3), np.nan)
    alpha_body = np.full((count, 3), np.nan)
    valid = np.zeros(count, dtype=bool)
    margin = max(2, window // 2)
    for run in contiguous_runs(visible):
        if len(run) < max(window, 2 * margin + 1):
            continue
        t = times[run]
        p = poses[run, :3].copy()
        q = poses[run, 3:][:, [1, 2, 3, 0]].copy()
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        for i in range(1, len(q)):
            if np.dot(q[i - 1], q[i]) < 0:
                q[i] *= -1
        if window > 1:
            p = savgol_filter(p, window, 3, axis=0, mode="interp")
            q = savgol_filter(q, window, 3, axis=0, mode="interp")
            qnorm = np.linalg.norm(q, axis=1, keepdims=True)
            if np.any(qnorm < 0.5):
                raise ValueError("rotation varies too much for the local quaternion smoother")
            q /= qnorm
        rotation = Rotation.from_quat(q)
        spline = RotationSpline(t, rotation)
        # RotationSpline rates are expressed in the rotating/body axes.
        omega, alpha = spline(t, 1), spline(t, 2)
        origin_acc = CubicSpline(t, p)(t, 2)
        com = body.com_object_m
        local_com_acc = np.cross(alpha, com) + np.cross(omega, np.cross(omega, com))
        acc = origin_acc + rotation.apply(local_com_acc)
        force = body.mass_kg * rotation.inv().apply(acc - body.gravity_world_m_s2)
        angular_momentum = omega @ body.inertia_com_object_kg_m2.T
        torque = alpha @ body.inertia_com_object_kg_m2.T + np.cross(omega, angular_momentum)
        smooth[run, :3] = p
        smooth[run, 3:] = q[:, [3, 0, 1, 2]]
        wrench[run] = np.column_stack((force, torque))
        com_acceleration[run] = acc
        omega_body[run], alpha_body[run] = omega, alpha
        valid[run[margin:-margin]] = True
    # Invalid derivatives, including near gaps, cannot accidentally enter a solve.
    for array in (wrench, com_acceleration, omega_body, alpha_body):
        array[~valid] = np.nan
    return dict(timestamp_s=times.copy(), raw_pose_xyz_wxyz=poses.copy(),
                smooth_pose_xyz_wxyz=smooth, visible=visible.copy(),
                valid_wrench=valid, required_wrench_object=wrench,
                com_acceleration_world_m_s2=com_acceleration,
                angular_velocity_object_rad_s=omega_body,
                angular_acceleration_object_rad_s2=alpha_body)


def world_points_to_object(pose_xyz_wxyz, points_world):
    poses = np.asarray(pose_xyz_wxyz, dtype=float)
    points = np.asarray(points_world, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or points.ndim != 3 or points.shape[0] != len(poses) or points.shape[2] != 3:
        raise ValueError("expected pose [T,7] and points [T,K,3]")
    if not np.isfinite(poses).all() or not np.isfinite(points).all():
        raise ValueError("select a finite, visible phase before converting points")
    rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    return np.einsum("tji,tkj->tki", rotation, points - poses[:, None, :3])
