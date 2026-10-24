"""Complete a pose timeline without turning interpolants into observations."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def complete_object_poses(times, poses, visible):
    times, poses, visible = np.asarray(times), np.asarray(poses), np.asarray(visible, bool)
    if poses.shape != (len(times), 7) or visible.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError('Expected increasing timestamps and matching xyz/wxyz observations')
    valid = visible & np.isfinite(poses).all(axis=1) & (np.linalg.norm(poses[:, 3:], axis=1) > 1e-8)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError('No usable observed object pose; cannot create a reference')
    result = np.empty_like(poses, dtype=float)
    for j in range(3):
        result[:, j] = np.interp(times, times[ids], poses[ids, j])
    rotations = Rotation.from_quat(poses[ids, 3:], scalar_first=True)
    result[:, 3:] = (rotations[0].as_quat(scalar_first=True) if len(ids) == 1 else
        Slerp(times[ids], rotations)(np.clip(times, times[ids[0]], times[ids[-1]])).as_quat(scalar_first=True))
    # Sign alignment is representational only; retain every observed rotation.
    for k in range(1, len(result)):
        if result[k, 3:]@result[k-1, 3:] < 0:
            result[k, 3:] *= -1
    return result, valid
