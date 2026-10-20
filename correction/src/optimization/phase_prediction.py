"""Local SE3 approach update from a preceding unassisted physical prediction."""
import numpy as np
from scipy.spatial.transform import Rotation
from .surface_reference import smoothstep


def updated_planning_poses(positions, wxyz, prediction_p, prediction_wxyz,
                           previous_shift, *, boundary=600, acquired=645,
                           fade_begin=660, fade_end=725, stationary_approach=False):
    """Keep the task observations separate from the temporary planning path.

    The world-frame rotation error left-multiplies the target orientation.
    Translation is around the object origin, not the world origin. Approach
    blending starts with zero first/second derivative at the frozen boundary.
    """
    positions, wxyz = np.asarray(positions), np.asarray(wxyz)
    if not 0 < boundary < acquired <= fade_begin < fade_end < len(positions):
        raise ValueError('Prediction update needs ordered original-frame boundaries')
    if stationary_approach and acquired == fade_begin:
        raise ValueError('Stationary approach needs a nonzero carry-transition duration')
    shift = np.asarray(prediction_p)-positions[boundary]
    rotation = (Rotation.from_quat(prediction_wxyz, scalar_first=True) *
                Rotation.from_quat(wxyz[boundary], scalar_first=True).inv()).as_rotvec()
    if np.linalg.norm(shift) > .12 or np.linalg.norm(rotation) > np.deg2rad(20):
        raise ValueError('Prediction outside declared 120mm/20degree update region')
    frames = np.arange(len(positions))
    blend = smoothstep((frames-boundary)/(acquired-boundary))
    decay = 1-smoothstep((frames-fade_begin)/(fade_end-fade_begin))
    old_positions = positions+decay[:, None]*previous_shift
    new_positions = positions+decay[:, None]*(
        (1-blend[:, None])*previous_shift+blend[:, None]*shift)
    new_rotations = (Rotation.from_rotvec((decay*blend)[:, None]*rotation) *
                     Rotation.from_quat(wxyz, scalar_first=True))
    if stationary_approach:
        # The settled object does not move along with a blended hand target.
        # Motion after acquisition is a planning hypothesis, not a physical
        # state write. Independently check the stationary assumption.
        carry = smoothstep((frames-acquired)/(fade_begin-acquired))
        new_positions = (1-carry[:, None])*prediction_p+carry[:, None]*new_positions
        initial = Rotation.from_quat(prediction_wxyz, scalar_first=True)
        relative = (new_rotations*initial.inv()).as_rotvec()
        new_rotations = Rotation.from_rotvec(carry[:, None]*relative)*initial
    return old_positions, new_positions, new_rotations.as_quat(scalar_first=True), dict(
        boundary_frame=boundary, acquired_frame=acquired, decay_frames=[fade_begin, fade_end],
        translation_residual_m=shift.tolist(), rotation_residual_world_rad=rotation.tolist(),
        prediction_source='prior full unassisted replay; command update only',
        stationary_approach=stationary_approach,
        target_motion_unchanged=True)


def transport_palms(palm_positions, palm_rotations, old_p, old_r, new_p, new_r):
    delta = Rotation.from_quat(new_r, scalar_first=True)*Rotation.from_quat(old_r, scalar_first=True).inv()
    return new_p+delta.apply(palm_positions-old_p), delta*palm_rotations
