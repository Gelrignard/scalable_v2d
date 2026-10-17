import numpy as np
from scipy.spatial.transform import Rotation
from .contacts import TIP_NAMES


def summary(values):
    a = np.asarray(values).reshape(-1)
    if not len(a):
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(len(a)),
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def evaluate(kin, motion, q, body, contacts, surface):
    """All geometry is re-evaluated on exported q, never intermediate solver q."""
    names = list(body.names) + list(TIP_NAMES) + kin.probe_names
    positions = []
    rotations = []
    for qt in q:
        poses = kin.poses(qt, names)
        positions.append([p.translation for p in poses])
        rotations.append([p.rotation for p in poses[: len(body.names)]])
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)
    b = len(body.names)
    tips = positions[:, b : b + 6]
    probes = positions[:, b + 6 :]
    object_rot = Rotation.from_quat(motion.object_wxyz, scalar_first=True).as_matrix()
    local_tips = np.einsum(
        "tji,tkj->tki", object_rot, tips - motion.object_positions[:, None, :]
    )
    _, active_surface_distance, _ = surface.nearest(local_tips[contacts.active])
    local_probes = np.einsum(
        "tji,tkj->tki", object_rot, probes - motion.object_positions[:, None, :]
    )
    d, _ = surface.hull_distance_normal(local_probes)
    penetration = np.maximum(np.asarray(kin.probe_radii) - d, 0.0)
    body_error = np.linalg.norm(positions[:, :b] - body.positions, axis=-1)
    human_error = np.linalg.norm(positions[:, :b] - body.source_positions, axis=-1)
    relative_rotation = np.einsum("tkji,tkjl->tkil", rotations, body.rotations)
    angular = np.degrees(
        Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3)).magnitude()
    ).reshape(len(q), b)
    tip_error = np.linalg.norm(tips - contacts.target_tips, axis=-1)
    anchors = contacts.anchors_local + contacts.clearance_m * contacts.normals_local
    contact_error = np.linalg.norm(local_tips - anchors, axis=-1)
    slip = (
        np.linalg.norm(np.diff(local_tips, axis=0), axis=-1)
        / np.diff(motion.timestamps)[:, None]
        if len(q) > 1
        else np.empty((0, 6))
    )
    stable = contacts.active[1:] & contacts.active[:-1]
    if len(q) > 1:
        velocity = (
            np.abs(np.diff(q[:, 7:], axis=0)) / np.diff(motion.timestamps)[:, None]
        )
        ratio = velocity / kin.robot.model.velocityLimit[6:]
        base_speed = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) / np.diff(
            motion.timestamps
        )
        base_rotation = Rotation.from_quat(q[:, 3:7])
        base_angular_speed = (
            base_rotation[:-1].inv() * base_rotation[1:]
        ).magnitude() / np.diff(motion.timestamps)
    else:
        ratio = np.empty((0, 43))
        base_speed = np.empty(0)
        base_angular_speed = np.empty(0)
    finger_ids = [i for i, n in enumerate(kin.joint_names) if "_hand_" in n]
    report = {
        "frames": len(q),
        "dynamics_tested": False,
        "contact_labels": "inferred geometric hypotheses",
        "collision_metric": "sphere probes vs convex hull; conservative for concave objects; incomplete robot coverage",
        "body_target_error_m": {
            n: summary(body_error[:, j]) for j, n in enumerate(body.names)
        },
        "body_human_error_m": {
            n: summary(human_error[:, j]) for j, n in enumerate(body.names)
        },
        "body_orientation_error_deg": {
            n: summary(angular[:, j]) for j, n in enumerate(body.names)
        },
        "tip_target_error_m": summary(tip_error),
        "active_contact_anchor_error_m": summary(contact_error[contacts.active]),
        "active_contact_surface_distance_m": summary(active_surface_distance),
        "active_contact_slip_m_s": summary(slip[stable]),
        "contact_frame_counts": {
            n: int(contacts.active[:, j].sum()) for j, n in enumerate(TIP_NAMES)
        },
        "hand_probe_penetration_m": summary(penetration),
        "frames_with_probe_penetration_over_5mm": int(
            np.any(penetration > 0.005, axis=1).sum()
        ),
        "joint_speed_limit_ratio": summary(ratio),
        "floating_base_linear_speed_m_s": summary(base_speed),
        "floating_base_angular_speed_rad_s": summary(base_angular_speed),
        "speed_violating_transitions": int(np.any(ratio > 1 + 1e-5, axis=1).sum()),
        "joint_limit_violation_rad": float(
            max(
                0.0,
                np.max(kin.robot.model.lowerPositionLimit[7:] - q[:, 7:]),
                np.max(q[:, 7:] - kin.robot.model.upperPositionLimit[7:]),
            )
        ),
        "finger_range_rad": {
            kin.joint_names[i]: float(np.ptp(q[:, 7 + i])) for i in finger_ids
        },
        "root_quaternion_norm_error": float(
            np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1))
        ),
    }
    arrays = dict(
        fk_body_positions=positions[:, :b],
        fk_tip_positions=tips,
        body_error_m=body_error,
        tip_error_m=tip_error,
        contact_error_m=contact_error,
        probe_penetration_m=penetration,
    )
    return report, arrays
