"""NVIDIA training export: recompute EE and fingers from the final robot pose."""

from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from ..paths import enable_upstream
from ..core.contacts import TIP_NAMES


def export_motion_v1(directory, motion, kin, solved, contacts, report, diagnostics):
    if motion.upstream_motion is None:
        raise ValueError(
            "motion_v1 export currently needs demo metadata; canonical inputs still export JSON/NPZ"
        )
    if len(motion.timestamps) > 1:
        dt = np.diff(motion.timestamps)
        if not np.allclose(dt, dt[0], atol=1e-8):
            raise ValueError(
                "motion_v1 requires uniform timestamps; resample explicitly"
            )
        fps = 1 / float(dt[0])
    else:
        fps = float(motion.metadata.get("source_fps", 200))
    enable_upstream()
    from robotic_grounding.motion_schema import save_motion_parquet

    palms = [f"{s}_hand_palm_link" for s in ("left", "right")]
    # Isaac Lab's current tracking code selects distal LINK origins, not the
    # added virtual tip sites. Export both by name instead of substituting one.
    hands = [
        [
            f"{side}_hand_{part}_link"
            for part in (
                "palm",
                "thumb_0",
                "thumb_1",
                "thumb_2",
                "index_0",
                "index_1",
                "middle_0",
                "middle_1",
            )
        ]
        + [n for n in TIP_NAMES if n.startswith(side)]
        for side in ("left", "right")
    ]
    names = palms + hands[0] + hands[1]
    poses = []
    for q in solved.q:
        poses.append(
            [
                np.r_[
                    p.translation,
                    Rotation.from_matrix(p.rotation).as_quat(scalar_first=True),
                ]
                for p in kin.poses(q, names)
            ]
        )
    poses = np.asarray(poses)
    finger_names = [
        [n for n in kin.joint_names if n.startswith(s + "_hand_")]
        for s in ("left", "right")
    ]
    fingers = [
        solved.q[:, [7 + kin.joint_names.index(n) for n in names]].tolist()
        for names in finger_names
    ]
    original_articulation = np.asarray(motion.upstream_motion.object_articulation)
    source_indices = motion.metadata.get(
        "selected_source_indices", list(range(len(motion.timestamps)))
    )
    articulation = (
        original_articulation[source_indices].tolist()
        if original_articulation.ndim > 0 and original_articulation.size
        else []
    )
    md = replace(
        motion.upstream_motion,
        robot_name="g1",
        motion_kind="single_robot",
        fps=fps,
        robot_joint_names=kin.joint_names,
        robot_root_position=solved.q[:, :3].tolist(),
        robot_root_wxyz=solved.q[:, [6, 3, 4, 5]].tolist(),
        robot_joint_positions=solved.q[:, 7:].tolist(),
        ee_link_names=palms,
        ee_pose_w=poses[:, :2].tolist(),
        hand_sides=["left", "right"],
        hand_frame_names=hands,
        hand_frames_w=[
            poses[:, 2 : 2 + len(hands[0])].tolist(),
            poses[:, 2 + len(hands[0]) :].tolist(),
        ],
        hand_finger_joint_names=finger_names,
        hand_finger_joints=fingers,
        object_root_position=motion.object_positions.tolist(),
        object_articulation=articulation,
        object_root_axis_angle=Rotation.from_quat(motion.object_wxyz, scalar_first=True)
        .as_rotvec()
        .tolist(),
        object_body_position=motion.object_positions[:, None, :].tolist(),
        object_body_wxyz=motion.object_wxyz[:, None, :].tolist(),
        # A per-hand hypothesis is not a fabricated force/contact manifold.
        hand_contact_active=[
            contacts.active[:, :3].any(axis=1).astype(float).tolist(),
            contacts.active[:, 3:].any(axis=1).astype(float).tolist(),
        ],
        hand_contact_link_names=[],
        hand_link_contact_positions=[],
        hand_link_contact_normals=[],
        hand_object_contact_positions=[],
        hand_object_contact_normals=[],
        hand_object_contact_part_ids=[],
        source_kind="v2d_retarget_canonical",
        source_payload=b"",
        source_joint_names=list(motion.joint_names),
        frame_task_errors=diagnostics["body_error_m"].tolist(),
        ik_error_per_frame=diagnostics["body_error_m"].mean(axis=1).tolist(),
        ik_num_iterations=solved.iterations.tolist(),
    )
    return save_motion_parquet(md, root_path=str(directory), file_name="data.parquet")
