import hashlib
import json
from pathlib import Path
import numpy as np
from ..core.contacts import TIP_NAMES


def save_result(
    directory, label, motion, kin, solved, body, contacts, config, report, diagnostics
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "v2d_retarget_v1",
        "method": label,
        "sequence_id": motion.sequence_id,
        "robot": "g1_dex3",
        "position_units": "metres",
        "joint_units": "radians",
        "time_units": "seconds",
        "coordinate_frame": "world_z_up",
        "q_layout": "root_xyz, root_quaternion_xyzw, ordered_actuated_joints",
        "joint_names": kin.joint_names,
        "body_frame_names": list(body.names),
        "tip_names": list(TIP_NAMES),
        "source": motion.metadata,
        "config": config,
        "elapsed_seconds": solved.elapsed_seconds,
        "urdf_sha256": hashlib.sha256(kin.config.urdf_path.read_bytes()).hexdigest(),
        "body_mapping": {
            n: {
                "source_joint": e.soma_joint,
                "position_cost": e.position_cost,
                "orientation_cost": e.orientation_cost,
                "rotation_correction": np.asarray(kin.config.r_per_link[n]).tolist(),
                "translation_offset": np.asarray(kin.config.t_per_link[n]).tolist(),
            }
            for n, e in kin.config.ik_map.items()
        },
        "dynamics_tested": False,
        "contact_labels": "inferred, not force-validated",
    }
    arrays = dict(
        q=solved.q,
        timestamps=motion.timestamps,
        iterations=solved.iterations,
        stop_reason=np.array(solved.stopped),
        body_targets=body.positions,
        body_rotations=body.rotations,
        tip_targets=contacts.target_tips,
        source_tips=contacts.source_tips,
        contact_active=contacts.active,
        contact_blend=contacts.blend,
        contact_anchors_local=contacts.anchors_local,
        contact_normals_local=contacts.normals_local,
        human_positions=motion.positions,
        human_joint_names=np.array(motion.joint_names),
        object_positions=motion.object_positions,
        object_wxyz=motion.object_wxyz,
        object_mesh=str(motion.object_mesh),
        metadata_json=json.dumps(metadata),
        **diagnostics,
    )
    np.savez_compressed(directory / f"{label}.npz", **arrays)
    # Human-readable trajectory, including the base pose necessary for replay.
    trajectory = {
        **metadata,
        "timestamps": motion.timestamps.tolist(),
        "root_positions": solved.q[:, :3].tolist(),
        "root_quaternions_wxyz": solved.q[:, [6, 3, 4, 5]].tolist(),
        "joint_angles": solved.q[:, 7:].tolist(),
        "object_positions": motion.object_positions.tolist(),
        "object_quaternions_wxyz": motion.object_wxyz.tolist(),
        "contact_active": contacts.active.tolist(),
        "contact_anchors_object_local": contacts.anchors_local.tolist(),
        "stop_reason": solved.stopped,
        "metrics": report,
    }
    (directory / f"{label}.json").write_text(
        json.dumps(trajectory, indent=2, allow_nan=False), encoding="utf-8"
    )
    return directory / f"{label}.npz"


def load_result(path):
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k].copy() for k in data.files}
    metadata = json.loads(str(out["metadata_json"]))
    if metadata["schema"] != "v2d_retarget_v1":
        raise ValueError("Unsupported trajectory schema")
    if not np.isfinite(out["q"]).all():
        raise ValueError("Nonfinite robot trajectory")
    return out, metadata
