"""Source adapters. Only the pinned, hash-verified example permits pickle."""

import hashlib
import json
import pickle
import subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .motion import HumanMotion
from ..paths import WORKSPACE, UPSTREAM, UPSTREAM_COMMIT, DEMO_PARQUET, enable_upstream


def load_demo(fps=None):
    enable_upstream()
    from robotic_grounding.motion_schema import load_motion_data_parquet
    from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
    from robotic_grounding.retarget.robot_config import load_robot_config

    rel = DEMO_PARQUET.relative_to(WORKSPACE / "upstream").as_posix()
    pointer = subprocess.check_output(
        ["git", "cat-file", "blob", f"{UPSTREAM_COMMIT}:{rel}"],
        cwd=WORKSPACE / "upstream",
        text=True,
    )
    expected = next(
        line.split("sha256:")[1]
        for line in pointer.splitlines()
        if line.startswith("oid sha256:")
    )
    digest = hashlib.sha256(DEMO_PARQUET.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("Example parquet failed pinned LFS hash verification")
    md = load_motion_data_parquet(DEMO_PARQUET, device=None)
    source = pickle.loads(md.source_payload)
    world = np.asarray(load_robot_config("g1").r_world)
    p = np.asarray(source["soma_joints"], dtype=float) @ world.T
    root = source.get("source_root_translation", source.get("nvhuman_root_translation"))
    p += (np.asarray(root) - p[:, SOMA_JOINTS_ORDER.index("Hips")])[:, None, :]
    head = source.get("source_head_translation", source.get("nvhuman_head_translation"))
    if not np.allclose(p[:, SOMA_JOINTS_ORDER.index("Head")], head, atol=1e-5):
        raise ValueError("Cached source head/root frames disagree")
    raw_q = np.asarray(source["soma_joints_wxyz"])
    r = Rotation.from_quat(raw_q.reshape(-1, 4), scalar_first=True).as_matrix()
    q = Rotation.from_matrix(world @ r).as_quat(scalar_first=True).reshape(raw_q.shape)
    rate = float(md.fps if fps is None else fps)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("FPS must be positive")
    mesh = Path(md.object_mesh_paths[0])
    if not mesh.is_absolute():
        mesh = UPSTREAM / mesh
    return HumanMotion(
        tuple(SOMA_JOINTS_ORDER),
        p,
        q,
        np.arange(len(p)) / rate,
        np.asarray(md.object_root_position),
        Rotation.from_rotvec(md.object_root_axis_angle).as_quat(scalar_first=True),
        mesh,
        md.sequence_id,
        dict(
            source=str(DEMO_PARQUET),
            source_sha256=digest,
            source_fps=float(md.fps),
            fps_override=fps,
            coordinate_frame="world_z_up",
            timestamp_source="parquet_declared_fps" if fps is None else "user_override",
            upstream_commit=UPSTREAM_COMMIT,
        ),
        md,
    )


def save_npz(motion, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_names=np.asarray(motion.joint_names),
        positions=motion.positions,
        rotations_wxyz=motion.rotations_wxyz,
        timestamps=motion.timestamps,
        object_positions=motion.object_positions,
        object_wxyz=motion.object_wxyz,
        object_mesh=str(motion.object_mesh),
        sequence_id=motion.sequence_id,
        metadata_json=json.dumps(motion.metadata),
        coordinate_frame="world_z_up",
        units="metres_seconds",
    )


def load_npz(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as a:
        if (
            str(a["coordinate_frame"]) != "world_z_up"
            or str(a["units"]) != "metres_seconds"
        ):
            raise ValueError("Canonical input must use world_z_up and metres_seconds")
        mesh = Path(str(a["object_mesh"]))
        if not mesh.is_absolute():
            mesh = path.parent / mesh
        return HumanMotion(
            tuple(a["joint_names"].tolist()),
            a["positions"],
            a["rotations_wxyz"],
            a["timestamps"],
            a["object_positions"],
            a["object_wxyz"],
            mesh,
            str(a["sequence_id"]),
            json.loads(str(a["metadata_json"])),
        )
