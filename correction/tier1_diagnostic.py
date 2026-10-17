"""Compute object-wrench demand on a real Tier 1 clip; does NOT infer hand contacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import trimesh

from .dynamics import object_wrench
from .fetch_sample import sha256
from .models import RigidBody


def verify_cache(root):
    manifest = json.loads((root / "download_manifest.json").read_text())
    if manifest["tier"] != 1:
        raise ValueError("this diagnostic is explicitly a Tier 1-only experiment")
    for relative, entry in manifest["files"].items():
        if sha256(root / relative) != entry["sha256"]:
            raise ValueError(f"cached source changed: {relative}")
    return manifest


def diagnostic(root, output, episode, mass, window):
    started = perf_counter()
    manifest = verify_cache(root)
    records = [json.loads(line) for line in (root / "meta/episodes_metadata.jsonl").read_text().splitlines() if line]
    entry = next(item for item in records if item["episode_index"] == episode)
    source_file = root / f"data/chunk-000/episode_{episode:06d}.parquet"
    table = pq.read_table(source_file)
    if table.schema.metadata.get(b"pose_convention") != b"world_T_object":
        raise ValueError("unexpected object pose convention")
    times = np.asarray(table["timestamp"], dtype=float)
    poses = np.asarray(table["observation.object.pose"].to_pylist(), dtype=float)
    visible = np.asarray(table["observation.object.visible"], dtype=bool)
    mesh = trimesh.load(root / entry["mesh"], force="mesh", process=False)
    bounds = np.asarray(mesh.bounds)
    lengths = bounds[1] - bounds[0]
    com = bounds.mean(axis=0)
    # Deliberate pilot assumptions. The released mesh/pose does not measure
    # mass, inertia or friction. Do not label this as the evaluator's body model.
    inertia = np.diag(mass / 12 * (np.sum(lengths**2) - lengths**2))
    plane = np.asarray(json.loads((root / entry["ground_plane"]).read_text())["plane"], dtype=float)
    plane /= np.linalg.norm(plane[:3])
    object_side = np.median(poses[visible, :3] @ plane[:3] + plane[3])
    if abs(object_side) < 0.02:
        raise ValueError("plane sign ambiguous; explicit gravity required")
    up = plane[:3] * np.sign(object_side)
    body = RigidBody(mass, com, inertia, -9.81 * up,
        "PILOT ASSUMPTION: user-selected mass; uniform solid axis-aligned mesh-bounding-box inertia/COM; gravity normal to fitted ground plane, sign chosen toward median visible object origin. Not calibrated simulator parameters.")
    raw = object_wrench(times, poses, visible, body, window=1)
    smooth = object_wrench(times, poses, visible, body, window=window)
    valid = raw["valid_wrench"] & smooth["valid_wrench"]
    if not np.any(valid):
        raise ValueError("no valid interior frames for requested smoothing window")
    smooth_pose = smooth["smooth_pose_xyz_wxyz"]
    position_change = np.linalg.norm(smooth_pose[valid, :3] - poses[valid, :3], axis=1)
    raw_rotation = Rotation.from_quat(poses[valid][:, [4, 5, 6, 3]])
    smooth_rotation = Rotation.from_quat(smooth_pose[valid][:, [4, 5, 6, 3]])
    angle_change = (raw_rotation.inv() * smooth_rotation).magnitude()

    def statistics(result):
        wrench = result["required_wrench_object"][valid]
        force, torque = np.linalg.norm(wrench[:, :3], axis=1), np.linalg.norm(wrench[:, 3:], axis=1)
        return {"force_norm_median_n": float(np.median(force)), "force_norm_p95_n": float(np.percentile(force, 95)),
                "force_norm_max_n": float(force.max()), "torque_norm_p95_nm": float(np.percentile(torque, 95)),
                "torque_norm_max_nm": float(torque.max())}

    physical = json.loads(json.dumps(asdict(body), default=lambda value: value.tolist()))
    report = {"episode_index": episode, "task": entry["task"], "object": entry["object"],
              "dataset_revision": manifest["revision"], "tier": 1,
              "input_sha256": sha256(source_file), "frame_count": len(times),
              "visible_frames": int(visible.sum()), "valid_interior_frames": int(valid.sum()),
              "duration_s": float(times[-1] - times[0]), "smoothing_window_frames": window,
              "mesh_bounds_object_m": bounds.tolist(), "body_model": physical,
              "unsmoothed_demand": statistics(raw), "smoothed_demand": statistics(smooth),
              "position_change_rms_m": float(np.sqrt(np.mean(position_change**2))),
              "position_change_max_m": float(position_change.max()),
              "orientation_change_rms_rad": float(np.sqrt(np.mean(angle_change**2))),
              "orientation_change_max_rad": float(angle_change.max()),
              "contacts_inferred": False, "robot_feasibility_checked": False,
              "physical_rollout_performed": False,
              "interpretation": "Required wrench from ALL nongravitational contacts, including table/floor. Not a hand-force estimate. Smoothing reduces derivative sensitivity; no evidence that it recovers the true motion.",
              "runtime_seconds": perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "assumed_body.json").write_text(json.dumps(physical, indent=2) + "\n")
    np.savez_compressed(output / "object_dynamics.npz", **smooth,
                        unsmoothed_required_wrench_object=raw["required_wrench_object"],
                        frame_index=np.asarray(table["frame_index"]), com_object_m=body.com_object_m,
                        object_frame_id=np.array(f"{entry['object']}:published_mesh"))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("correction/data/tier1"))
    parser.add_argument("--output", type=Path, default=Path("correction/outputs/tier1_episode_000011"))
    parser.add_argument("--episode", type=int, default=11)
    parser.add_argument("--mass-kg", type=float, required=True, help="explicit pilot mass assumption, not a dataset measurement")
    parser.add_argument("--window", type=int, default=11)
    args = parser.parse_args()
    diagnostic(args.data, args.output, args.episode, args.mass_kg, args.window)
