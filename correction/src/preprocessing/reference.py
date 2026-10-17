"""Convert a retargeter's explicit geometric reference into one correction phase.

No SOMA decoding, robot model imports, learned inference, or frame guessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from correction.src.dynamics.object_wrench import object_wrench, world_points_to_object
from correction.src.dynamics.models import RigidBody


def prepare(reference_path, config_path, output, start, stop, stride=3, window=11):
    config = json.loads(config_path.read_text())
    if config.get("contact_mode") != "sticking":
        raise ValueError("select one sticking phase, not a full episode with changing contacts")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    body = RigidBody(**config["body_model"])
    with np.load(reference_path, allow_pickle=False) as data:
        for key in ("world_frame_id", "object_frame_id"):
            if str(data[key]) != config[key]:
                raise ValueError(f"reference and config disagree on {key}")
        if list(data["point_names"].astype(str)) != [item["name"] for item in config["patches"]]:
            raise ValueError("point_names must match patch names in exactly the same order")
        times = data["timestamp_s"]
        poses = data["object_pose_xyz_wxyz"]
        points = data["reference_points_world_m"]
        frames = data["frame_index"]
        if frames.shape != times.shape or np.any(np.diff(frames) <= 0):
            raise ValueError("frame_index must be unique and increasing")
        dynamics = object_wrench(times, poses, data["object_visible"], body, window=window)
        selected = np.flatnonzero((frames >= start) & (frames < stop))
        if len(selected) < 2 or np.any(np.diff(selected) != 1):
            raise ValueError("select a nonempty contiguous phase with >=2 frames")
        if not np.all(dynamics["valid_wrench"][selected]):
            raise ValueError("phase crosses missing/invalid poses or unreliable derivative boundaries")
        dense_pose = dynamics["smooth_pose_xyz_wxyz"][selected]
        local_points = world_points_to_object(dense_pose, points[selected])
        knots = np.unique(np.r_[np.arange(0, len(selected), stride), len(selected)-1])
        if len(knots) > 64:
            raise ValueError("CPU prototype caps each phase at 64 force knots; increase stride or split the phase")
        # The environmental wrench must already use the object axes and COM.
        environment = data["environment_wrench_object"][selected] if "environment_wrench_object" in data else np.zeros((len(selected), 6))
        required = dynamics["required_wrench_object"][selected]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output,
            timestamp_s=times[selected][knots], required_wrench_object=required[knots],
            reference_points_object_m=local_points[knots], com_object_m=body.com_object_m,
            environment_wrench_object=environment[knots], object_frame_id=data["object_frame_id"],
            world_frame_id=data["world_frame_id"],
            dense_timestamp_s=times[selected], dense_required_wrench_object=required,
            dense_environment_wrench_object=environment, dense_frame_index=frames[selected],
            dense_object_pose_xyz_wxyz=dense_pose,
            dense_original_object_pose_xyz_wxyz=poses[selected],
            dense_reference_points_object_m=local_points)
    return {"output": str(output.resolve()), "dense_frames": len(selected), "force_knots": len(knots),
            "window_frames": window, "contact_mode": "sticking", "start_frame_inclusive": start,
            "stop_frame_exclusive": stop}
