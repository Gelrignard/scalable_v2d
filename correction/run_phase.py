"""Run the core on an explicit, retargeter-independent sticking-phase bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .models import ContactPatch
from .optimizer import CorrectionSettings, PhaseCorrector, validate_dense


def save_result(result, output: Path, provenance, extra_arrays=None):
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: result[name] for name in ("timestamp_s", "required_wrench_object", "environment_wrench_object", "reference_points_object_m")}
    for stage in ("before", "after"):
        arrays.update({f"{stage}_{name}": value for name, value in result[stage].items() if name != "metrics"})
    arrays.update(extra_arrays or {})
    np.savez_compressed(output / "correction.npz", **arrays)
    report = {**result["report"], "provenance": provenance}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("contact_mode") != "sticking":
        parser.error("this first prototype supports only a single sticking phase")
    if not config.get("provenance") or not config.get("object_frame_id"):
        parser.error("config needs provenance and an explicit object_frame_id")
    patches = [ContactPatch(**item) for item in config["patches"]]
    settings = CorrectionSettings(**config.get("settings", {}))
    with np.load(args.phase, allow_pickle=False) as phase:
        if str(phase["object_frame_id"]) != config["object_frame_id"]:
            parser.error("phase and contact patches use different object frames")
        corrector = PhaseCorrector(
            phase["timestamp_s"], patches, phase["reference_points_object_m"],
            phase["required_wrench_object"], phase["com_object_m"], settings,
            phase["environment_wrench_object"] if "environment_wrench_object" in phase else None)
        result = corrector.solve()
        extra = {}
        if "dense_timestamp_s" in phase:
            dense = validate_dense(result, phase["dense_timestamp_s"], phase["dense_required_wrench_object"],
                                   phase["com_object_m"], settings, phase["dense_environment_wrench_object"])
            result["report"]["dense_validation"] = dense.pop("report")
            extra.update(dense)
            extra.update({key: phase[key] for key in phase.files if key.startswith("dense_")})
            pose = phase["dense_object_pose_xyz_wxyz"]
            rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
            extra["corrected_contact_targets_world_m"] = np.einsum("tij,kj->tki", rotation, result["after"]["contact_points_object_m"]) + pose[:, None, :3]
            extra["world_frame_id"] = phase["world_frame_id"]
        extra["object_frame_id"] = phase["object_frame_id"]
    report = save_result(result, args.output, config, extra)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
