"""Serialize contact targets, force trajectories, and their audit report."""
import json
from pathlib import Path
import numpy as np


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
