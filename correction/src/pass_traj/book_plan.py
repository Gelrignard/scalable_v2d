"""Persist the intermediate wrench plan consumed by robot refinement."""
import json
import numpy as np


def save_wrench_plan(directory, arrays, report):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "wrench_plan.npz", **arrays, report_json=json.dumps(report))
    (directory / "wrench_plan.json").write_text(json.dumps(report, indent=2))
