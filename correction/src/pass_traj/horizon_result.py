"""Separate a diagnostic plan from an accepted, independently replayed result."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np


def save_horizon(directory, ref, result, report, records=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'problem.json').write_text(json.dumps(asdict(ref), indent=2,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value), allow_nan=False), encoding='utf-8')
    numeric = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(directory/'plan.npz', times=ref.times, **numeric)
    np.savez_compressed(directory/'observations.npz', times=ref.observation_times,
                        positions=ref.observed_positions, wxyz=ref.observed_wxyz,
                        initial_q=ref.initial_q, joint_reference=ref.joint_reference, base=ref.base)
    for name, record in (records or {}).items():
        np.savez_compressed(directory/f'{name}.npz', **record)
    report = dict(report, ready_for_training=False, input_metadata=ref.metadata,
                  solver={key: value for key, value in result.items() if not isinstance(value, np.ndarray)})
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    # Deliberately no training-ready export based on solver status alone.
    return directory/'report.json'
