"""Export robot references and supported physics experiments.

Neither export certifies that the reference is ready for policy training.
"""
import json
from pathlib import Path
import numpy as np


def save_robot_reference(directory, result, label="dynamics_guided"):
    # Import lazily so generic phase output does not require the robot runtime.
    from correction.src.preprocessing import retarget_bridge  # enables pinned dependencies
    from v2d_retarget.output import save_result

    path = save_result(directory, label, result.motion, result.robot,
                       result.solution, result.body_targets, result.contacts,
                       result.config, result.metrics, result.diagnostics)
    (directory / "robot_repair_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    return path


def save_robot_experiment(directory,report,records,interface,xml):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for name,data in records.items():
        np.savez_compressed(directory/f'{name}.npz',**data)
    np.savez_compressed(directory/'reference.npz',**interface)
    (directory/'scene.xml').write_text(xml,encoding='utf-8')
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
