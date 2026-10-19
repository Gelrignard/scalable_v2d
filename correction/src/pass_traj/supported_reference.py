"""Save measured dynamics separately from the commands and desired object pose."""
import json
from pathlib import Path
import numpy as np


def save_supported(directory, reference, report, records, plan, xml):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for label, record in records.items():
        np.savez_compressed(directory/f'{label}.npz', **record)
    np.savez_compressed(directory/'reference.npz', frame_ids=reference.ids, times=reference.times,
                        object_positions=reference.positions, object_wxyz=reference.quaternions,
                        robot_tips=reference.robot_tips, human_tips=reference.human_tips,
                        bounds=reference.bounds, object_mesh=str(reference.mesh_path),
                        table_center=reference.table[0], table_radius=reference.table[1],
                        table_height=reference.table[2], **plan)
    (directory/'scene.xml').write_text(xml, encoding='utf-8')
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
