"""Read the saved retargeting interface without loading a robot runtime."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import trimesh


@dataclass
class SupportedReference:
    input_path: Path
    ids: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    human_tips: np.ndarray
    robot_tips: np.ndarray
    bounds: np.ndarray
    mesh_path: Path
    table: tuple
    all_times: np.ndarray
    all_poses: np.ndarray


def load_supported_reference(path, start, stop):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        times = data['timestamps'].copy()
        p, r = data['object_positions'].copy(), data['object_wxyz'].copy()
        human, robot = data['source_tips'].copy(), data['fk_tip_positions'].copy()
        mesh_path = Path(str(data['object_mesh']))
        metadata = json.loads(str(data['metadata_json']))
    if not 6 <= start < stop <= len(times)-6:
        raise ValueError('Need six surrounding frames for filtered inverse dynamics')
    if not np.isfinite(np.c_[p, r]).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Object poses/timestamps must be finite and increasing')
    bounds = trimesh.load(mesh_path, force='mesh').bounds
    support = Path(metadata['source']['support_surface']).read_text()
    radius = float(re.search(r'double radius = ([\d.eE+-]+)', support).group(1))
    height = float(re.search(r'double height = ([\d.eE+-]+)', support).group(1))
    center = np.array(list(map(float, re.search(r'xformOp:translate = \(([^)]+)\)', support).group(1).split(','))))
    ids = np.arange(start, stop)
    return SupportedReference(path, ids, times[ids], p[ids], r[ids], human[ids], robot[ids],
                              bounds, mesh_path, (center, radius, height), times, np.c_[p, r])
