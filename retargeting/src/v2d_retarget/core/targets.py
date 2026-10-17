from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BodyTargets:
    names: tuple[str, ...]
    positions: np.ndarray
    rotations: np.ndarray
    source_positions: np.ndarray
    position_costs: np.ndarray
    orientation_costs: np.ndarray


def build_body_targets(motion, config):
    names = tuple(config.ik_map)
    entries = [config.ik_map[n] for n in names]
    ids = [motion.index(e.soma_joint) for e in entries]
    source = motion.positions[:, ids].copy()
    rotations = (
        Rotation.from_quat(
            motion.rotations_wxyz[:, ids].reshape(-1, 4), scalar_first=True
        )
        .as_matrix()
        .reshape(len(source), len(names), 3, 3)
    )
    for j, name in enumerate(names):
        rotations[:, j] = rotations[:, j] @ config.r_per_link[name]
    offsets = np.array([config.t_per_link[n] for n in names])
    positions = source + np.einsum("tkij,kj->tki", rotations, offsets)
    feet = [names.index(n) for n in config.foot_frames]
    dz = config.ankle_roll_offset - positions[:, feet, 2].min(axis=1)
    positions[:, feet, 2] += dz[:, None]
    return BodyTargets(
        names,
        positions,
        rotations,
        source,
        np.array([e.position_cost for e in entries]),
        np.array([e.orientation_cost for e in entries]),
    )
