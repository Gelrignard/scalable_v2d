"""Small demonstration-local beam of box-face contact explanations.

This first experiment is specifically for the rigid book. The box is a
conservative proxy; the attained contacts are checked against scene geometry.
The skeleton supplies candidate locations/history, never measured force.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Hypothesis:
    name: str
    points: np.ndarray
    inward: np.ndarray
    active: np.ndarray
    evidence_cost: float
    load_weights: np.ndarray | None = None


def face_projection(point, bounds, face, inset=.002):
    axis, sign = divmod(int(face), 2)
    point = np.clip(point, bounds[0]+inset, bounds[1]-inset)
    point[axis] = bounds[sign, axis]
    inward = np.zeros(3)
    inward[axis] = 1. if sign == 0 else -1.
    return point, inward


def hypotheses(source_local, robot_local, bounds, observed_active, previous=None,
               max_candidates=5, neighborhood=.08):
    """Change one whole hand's face assignment at a time, not 6**T modes.

Each finger may remain on its own nearest face in the reference hypothesis.
Alternatives assign one hand to one plausible face. Candidates beyond the
demonstration neighborhood are rejected. Activation is still a hypothesis,
inferred from proximity; it is checked again on attained robot geometry.
"""
    if max_candidates < 1:
        raise ValueError('max_candidates must be positive')
    source = np.median(source_local, axis=0)
    robot = np.median(robot_local, axis=0)
    evidence = .7*source + .3*robot
    all_points, all_normals = [], []
    for finger in range(6):
        options = [face_projection(evidence[finger], bounds, f) for f in range(6)]
        all_points.append([p for p, _ in options])
        all_normals.append([n for _, n in options])
    all_points, all_normals = np.asarray(all_points), np.asarray(all_normals)
    distances = np.linalg.norm(all_points-source[:, None], axis=-1)
    nearest = np.argmin(distances, axis=1)
    choices = [('nearest', nearest)]
    # A dexterous pinch needs opposing finger normals; moving every finger of
    # a hand to the same face misses this common explanation of penetration.
    thin_axis = int(np.argmin(bounds[1]-bounds[0]))
    for left_sign in range(2):
        for right_sign in range(2):
            ids = nearest.copy()
            for hand, sign in enumerate((left_sign, right_sign)):
                ids[3*hand] = 2*thin_axis+sign
                ids[3*hand+1:3*hand+3] = 2*thin_axis+1-sign
            choices.append((f'pinch_{left_sign}_{right_sign}', ids))
    for hand, hand_name in enumerate(('left', 'right')):
        sl = slice(3*hand, 3*hand+3)
        for face in range(6):
            ids = nearest.copy()
            ids[sl] = face
            choices.append((f'{hand_name}_face_{face}', ids))
    result, seen = [], set()
    for name, ids in choices:
        if tuple(ids) in seen:
            continue
        seen.add(tuple(ids))
        p, n = all_points[np.arange(6), ids], all_normals[np.arange(6), ids]
        distance = np.linalg.norm(p-source, axis=-1)
        active = observed_active.copy() | (np.linalg.norm(source_local-p[None], axis=-1) < .035)
        active &= np.linalg.norm(source_local-p[None], axis=-1) <= neighborhood
        if np.any((distance > neighborhood) & active.any(axis=0)):
            continue
        score = float(np.mean(np.minimum(distance, neighborhood)**2)/.02**2)
        if previous is not None:
            score += .4*float(np.mean(np.linalg.norm(p-previous.points, axis=-1)**2)/.02**2)
        result.append(Hypothesis(name, p, n, active, score))
    result.sort(key=lambda h: (h.evidence_cost, h.name))
    if previous is not None:
        p = previous.points
        active = observed_active & (np.linalg.norm(source_local-p[None], axis=-1) <= neighborhood)
        continuation = Hypothesis('continue_previous', p.copy(), previous.inward.copy(), active, 0.)
        result.insert(0, continuation)
    return result[:max_candidates]
