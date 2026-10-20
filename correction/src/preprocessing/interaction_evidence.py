"""Compact, uncertain interaction evidence extracted from retargeting output.

Object-frame hand evidence limits a local contact search. It is not measured
contact or force and does not make raw frame accelerations into observations.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation


def runs(mask):
    indices = np.flatnonzero(mask)
    return [] if len(indices) == 0 else np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)


def object_coordinates(world_points, positions, wxyz):
    world_points, positions = np.asarray(world_points), np.asarray(positions)
    if world_points.ndim != 3 or world_points.shape[0] != len(positions):
        raise ValueError('Expected [frame, point, xyz] observations')
    r = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
    return np.einsum('tji,tkj->tki', r, world_points-positions[:, None, :])


@dataclass
class ContactRegion:
    side: str
    first_frame: int
    last_frame: int
    center_object_m: list
    outward_object: list
    normal_coherence: float
    radius90_m: float
    evidence_samples: int
    source_channels: list


def regions(raw, first, last):
    """One coarse region per hand; retain dispersion instead of hard labels.

    Normal evidence is a medoid: averaging opposite wall/rim normals would
    invent a surface. Coherence reports whether the chosen direction dominates.
    """
    if first < 0 or last >= len(raw['timestamps']) or first > last:
        raise ValueError('Invalid frame interval')
    result = []
    for side, columns in [('left', [0, 1, 2]), ('right', [3, 4, 5])]:
        active = raw['contact_active'][first:last+1][:, columns]
        anchors = raw['contact_anchors_local'][first:last+1][:, columns][active]
        normals = raw['contact_normals_local'][first:last+1][:, columns][active]
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-8
        anchors, normals = anchors[valid], normals[valid]/lengths[valid, None]
        if len(anchors) == 0:
            continue
        # Unique rounded normals avoid a quadratic frame-by-frame distance table.
        directions, counts = np.unique(np.round(normals, 4), axis=0, return_counts=True)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        score = (directions@directions.T)@counts
        direction = directions[np.argmax(score)]
        compatible = normals@direction >= .5
        center = np.median(anchors[compatible], axis=0)
        radius = np.quantile(np.linalg.norm(anchors[compatible]-center, axis=1), .9)
        result.append(ContactRegion(side, first, last, center.tolist(), direction.tolist(),
            float(compatible.mean()), float(radius), len(anchors), columns))
    return result


def evidence_report(raw):
    local = object_coordinates(raw['source_tips'], raw['object_positions'], raw['object_wxyz'])
    phases = []
    for part in runs(np.any(raw['contact_active'], axis=1)):
        if len(part) < 4:
            continue
        first, last = int(part[0]), int(part[-1])
        phases.append(dict(first_frame=first, last_frame=last,
            time_s=[float(raw['timestamps'][first]-raw['timestamps'][0]),
                    float(raw['timestamps'][last]-raw['timestamps'][0])],
            object_displacement_world_m=(raw['object_positions'][last]-raw['object_positions'][first]).tolist(),
            regions=[asdict(r) for r in regions(raw, first, last)]))
    return dict(kind='demonstration-supported uncertain evidence; not physical contact labels',
                phases=phases, object_frame_source_tips_shape=list(local.shape))
