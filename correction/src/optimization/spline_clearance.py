"""Linearize actual mesh clearance in the same cubic command coefficients."""
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline, CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp
from ..dynamics.collision_distance import distance_row


def clearance_rows(scene, reference, knots, obstacle, remembered=(), *, clearance=.002, frequency=150):
    """No slack or exemption: only an explicitly inactive approach interval."""
    start, end = obstacle['frames']
    if np.any(reference['contact_active'][start:end+1]):
        raise ValueError('Free-space clearance cannot be imposed on loaded contacts')
    t = reference['timestamps']; width = len(scene.movable)
    samples = np.linspace(t[start], t[end], int(np.ceil((t[end]-t[start])*frequency))+1)
    curve = CubicHermiteSpline(t, reference['q'][:, 7+scene.movable], reference['command_velocity'])
    root_rot = Slerp(t, Rotation.from_quat(reference['q'][:, 3:7]))(samples).as_quat()
    object_rot = Slerp(t, Rotation.from_quat(obstacle['object_wxyz'], scalar_first=True))(samples).as_quat(scalar_first=True)
    operators = BSpline.design_matrix(samples, knots, 3).tocsr()
    states = []; pairs = set(tuple(p) for p in remembered)
    for i, time in enumerate(samples):
        q = reference['q'][start].copy()
        q[:3] = [np.interp(time, t, reference['q'][:, j]) for j in range(3)]
        q[3:7] = root_rot[i]; q[7+scene.movable] = curve(time)
        p = np.array([np.interp(time, t, obstacle['object_positions'][:, j]) for j in range(3)])
        scene.set_state(q, p, object_rot[i])
        pairs.update((int(c.geom1), int(c.geom2)) for c in scene.data.contact)
        states.append((q, p, object_rot[i]))
    rows, bounds = [], []
    for i, (q, p, r) in enumerate(states):
        scene.set_state(q, p, r)
        for a, b in sorted(pairs):
            geometry = distance_row(scene, a, b)
            if geometry is None:
                continue
            jac = geometry['jacobian']
            row = sparse.kron(operators[i:i+1], sparse.csr_matrix(jac[None]), format='csc')
            rows.append(row)
            bounds.append(clearance-geometry['distance']+jac@q[7+scene.movable])
    matrix = sparse.vstack(rows, format='csc') if rows else sparse.csc_matrix((0, operators.shape[1]*width))
    return matrix, np.asarray(bounds), dict(
        frames=[start, end], frequency_hz=frequency, samples=len(samples), rows=len(rows),
        remembered_pairs=[list(p) for p in sorted(pairs)], clearance_m=clearance,
        scope='One local signed-distance linearization in executed spline coefficients; dense nonlinear audit required')
