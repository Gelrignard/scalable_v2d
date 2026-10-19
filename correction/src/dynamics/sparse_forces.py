"""Sparse, time-coupled force allocation for prescribed contact hypotheses.

This is an object-wrench existence test, not a robot execution certificate.
All vectors are in object coordinates and moments are about the object COM.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse


def cross_matrix(p):
    x, y, z = p
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def contact_basis(inward):
    n = np.asarray(inward, float)
    n = n / np.linalg.norm(n)
    tangent = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((n, tangent, np.cross(n, tangent)))


@dataclass
class ForceResult:
    forces: np.ndarray
    residual: np.ndarray
    feasible: np.ndarray
    status: str
    seconds: float
    matrix_nnz: int


def allocate_forces(points, inward, active, required, times, com, *, mu=.6,
                    cap=15., time_limit=2., effort=1e-5, rate=1e-5, solver_tolerance=1e-7):
    """Solve a convex QP with conservative diamond friction cones.

Inactive contacts have exactly zero force. Wrench residuals are penalized,
then independently checked in physical units (0.02 N, 0.002 Nm). No force
solution or numerical convergence is interpreted as manipulation success.
"""
    started = perf_counter()
    points, inward, required = map(lambda x: np.asarray(x, float), (points, inward, required))
    active, times, com = np.asarray(active, bool), np.asarray(times, float), np.asarray(com, float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('points must have shape [T,K,3]')
    t, k, _ = points.shape
    if inward.shape != points.shape or active.shape != (t, k) or required.shape != (t, 6):
        raise ValueError('inconsistent contact/wrench shapes')
    if times.shape != (t,) or t < 1 or np.any(np.diff(times) <= 0) or com.shape != (3,):
        raise ValueError('invalid timestamps or COM')
    if not all(np.isfinite(x).all() for x in (points, inward, required, times, com)):
        raise ValueError('force problem must be finite')
    caps = np.broadcast_to(np.asarray(cap, float), (t, k))
    if (np.any(np.linalg.norm(inward, axis=-1) < 1e-9) or not np.isfinite(mu) or mu < 0
            or not np.isfinite(caps).all() or np.any(caps <= 0) or time_limit <= 0
            or not np.isfinite(solver_tolerance) or solver_tolerance <= 0):
        raise ValueError('invalid normal, friction, force cap or time budget')
    scale = np.array([10., 10., 10., 1., 1., 1.])
    operators, cones, lower, upper = [], [], [], []
    for frame in range(t):
        operators.append(sparse.csc_matrix(np.column_stack([
            np.vstack((np.eye(3), cross_matrix(p-com))) for p in points[frame]
        ]) / scale[:, None]))
        for finger in range(k):
            basis = contact_basis(inward[frame, finger])
            cone = np.array([[1, 0, 0]] + [[-.85*mu, a, b]
                           for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]) @ basis.T
            cones.append(sparse.csc_matrix(cone))
            lower.extend([0., -np.inf, -np.inf, -np.inf, -np.inf])
            upper.extend([caps[frame, finger] if active[frame, finger] else 0., 0., 0., 0., 0.])
    operator = sparse.block_diag(operators, format='csc')
    constraints = sparse.block_diag(cones, format='csc')
    n = 3*t*k
    hessian = operator.T @ operator + effort*sparse.eye(n, format='csc')
    if t > 1:
        diff = sparse.diags([-1/np.diff(times), 1/np.diff(times)], [0, 1], shape=(t-1, t))
        difference = sparse.kron(diff, sparse.eye(3*k), format='csc')
        hessian += rate*(difference.T @ difference)
    problem = osqp.OSQP()
    problem.setup(P=sparse.triu(2*hessian, format='csc'),
                  q=-2*np.asarray(operator.T @ (required/scale).ravel()).ravel(),
                  A=constraints, l=np.asarray(lower), u=np.asarray(upper),
                  verbose=False, eps_abs=solver_tolerance, eps_rel=solver_tolerance, max_iter=5000,
                  polishing=True, time_limit=max(.001, time_limit))
    solved = problem.solve(raise_error=False)
    ok = solved.info.status_val in (1, 2) and solved.x is not None and np.isfinite(solved.x).all()
    forces = solved.x.reshape(t, k, 3) if ok else np.zeros((t, k, 3))
    residual = np.column_stack((forces.sum(axis=1),
                               np.cross(points-com, forces).sum(axis=1))) - required
    violation = constraints @ forces.ravel()
    constraints_ok = (np.asarray(violation) >= np.asarray(lower)-1e-5).all() and (
        np.asarray(violation) <= np.asarray(upper)+1e-5).all()
    feasible = (np.linalg.norm(residual[:, :3], axis=1) <= .02) & (
        np.linalg.norm(residual[:, 3:], axis=1) <= .002) & ok & constraints_ok
    return ForceResult(forces, residual, feasible, solved.info.status,
                       perf_counter()-started, int(hessian.nnz+constraints.nnz))
