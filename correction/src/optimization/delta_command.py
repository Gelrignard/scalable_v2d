"""Continuous accumulated commands with conservative whole-curve bounds."""
import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import BPoly


def command_curve(initial, endpoint, duration):
    """Start from previous command q/v/a, never from a new observation."""
    zero = np.zeros_like(endpoint)
    return BPoly.from_derivatives([0., duration], [list(initial), [endpoint, zero, zero]])


def held_value(curve, time, derivative=0):
    if time > curve.x[-1] and derivative:
        return np.zeros_like(curve(curve.x[-1]))
    return curve(np.clip(time, 0., curve.x[-1]), derivative)


def bernstein_bounds(curve):
    return [(curve.derivative(d).c[:, 0].min(axis=0),
             curve.derivative(d).c[:, 0].max(axis=0)) for d in range(3)]


def solve_endpoint(initial, duration, target, rows, values, lower, upper,
                   speed, acceleration):
    """Endpoint QP; all Bernstein control points satisfy joint/derivative limits.

    The decision is an increment from the previous command position. Bounds
    constrain the entire polynomial, not just video samples. Infeasibility is
    returned to the caller; no post-solve clipping of the curve is permitted.
    """
    q0 = initial[0]; n = len(q0)
    base = command_curve(initial, q0, duration)
    unit = command_curve(tuple(np.zeros(1) for _ in range(3)), np.ones(1), duration)
    constraints, lows, highs = [], [], []
    for d, lo, hi in [(0, lower, upper), (1, -speed, speed),
                      (2, -acceleration, acceleration)]:
        offsets = base.derivative(d).c[:, 0]
        factors = unit.derivative(d).c[:, 0, 0]
        for offset, factor in zip(offsets, factors):
            constraints.append(factor*sparse.eye(n, format='csc'))
            lows.extend(lo-offset); highs.extend(hi-offset)
    a = np.vstack([np.eye(n), *rows])
    b = np.concatenate([target-q0, *[v-r@q0 for r, v in zip(rows, values)]])
    problem = osqp.OSQP()
    problem.setup(P=sparse.csc_matrix(2*(a.T@a+1e-6*np.eye(n))), q=-2*a.T@b,
                  A=sparse.vstack(constraints, format='csc'), l=np.array(lows), u=np.array(highs),
                  verbose=False, eps_abs=2e-6, eps_rel=2e-6, max_iter=4000,
                  polishing=True)
    fit = problem.solve(raise_error=False)
    audit = dict(status=fit.info.status, solved=False)
    if fit.info.status_val not in (1, 2) or fit.x is None:
        return None, audit
    curve = command_curve(initial, q0+fit.x, duration)
    bounds = bernstein_bounds(curve)
    violation = max(float(np.maximum(lo-vlo, 0).max()) for (vlo, vhi), lo, hi in
                    zip(bounds, [lower, -speed, -acceleration], [upper, speed, acceleration]))
    violation = max(violation, max(float(np.maximum(vhi-hi, 0).max()) for (vlo, vhi), lo, hi in
                    zip(bounds, [lower, -speed, -acceleration], [upper, speed, acceleration])))
    audit.update(solved=violation < 1e-4, constraint_violation=violation)
    return (curve if audit['solved'] else None), audit
