"""Instantaneous force-stage update about a staged nominal; no horizon or rollout.

Zero is always in the static trust region. Inherited torque/cone violations
may not increase at the current state, but are NOT certified as feasible.
"""
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse

from ..dynamics.sparse_forces import contact_basis, cross_matrix


def force_update(*, points, normals, touching, com, jacobians, nominal_force,
                 nominal_torque, caps, joint_position, lower, upper, goal,
                 previous, dt, limit, rate, friction, normal_cap,
                 joint_margin=.01, regularization=.1):
    started = perf_counter()
    count = len(points); size = 3*count
    jt = jacobians.reshape(size, -1).T
    grasp = np.column_stack([np.vstack((np.eye(3), cross_matrix(p-com)))*bool(touch)
                            for p, touch in zip(points, touching)])
    # A separated channel has no object-wrench authority. Any residual already
    # on the robot unloads through the same rate box rather than being reset.
    scale = np.array([3., 3., 3., .15, .15, .15])
    operator = grasp/scale[:, None]
    target = goal/scale
    rows, bounds = [], []
    for i in range(count):
        if not touching[i]:
            continue
        basis = contact_basis(normals[i])
        # Circumscribed 8-facet cone approximation with an inscribed radius.
        # Preserve any inherited violation rather than silently changing the
        # nominal controller just because this adapter was enabled.
        angles = np.arange(8)*np.pi/4
        cone = np.array([[-friction*np.cos(np.pi/8), np.cos(a), np.sin(a)] for a in angles])@basis.T
        local = np.vstack((-normals[i], normals[i], cone))
        base = local@nominal_force[i]
        # Retain most of the nominal compressive preload. Do not reduce a
        # grasp to zero just because opposing forces cancel in the task wrench.
        normal_floor = .75*max(0., normals[i]@nominal_force[i])
        absolute = np.r_[-normal_floor, normal_cap, np.zeros(8)]
        block = np.zeros((len(local), size)); block[:, 3*i:3*i+3] = local
        rows.extend(block); bounds.extend(np.maximum(absolute, base)-base)
    # Full nominal PD + gravity/passive/inertial compensation + force guidance.
    # Caps are not enlarged in execution. This permits only no-worse inherited
    # saturation, and the report separately rejects physical cap violations.
    envelope = np.maximum(caps, abs(nominal_torque))
    rows.extend(jt); bounds.extend(envelope-nominal_torque)
    rows.extend(-jt); bounds.extend(envelope+nominal_torque)
    for j in range(len(caps)):
        if joint_position[j] >= upper[j]-joint_margin:
            rows.append(jt[j]); bounds.append(0.)
        if joint_position[j] <= lower[j]+joint_margin:
            rows.append(-jt[j]); bounds.append(0.)
    matrix = np.asarray(rows); bound = np.asarray(bounds)
    old = np.asarray(previous).ravel()
    hessian = operator.T@operator+regularization*np.eye(size)
    linear = -2*operator.T@target
    # Separated channels and unobservable internal force directions tend to
    # zero through minimum-change force allocation, not invented supports.
    def solve(with_rate):
        low, high = np.full(size, -limit), np.full(size, limit)
        if with_rate:
            low = np.maximum(low, old-rate*dt); high = np.minimum(high, old+rate*dt)
        problem = osqp.OSQP()
        problem.setup(P=sparse.csc_matrix(np.triu(2*hessian)), q=linear,
            A=sparse.csc_matrix(np.vstack((matrix, np.eye(size)))),
            l=np.r_[np.full(len(bound), -np.inf), low], u=np.r_[bound, high],
            verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=2000, polishing=False)
        problem.warm_start(x=np.clip(old, low, high))
        fit = problem.solve(raise_error=False)
        if fit.x is None or not np.isfinite(fit.x).all() or fit.info.status_val not in (1, 2):
            return None, fit.info.status
        violation = max(np.max(matrix@fit.x-bound), np.max(low-fit.x), np.max(fit.x-high))
        return (fit.x if violation <= 1e-6 else None), fit.info.status
    answer, status = solve(True)
    rate_status = status
    conflict = answer is None
    if conflict:
        # A changing contact/torque constraint can make the rate box disjoint.
        # Prefer static actuator safety, log the discontinuity, reject the run.
        answer, status = solve(False)
    failed = answer is None
    if failed:
        answer = np.zeros(size)  # statically admissible; never hide the rate failure
    return answer.reshape(count, 3), dict(status=status, rate_status=rate_status, rate_conflict=conflict, failed=failed,
        seconds=perf_counter()-started,
        delta_wrench=grasp@answer, unmet_wrench=goal-grasp@answer,
        static_violation=max(0., float(np.max(matrix@answer-bound))))
