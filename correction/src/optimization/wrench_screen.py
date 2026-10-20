"""Small actuator-bounded wrench screen for an explicitly stated contact mode.

This is a necessary static-capacity diagnostic, not inverse dynamics from
noisy observations and not proof of task infeasibility when a screen fails.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from correction.src.dynamics.sparse_forces import contact_basis, cross_matrix


def screen_wrench(points, inward, com, required, jacobians, bias, torque_caps,
                  *, friction=.45, normal_caps=15., seconds=2.):
    p, n, com, wanted, jac, bias, caps = map(lambda x: np.asarray(x, float),
        (points, inward, com, required, jacobians, bias, torque_caps))
    count = len(p)
    if p.shape != (count, 3) or n.shape != p.shape or wanted.shape != (6,):
        raise ValueError('Contact points/normals and six-dimensional wrench required')
    if jac.shape != (count, 3, len(bias)) or caps.shape != bias.shape:
        raise ValueError('Contact Jacobians and actuator limits have incompatible shapes')
    if not all(np.isfinite(x).all() for x in (p, n, com, wanted, jac, bias, caps)):
        raise ValueError('Nonfinite screen data')
    normal_caps = np.broadcast_to(np.asarray(normal_caps, float), (count,))
    if (np.any(np.linalg.norm(n, axis=1) < 1e-8) or not np.isfinite(friction) or friction < 0 or
            not np.isfinite(normal_caps).all() or np.any(normal_caps <= 0) or np.any(caps <= 0)):
        raise ValueError('Invalid normals or physical limits')
    start = perf_counter()
    size = 3*count+12
    grasp = np.column_stack([np.vstack((np.eye(3), cross_matrix(r-com))) for r in p])
    equality = np.column_stack((grasp, -np.eye(6), np.eye(6)))
    rows, rhs = [], []
    for i in range(count):
        basis = contact_basis(n[i])
        cone = np.array([[1, 0, 0], [-1, 0, 0]]+
                        [[-.85*friction, a, b] for a, b in [(1,1),(1,-1),(-1,1),(-1,-1)]])@basis.T
        block = np.zeros((6, size)); block[:, 3*i:3*i+3] = cone
        rows.extend(block); rhs.extend([normal_caps[i], 0, 0, 0, 0, 0])
    jt = jac.reshape(3*count, len(bias)).T
    torque = np.column_stack((jt, np.zeros((len(bias), 12))))
    rows.extend(torque); rhs.extend(caps-bias)
    rows.extend(-torque); rhs.extend(caps+bias)
    cost = np.zeros(size)
    # Force/moment slack in declared units. A tiny compression penalty chooses
    # among equivalent allocations without hiding physical residuals.
    cost[-12:] = np.tile([.1, .1, .1, 1., 1., 1.], 2)
    cost[:3*count] = (1e-6*n/np.linalg.norm(n, axis=1)[:, None]).ravel()
    fit = linprog(cost, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                  A_eq=equality, b_eq=wanted,
                  bounds=[(None, None)]*(3*count)+[(0, None)]*12,
                  method='highs', options={'time_limit': seconds})
    attempts = [str(fit.message)]
    if fit.status == 4 and perf_counter()-start < seconds:
        # Degenerate support vertices occasionally trigger a HiGHS presolve
        # error. Retry another algorithm without presolve under the same budget.
        fit = linprog(cost, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                      A_eq=equality, b_eq=wanted,
                      bounds=[(None, None)]*(3*count)+[(0, None)]*12,
                      method='highs-ipm', options={'presolve': False,
                          'time_limit': max(.001, seconds-(perf_counter()-start))})
        attempts.append(str(fit.message))
    force = fit.x[:3*count].reshape(count, 3) if fit.success else np.zeros_like(p)
    residual = grasp@force.ravel()-wanted
    actual_torque = bias+jt@force.ravel()
    force_error, moment_error = np.linalg.norm(residual[:3]), np.linalg.norm(residual[3:])
    primal_violation = float(max(0., np.max(np.asarray(rows)@fit.x-np.asarray(rhs)),
        np.max(abs(equality@fit.x-wanted)))) if fit.success else None
    return dict(kind='static_capacity_screen_not_execution_certificate', status=str(fit.message),
                solver_attempts=attempts, primal_violation=primal_violation,
                solved=bool(fit.success), feasible=bool(fit.success and primal_violation <= 1e-5 and force_error <= .1 and moment_error <= .01),
                force_residual_n=float(force_error), moment_residual_nm=float(moment_error),
                required_wrench=wanted.tolist(), forces=force.tolist(),
                max_torque_fraction=float(np.max(abs(actual_torque)/caps)),
                seconds=perf_counter()-start)
