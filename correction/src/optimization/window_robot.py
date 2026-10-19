"""Sparse local SQP for robot contact realization and temporal constraints."""
from time import perf_counter

import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse
from scipy.spatial.transform import Rotation

from .budget import BudgetExceeded


def temporal_operators(times, dofs):
    """Velocity and acceleration at interval midpoints, for nonuniform times."""
    dt = np.diff(times)
    velocity = sparse.diags([-1/dt, 1/dt], [0, 1], shape=(len(dt), len(times)))
    if len(dt) > 1:
        mid_dt = (dt[:-1]+dt[1:])/2
        acceleration = sparse.diags([-1/mid_dt, 1/mid_dt], [0, 1],
                                    shape=(len(mid_dt), len(dt))) @ velocity
    else:
        acceleration = sparse.csr_matrix((0, len(times)))
    return (sparse.kron(velocity, sparse.eye(dofs), format='csc'),
            sparse.kron(acceleration, sparse.eye(dofs), format='csc'))


def project_temporal(kin, desired, reference, times, movable, *, fixed_prefix=2,
                     fixed_suffix=2, trust=.9, time_limit=3., fixed_reference=None):
    """Project onto joint, speed, acceleration and endpoint constraints together.

    Used before local SQP and at export seams. Returns no accepted result if
    the convex constraints are inconsistent or their numerical check fails.
    """
    count, d = len(desired), len(movable)
    qids, vids = np.asarray(movable)+7, np.asarray(movable)+6
    vop, aop = temporal_operators(times, d)
    velocity = .9*kin.robot.model.velocityLimit[vids]
    accel = np.array([200. if 'hand' in kin.joint_names[i] else 80. for i in movable])
    lower = np.maximum(reference[:, qids]-trust, kin.robot.model.lowerPositionLimit[qids])
    upper = np.minimum(reference[:, qids]+trust, kin.robot.model.upperPositionLimit[qids])
    fixed = np.unique(np.r_[np.arange(fixed_prefix), np.arange(count-fixed_suffix, count)])
    boundary = reference if fixed_reference is None else fixed_reference
    lower[fixed] = boundary[fixed][:, qids]
    upper[fixed] = lower[fixed]
    matrix = sparse.vstack([sparse.eye(count*d), vop, aop], format='csc')
    lows = np.r_[lower.ravel(), np.tile(-velocity, count-1), np.tile(-accel, count-2)]
    highs = np.r_[upper.ravel(), np.tile(velocity, count-1), np.tile(accel, count-2)]
    solver = osqp.OSQP()
    solver.setup(P=2*sparse.eye(count*d, format='csc'), q=-2*desired[:, qids].ravel(),
                 A=matrix, l=lows, u=highs, eps_abs=1e-6, eps_rel=1e-6,
                 max_iter=15000, polishing=True, time_limit=max(.001, time_limit), verbose=False)
    result = solver.solve(raise_error=False)
    if result.x is None or result.info.status_val not in (1, 2):
        return None, result.info.status
    values = matrix@result.x
    if np.any(values < lows-1e-3) or np.any(values > highs+1e-3):
        return None, 'constraint check failed'
    output = desired.copy()
    output[:, qids] = result.x.reshape(count, d)
    output[fixed] = boundary[fixed]
    return output, result.info.status


def refine_window(kin, scene, reference, initial, positions, quaternions, times,
                  hypothesis, movable, budget, *, iterations=8, prefix=2,
                  suffix=0, trust=.7, arm_acceleration=80., finger_acceleration=200.):
    """Collision slack permits recovery from infeasible input, never acceptance.

    Optimize every saved frame in a short window. Boundary samples are fixed
    to previously committed output, so velocity/acceleration cross the seam.
    The nonlinear collision checks and force audit are performed afterwards.
    """
    from v2d_retarget.core.contacts import TIP_NAMES
    started = perf_counter()
    q = initial.copy()
    count, d = len(q), len(movable)
    dimension = count*d
    qids = np.asarray(movable)+7
    vids = np.asarray(movable)+6
    names = [kin.joint_names[i] for i in movable]
    accel_cap = np.array([finger_acceleration if 'hand' in n else arm_acceleration for n in names])
    velocity_cap = .9*kin.robot.model.velocityLimit[vids]
    vop, aop = temporal_operators(times, d)
    rotations = Rotation.from_quat(quaternions, scalar_first=True).as_matrix()
    desired = np.einsum('tij,kj->tki', rotations, hypothesis.points-.002*hypothesis.inward)+positions[:, None]
    tip_ids = [kin.robot.model.getFrameId(n) for n in TIP_NAMES]
    fixed = list(range(prefix)) + list(range(count-suffix, count))
    fixed = np.unique(fixed).astype(int)
    statuses, last_slack = [], None
    q, projection_status = project_temporal(kin, q, reference, times, movable,
        fixed_prefix=prefix, fixed_suffix=suffix, trust=trust,
        fixed_reference=initial, time_limit=max(.001, min(1., budget.remaining)))
    if q is None:
        return initial.copy(), dict(seconds=perf_counter()-started, iterations=0,
            qp_statuses=[f'window temporal projection: {projection_status}'],
            last_linearized_collision_slack_m=None)
    for iteration in range(iterations):
        budget.check()
        blocks, errors, collision_rows, gaps = [], [], [], []
        for t in range(count):
            budget.check()
            kin.configuration.update(q[t])
            block, error = [], []
            for j, frame in enumerate(tip_ids):
                if not hypothesis.active[t, j]:
                    continue
                normal = rotations[t] @ hypothesis.inward[j]
                load = 1. if hypothesis.load_weights is None else hypothesis.load_weights[j]
                weight = load*(np.eye(3)+2*np.outer(normal, normal))/.01
                actual = kin.configuration.data.oMf[frame].translation
                jac = pin.getFrameJacobian(kin.robot.model, kin.configuration.data, frame,
                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, vids]
                block.append(weight@jac)
                error.extend(weight@(actual-desired[t, j]))
            blocks.append(sparse.csc_matrix(np.vstack(block) if block else np.zeros((0, d))))
            errors.extend(error)
            scene.set_state(q[t], positions[t], quaternions[t])
            for contact in scene.contacts(movable):
                jac = contact['jacobian']
                if np.linalg.norm(jac) < 1e-9 or contact['distance'] > .002:
                    continue
                collision_rows.append((t, jac))
                gaps.append(contact['target']-contact['distance'])
        task = sparse.block_diag(blocks, format='csc')
        offset = (q[:, qids]-reference[:, qids]).ravel()
        current = q[:, qids].ravel()
        hessian = task.T@task + .2*sparse.eye(dimension) + .001*(vop.T@vop) + 1e-5*(aop.T@aop)
        gradient = task.T@np.asarray(errors) + .2*offset + .001*(vop.T@(vop@current)) + 1e-5*(aop.T@(aop@current))
        lower = np.maximum(kin.robot.model.lowerPositionLimit[qids]-q[:, qids], -.18)
        upper = np.minimum(kin.robot.model.upperPositionLimit[qids]-q[:, qids], .18)
        lower = np.maximum(lower, reference[:, qids]-trust-q[:, qids])
        upper = np.minimum(upper, reference[:, qids]+trust-q[:, qids])
        lower[fixed] = initial[fixed][:, qids]-q[fixed][:, qids]
        upper[fixed] = lower[fixed]
        constraints = [sparse.eye(dimension, format='csc'), vop, aop]
        lows = [lower.ravel(), np.tile(-velocity_cap, count-1)-vop@current,
                np.tile(-accel_cap, count-2)-aop@current]
        highs = [upper.ravel(), np.tile(velocity_cap, count-1)-vop@current,
                 np.tile(accel_cap, count-2)-aop@current]
        collisions = len(gaps)
        if collisions:
            row, col, data = [], [], []
            for i, (t, jac) in enumerate(collision_rows):
                row.extend([i]*d)
                col.extend(t*d+np.arange(d))
                data.extend(jac)
            collision_matrix = sparse.csc_matrix((data, (row, col)), shape=(collisions, dimension))
            matrix = sparse.vstack([
                sparse.hstack([sparse.vstack(constraints), sparse.csc_matrix((sum(c.shape[0] for c in constraints), collisions))]),
                sparse.hstack([collision_matrix, sparse.eye(collisions)]),
                sparse.hstack([sparse.csc_matrix((collisions, dimension)), sparse.eye(collisions)])
            ], format='csc')
            lows.extend([np.asarray(gaps), np.zeros(collisions)])
            highs.extend([np.full(collisions, np.inf), np.full(collisions, np.inf)])
            hessian = sparse.block_diag([hessian, 2e6*sparse.eye(collisions)], format='csc')
            gradient = np.r_[gradient, np.full(collisions, 100.)]
        else:
            matrix = sparse.vstack(constraints, format='csc')
        solver = osqp.OSQP()
        solver.setup(P=sparse.triu(2*hessian, format='csc'), q=2*gradient, A=matrix,
                     l=np.concatenate(lows), u=np.concatenate(highs), verbose=False,
                     eps_abs=1e-5, eps_rel=1e-5, max_iter=3000, polishing=False,
                     time_limit=max(.001, min(2., budget.remaining)))
        result = solver.solve(raise_error=False)
        statuses.append(result.info.status)
        if result.x is None or result.info.status_val not in (1, 2):
            break
        # A trust-region step is followed by fresh nonlinear geometry next
        # iteration. Acceptance is based on a separate dense audit, not this QP.
        step = result.x[:dimension].reshape(count, d)
        q[:, qids] += step
        q[fixed] = initial[fixed]
        last_slack = float(np.max(result.x[dimension:])) if collisions else 0.
        if np.max(np.abs(step)) < 2e-4:
            break
    return q, dict(seconds=perf_counter()-started, iterations=len(statuses),
                   qp_statuses=statuses, last_linearized_collision_slack_m=last_slack)
