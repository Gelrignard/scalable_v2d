"""Sparse, local robot-command refinement with C2 splines and torque limits.

The object observations are never differentiated. Robot mass matrices, the
known supported-base path and planned contact forces define a frozen-model
torque constraint. Contact Jacobians preserve the existing grasp locally.
This single linearization still needs nonlinear audits and physical replay.
"""
from time import perf_counter
import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


def spline_operators(times, stride=3):
    times = np.asarray(times, float)
    if len(times)<5 or np.any(np.diff(times)<=0):
        raise ValueError('Spline times must be strictly increasing with at least five samples')
    knots = np.r_[np.repeat(times[0], 4), times[stride:-1:stride], np.repeat(times[-1], 4)]
    basis = BSpline.design_matrix(times, knots, 3).tocsc()
    n = basis.shape[1]
    denominator = knots[4:n+3]-knots[1:n]
    first = sparse.diags([-3/denominator, 3/denominator], [0, 1], shape=(n-1, n), format='csc')
    reduced = knots[1:-1]
    denominator2 = reduced[3:n+1]-reduced[1:n-1]
    second = sparse.diags([-2/denominator2, 2/denominator2], [0, 1], shape=(n-2, n-1), format='csc')
    velocity = BSpline.design_matrix(times, reduced, 2).tocsc()@first
    acceleration = BSpline.design_matrix(times, knots[2:-2], 1).tocsc()@second@first
    return knots, basis, velocity, acceleration


def frozen_prefix_coefficients(knots, coefficients, boundary):
    """Fix every cubic basis whose support starts before the boundary.

    The remaining bases start at/after an ordinary interior knot and have
    zero value, velocity and acceleration at that knot. This preserves C2
    and the entire earlier curve exactly, rather than as solver equalities.
    """
    knots, coefficients = np.asarray(knots), np.asarray(coefficients)
    if boundary not in knots[4:-4] or np.count_nonzero(knots == boundary) != 1:
        raise ValueError('Frozen boundary must be a simple interior cubic knot')
    fixed_rows = knots[:len(coefficients)] < boundary
    fixed = np.repeat(fixed_rows, coefficients.shape[1])
    constant = np.where(fixed, coefficients.ravel(), 0.)
    return constant, np.flatnonzero(~fixed)


def refine_commands(scene, reference, points, *, stride=3, torque_fraction=.9,
                    trust_rad=.06, contact_tolerance_m=.001, seconds=15., diagnose=False, free_trust_rad=None,
                    continuous_bounds=False, freeze_prefix_frame=None, clearance_obstacle=None,
                    remembered_pairs=()):
    start = perf_counter()
    t = reference['timestamps']; q = reference['q']; count = len(t); width = len(scene.movable)
    knots, basis, velocity, acceleration = spline_operators(t, stride)
    b, v, a = [sparse.kron(operator, sparse.eye(width), format='csc')
               for operator in (basis, velocity, acceleration)]
    nominal = q[:, 7+scene.movable]
    lower = scene.kin.robot.model.lowerPositionLimit[7+scene.movable]
    upper = scene.kin.robot.model.upperPositionLimit[7+scene.movable]
    speed = scene.kin.robot.model.velocityLimit[6+scene.movable]
    initial_velocity = np.asarray(reference.get('initial_velocity', reference['command_velocity'][0]))
    initial_position = np.asarray(reference.get('initial_q', q[0]))[7+scene.movable]
    if np.any(abs(initial_velocity)>speed+1e-10):
        return None, dict(solved=False, status='initial physical state exceeds speed bound',
            max_initial_speed_fraction=float(np.max(abs(initial_velocity)/speed)),
            violating_joint_columns=np.flatnonzero(abs(initial_velocity)>speed+1e-10).tolist(),
            wall_seconds=perf_counter()-start, ready_for_training=False)
    active = reference['contact_active']
    trust = np.full_like(nominal, trust_rad)
    if free_trust_rad is not None:
        if not np.isfinite(free_trust_rad) or free_trust_rad<trust_rad:
            raise ValueError('Inactive-motion trust must be finite and at least the loaded-motion trust')
        trust[~np.any(active,axis=1)] = free_trust_rad
    # Preserve the complete finger shape while any selected contact is loaded.
    trust[np.any(active, axis=1)[:, None] & scene.hand[None, :]] = .0001
    constraints = [b, v, b[:width], v[:width]]
    groups = ['position_trust', 'speed', 'initial_position', 'initial_velocity']
    lows = [np.maximum(lower, nominal-trust).ravel(), np.tile(-speed, count),
            initial_position, initial_velocity]
    highs = [np.minimum(upper, nominal+trust).ravel(), np.tile(speed, count),
             initial_position, initial_velocity]
    if continuous_bounds:
        # Cubic B-spline values are convex combinations of their coefficients;
        # derivative values are convex combinations of derivative coefficients.
        # Bounding these is sufficient over all time, including between frames.
        # It can be more restrictive than necessary and remains opt-in.
        coefficients = basis.shape[1]
        denominator = knots[4:coefficients+3]-knots[1:coefficients]
        derivative = sparse.diags([-3/denominator, 3/denominator], [0, 1],
                                  shape=(coefficients-1, coefficients), format='csc')
        constraints.extend([sparse.eye(coefficients*width, format='csc'),
                            sparse.kron(derivative, sparse.eye(width), format='csc')])
        groups.extend(['continuous_position', 'continuous_speed'])
        lows.extend([np.tile(lower, coefficients), np.tile(-speed, coefficients-1)])
        highs.extend([np.tile(upper, coefficients), np.tile(speed, coefficients-1)])
    root_p = CubicSpline(t, q[:, :3]); root_r = RotationSpline(t, Rotation.from_quat(q[:, 3:7]))
    mass_blocks = []; bias = []; geometry_blocks = []; geometry_low = []; geometry_high = []
    for frame in range(count):
        scene.set_state(q[frame], reference['object_positions'][frame], reference['object_wxyz'][frame])
        scene.data.qvel[scene.active_v] = reference['command_velocity'][frame]
        scene.data.qvel[scene.root_v:scene.root_v+3] = root_p(t[frame], 1)
        scene.data.qvel[scene.root_v+3:scene.root_v+6] = root_r(t[frame], 1)
        mujoco.mj_forward(scene.model, scene.data)
        mass = np.zeros((scene.model.nv, scene.model.nv))
        mujoco.mj_fullM(scene.model, scene.data, mass)
        mass_blocks.append(sparse.csc_matrix(mass[np.ix_(scene.active_v, scene.active_v)]/scene.caps[:, None]))
        fixed_acc = np.zeros(scene.model.nv)
        fixed_acc[scene.root_v:scene.root_v+3] = root_p(t[frame], 2)
        fixed_acc[scene.root_v+3:scene.root_v+6] = root_r(t[frame], 2)
        fixed = scene.data.qfrc_bias[scene.active_v]-scene.data.qfrc_passive[scene.active_v]
        fixed += mass[scene.active_v]@fixed_acc
        rows = []; tolerances = []
        for channel, point in enumerate(points):
            local = reference['robot_local_points'][frame, channel]
            world = scene.data.xpos[point.body]+scene.data.xmat[point.body].reshape(3, 3)@local
            jac = np.zeros((3, scene.model.nv))
            mujoco.mj_jac(scene.model, scene.data, jac, None, world, point.body)
            jac = jac[:, scene.active_v]
            fixed += jac.T@reference['force_ff'][frame, channel]
            if active[frame, channel]:
                rows.extend(jac); tolerances.extend([contact_tolerance_m]*3)
        if rows:
            jac = np.asarray(rows)
            geometry_low.extend(jac@nominal[frame]-tolerances)
            geometry_high.extend(jac@nominal[frame]+tolerances)
            geometry_blocks.append(sparse.csc_matrix(jac))
        else:
            geometry_blocks.append(sparse.csc_matrix((0, width)))
        bias.append(fixed/scene.caps)
    dynamics = sparse.block_diag(mass_blocks, format='csc')@a
    # The frozen prefix already has its own numerical and physical audit.
    # Do not pretend a new linearization can repair that unchanged execution.
    dynamics_begin = 0 if freeze_prefix_frame is None else (freeze_prefix_frame+1)*width
    constraints.append(dynamics[dynamics_begin:])
    groups.append('dynamics')
    lows.append((-torque_fraction-np.asarray(bias).ravel())[dynamics_begin:])
    highs.append((torque_fraction-np.asarray(bias).ravel())[dynamics_begin:])
    # Express contact constraints in centimeters for solver conditioning.
    # Both sides are scaled identically; the feasible physical set is unchanged.
    constraints.append(100*sparse.block_diag(geometry_blocks, format='csc')@b)
    groups.append('contact_geometry')
    lows.append(100*np.asarray(geometry_low)); highs.append(100*np.asarray(geometry_high))
    clearance_audit = None
    if clearance_obstacle is not None:
        from .spline_clearance import clearance_rows
        clearance_matrix, clearance_low, clearance_audit = clearance_rows(
            scene, reference, knots, clearance_obstacle, remembered_pairs)
        constraints.append(100*clearance_matrix); groups.append('inactive_clearance')
        lows.append(100*clearance_low); highs.append(np.full(len(clearance_low), np.inf))
    matrix = sparse.vstack(constraints, format='csc')
    low, high = np.concatenate(lows), np.concatenate(highs)
    hessian = b.T@b+2e-6*(a.T@a)+1e-9*sparse.eye(b.shape[1])
    if freeze_prefix_frame is not None and diagnose:
        raise ValueError('Constraint-isolation mode does not support a frozen prefix')
    if diagnose:
        build_time=perf_counter()-start
        audits = {}
        for name, excluded in [
            ('all', []), ('without_dynamics', ['dynamics']),
            ('without_geometry', ['contact_geometry']),
            ('without_speed', ['speed','continuous_speed']),
            ('position_and_initial_only', ['speed','continuous_speed','dynamics','contact_geometry']),
            ('position_only', ['speed','continuous_speed','dynamics','contact_geometry','initial_velocity'])]:
            selected=[i for i,g in enumerate(groups) if g not in excluded]
            mat=sparse.vstack([constraints[i] for i in selected],format='csc')
            lower=np.concatenate([lows[i] for i in selected])
            upper=np.concatenate([highs[i] for i in selected])
            solver=osqp.OSQP(); before=perf_counter()
            solver.setup(P=sparse.triu(2*hessian,format='csc'),q=-2*np.asarray(b.T@nominal.ravel()).ravel(),
                A=mat,l=lower,u=upper,eps_abs=1e-5,eps_rel=1e-5,max_iter=20000,
                time_limit=seconds,verbose=False,polishing=True)
            sol=solver.solve(raise_error=False)
            audit=dict(status=sol.info.status,wall_seconds=perf_counter()-before,
                       primal_residual=float(sol.info.prim_res), dual_residual=float(sol.info.dual_res))
            sizes=[constraints[i].shape[0] for i in selected]
            if sol.info.status_val in (1,2):
                val=mat@sol.x
                audit['constraint_violation']=float(max(0.,np.max(lower-val),np.max(val-upper)))
            if sol.info.status_val in (3,4) and sol.prim_inf_cert is not None:
                certificate=np.asarray(sol.prim_inf_cert)
                if np.isfinite(certificate).all():
                    certificate/=max(np.max(abs(certificate)),1e-12)
                    positive=certificate>0; negative=certificate<0
                    score=float(upper[positive]@certificate[positive]+lower[negative]@certificate[negative])
                    blocks=np.split(certificate,np.cumsum(sizes)[:-1])
                    audit['certificate']=dict(transpose_residual=float(np.max(abs(mat.T@certificate))),
                        bound_dot=score, note='Approximate OSQP certificate; tolerances/status matter, not a proof for the nonlinear robot',
                        blocks={groups[i]:dict(l1=float(abs(block).sum()),largest_row=int(np.argmax(abs(block))),
                                               max_abs=float(np.max(abs(block))))
                                for i,block in zip(selected,blocks)})
            audits[name]=audit
        return None, dict(scope='Read-only constraint isolation; relaxed problems never exported or replayed',
            diagnostic=True,build_wall_seconds=build_time,wall_seconds=perf_counter()-start,
            seconds_per_case=seconds,free_trust_rad=free_trust_rad,cases=audits,ready_for_training=False)
    solver = osqp.OSQP()
    build_time = perf_counter()-start
    from scipy.sparse.linalg import lsqr
    initial_coefficients = np.column_stack([lsqr(basis, column, atol=1e-10, btol=1e-10)[0]
                                           for column in nominal.T])
    constant = np.zeros(b.shape[1]); variables = np.arange(b.shape[1])
    if freeze_prefix_frame is not None:
        if not np.array_equal(reference.get('spline_knots'), knots):
            raise ValueError('Frozen prefix needs the identical saved spline basis')
        coefficients = reference['spline_coefficients']
        if coefficients.shape != initial_coefficients.shape:
            raise ValueError('Frozen spline coefficient dimensions differ')
        constant, variables = frozen_prefix_coefficients(knots, coefficients, t[freeze_prefix_frame])
        # Verify input metadata really represents the prefix being preserved.
        end = freeze_prefix_frame+1
        np.testing.assert_allclose(basis[:end]@coefficients, nominal[:end], atol=1e-11, rtol=0)
        np.testing.assert_allclose(velocity[:end]@coefficients, reference['command_velocity'][:end],
                                   atol=1e-10, rtol=0)
    reduced = matrix[:, variables]
    offset = matrix@constant
    low_reduced, high_reduced = low-offset, high-offset
    rows = np.asarray(abs(reduced).sum(axis=1)).ravel() > 0
    if np.max(np.maximum(low_reduced[~rows], -high_reduced[~rows]), initial=0.) > 1e-4:
        raise ValueError('Fixed prefix violates a retained constraint; cannot optimize it away')
    linear = -2*np.asarray(b.T@nominal.ravel()).ravel()
    solver.setup(P=sparse.triu(2*hessian[variables][:, variables], format='csc'),
        q=(linear+2*hessian@constant)[variables], A=reduced[rows].tocsc(),
        l=low_reduced[rows], u=high_reduced[rows], eps_abs=1e-5, eps_rel=1e-5,
        max_iter=20000, time_limit=seconds, verbose=False, polishing=True)
    solver.warm_start(x=initial_coefficients.ravel()[variables])
    solution = solver.solve(raise_error=False)
    solved = solution.info.status_val in (1, 2) and solution.x is not None and np.isfinite(solution.x).all()
    audit = dict(scope='single frozen dynamics/Jacobian linearization; nonlinear replay required',
        status=solution.info.status, solved=bool(solved), stride=stride,
        iterations=int(solution.info.iter),
        primal_residual=float(solution.info.prim_res), dual_residual=float(solution.info.dual_res),
        torque_fraction_limit=torque_fraction, trust_rad=trust_rad, contact_tolerance_m=contact_tolerance_m,
        free_trust_rad=free_trust_rad, contact_constraint_scale=100.,
        initial_velocity_matches_physical_state=True, initialization='least-squares spline fit',
        continuous_position_speed_bounds=continuous_bounds,
        frozen_prefix_frame=freeze_prefix_frame, optimized_coefficients=int(len(variables)),
        dynamics_scope='entire command' if freeze_prefix_frame is None else 'only frames after frozen prefix',
        clearance_linearization=clearance_audit,
        build_wall_seconds=build_time, wall_seconds=perf_counter()-start, ready_for_training=False)
    if not solved:
        return None, audit
    coefficient_vector = constant.copy(); coefficient_vector[variables] = solution.x
    value = matrix@coefficient_vector
    offsets=np.r_[0,np.cumsum([c.shape[0] for c in constraints])]
    scaled_violation={
        group:float(max(0.,np.max(low[a:b]-value[a:b],initial=0.),np.max(value[a:b]-high[a:b],initial=0.)))
        for group,a,b in zip(groups,offsets[:-1],offsets[1:])}
    physical_violation={group: value/(100. if group in ('contact_geometry', 'inactive_clearance') else 1.)
                        for group,value in scaled_violation.items()}
    audit['solver_scaled_violation_by_group']=scaled_violation
    audit['constraint_violation_by_group']=physical_violation
    violation=max(physical_violation.values())
    audit['constraint_violation'] = violation
    if violation>1e-4:
        audit['solved'] = False
        return None, audit
    result = dict(reference)
    result['q'] = q.copy(); result['q'][:, 7+scene.movable] = (b@coefficient_vector).reshape(count, width)
    result['command_velocity'] = (v@coefficient_vector).reshape(count, width)
    result['command_acceleration'] = (a@coefficient_vector).reshape(count, width)
    coefficients = coefficient_vector.reshape(basis.shape[1], width)
    result['spline_knots'] = knots; result['spline_coefficients'] = coefficients
    if freeze_prefix_frame is not None:
        # Preserve exact floating-point command arrays used by the simulator.
        end = freeze_prefix_frame+1
        for key in ['q', 'command_velocity', 'command_acceleration']:
            np.testing.assert_allclose(result[key][:end], reference[key][:end], atol=1e-9, rtol=0)
            result[key][:end] = reference[key][:end]
    audit['max_joint_change_rad'] = float(np.max(abs(result['q'][:, 7+scene.movable]-nominal)))
    audit['max_frozen_torque_fraction'] = float(np.max(abs(
        dynamics@coefficient_vector+np.asarray(bias).ravel())[dynamics_begin:]))
    audit['max_acceleration_rad_s2'] = float(np.max(abs(result['command_acceleration'])))
    return result, audit
