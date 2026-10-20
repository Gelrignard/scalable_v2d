"""A geometry-only initial guess; never an accepted/exported correction.

The coupled horizon must subsequently enforce mechanics, torque and continuity.
This helper exists because the supplied retargeting can start inside the object,
where a nonlinear contact solve has a poor local initialization.
"""
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def initialize_contacts(ref, robot, scene, *, evaluations=70):
    started = perf_counter()
    q = ref.joint_reference.copy()
    positions = np.column_stack([np.interp(ref.times, ref.observation_times, ref.observed_positions[:, j]) for j in range(3)])
    rotations = Slerp(ref.observation_times, Rotation.from_quat(ref.observed_wxyz, scalar_first=True))(ref.times)
    quaternions = rotations.as_quat(scalar_first=True)
    ground = np.array([any(c.support_z is not None and c.active[k] for c in ref.contacts) for k in range(len(ref.times))])
    positions[ground] = ref.initial_position
    quaternions[ground] = ref.initial_wxyz
    matrices = Rotation.from_quat(quaternions, scalar_first=True).as_matrix()
    com = scene.model.body_ipos[scene.book_body]
    pair_count = scene.model.ngeom**2
    history = []
    first = None
    for k in range(1, len(ref.times)):
        active = [c for c in ref.contacts if c.robot_index >= 0 and c.active[k]]
        if not active: continue
        if first is None: first = k
        ids = np.array([c.robot_index for c in active])
        targets = positions[k]+np.array([matrices[k]@c.center for c in active])
        reference = ref.joint_reference[k]
        full = scene.fixed_q.copy()
        full[:3], full[3:7] = ref.base[k, :3], ref.base[k, [4, 5, 6, 3]]
        cache = {}
        def evaluate(angles):
            if 'angles' in cache and np.array_equal(angles, cache['angles']):
                return cache['residual'], cache['jacobian']
            points = np.asarray(robot.kinematics(angles, ref.base[k])).T[ids]
            jac = np.asarray(robot.jacobian(angles, ref.base[k])).reshape(robot.count, 3, robot.n)[ids].reshape(-1, robot.n)
            full[7+scene.movable] = angles
            scene.set_state(full, positions[k]-matrices[k]@com, quaternions[k])
            gaps = np.zeros(pair_count)
            rows = np.zeros((pair_count, robot.n))
            for contact, row in zip(scene.data.contact, scene.contacts(scene.movable)):
                if contact.dist >= 0 or np.linalg.norm(row['jacobian']) < 1e-10: continue
                index = int(contact.geom1)*scene.model.ngeom+int(contact.geom2)
                if -contact.dist > gaps[index]:
                    gaps[index] = -contact.dist
                    rows[index] = -row['jacobian']
            residual = np.r_[100*(points-targets).ravel(), .15*(angles-reference), 500*gaps]
            matrix = sparse.csr_matrix(np.vstack((100*jac, .15*np.eye(robot.n), 500*rows)))
            cache.update(angles=angles.copy(), residual=residual, jacobian=matrix)
            return residual, matrix
        initial = reference if k == first else q[k-1]
        solution = least_squares(lambda z: evaluate(z)[0], np.clip(initial, robot.lower+1e-8, robot.upper-1e-8),
            jac=lambda z: evaluate(z)[1], bounds=(robot.lower, robot.upper), max_nfev=evaluations,
            tr_solver='lsmr', ftol=1e-7, xtol=1e-7, gtol=1e-6)
        q[k] = solution.x
        gap = np.linalg.norm(np.asarray(robot.kinematics(q[k], ref.base[k])).T[ids]-targets, axis=1)
        evaluate(q[k])
        contacts = scene.contacts()
        penetration = max((max(0., -c['distance']) for c in contacts), default=0.)
        worst = min(contacts, key=lambda c: c['distance'])['pair'] if contacts else ''
        history.append(dict(knot=k, time=float(ref.times[k]), max_contact_error_m=float(gap.max()),
                            max_penetration_m=float(penetration), worst_pair=worst,
                            evaluations=solution.nfev, optimizer_status=int(solution.status)))
    if first is not None:
        # Blend an initial-guess correction into approach; do not alter q(0).
        onset = ref.times[first]
        delta = q[first]-ref.joint_reference[first]
        for k in range(1, first):
            u = np.clip((ref.times[k]-max(ref.times[0], onset-.8))/.8, 0, 1)
            q[k] += u*u*(3-2*u)*delta
    print(f'Contact initialization: {perf_counter()-started:.2f}s; worst active-point residual '
          f'{1000*max([h["max_contact_error_m"] for h in history], default=0):.2f}mm', flush=True)
    return dict(q=q, position=positions, wxyz=quaternions), dict(seconds=perf_counter()-started, knots=history,
        purpose='Initial guess only; no mechanical feasibility or execution claim')
