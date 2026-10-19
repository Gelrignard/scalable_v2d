"""Collision-aware fingertip realization on the supported real G1 geometry."""
import numpy as np
from scipy.optimize import least_squares
from scipy import sparse
from scipy.spatial.transform import Rotation



def solve_frame(scene, initial, targets, position, quaternion, *, reference=None, evaluations=100,
                collision_weight=300., posture_weight=.1, clearance=0., tangent_weight=100., lock_fingers=False):
    columns = np.flatnonzero(~scene.hand) if lock_fingers else np.arange(len(scene.movable))
    indices = scene.movable[columns]+7
    reference = initial if reference is None else reference
    q = initial.copy()
    ngeom = scene.model.ngeom
    dimension = len(indices)
    lower = scene.kin.robot.model.lowerPositionLimit[indices]
    upper = scene.kin.robot.model.upperPositionLimit[indices]
    cache = {}
    rotation = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    weights = np.diag([tangent_weight, 100., tangent_weight]) @ rotation.T

    def evaluate(x):
        if 'x' in cache and np.array_equal(x, cache['x']):
            return cache['r'], cache['j']
        q[indices] = x
        scene.set_state(q, position, quaternion)
        tips, jac = scene.tips_and_jacobian()
        jac = jac[:,:,columns]
        # One stable row per geom pair; take its deepest contact manifold point.
        gaps = np.zeros(ngeom*ngeom)
        jacs = np.zeros((ngeom*ngeom, dimension))
        for row, c in zip(scene.contacts(scene.movable), scene.data.contact):
            index = int(c.geom1)*ngeom+int(c.geom2)
            violation = max(0., clearance-c.dist)
            if violation > gaps[index] and np.linalg.norm(row['jacobian']) > 1e-12:
                gaps[index] = violation
                jacs[index] = -row['jacobian'][columns]
        residual = np.r_[((tips-targets)@weights.T).ravel(), posture_weight*(x-reference[indices]), collision_weight*gaps]
        matrix = sparse.csr_matrix(np.vstack((np.einsum('ab,kbj->kaj',weights,jac).reshape(-1, dimension), posture_weight*np.eye(dimension), collision_weight*jacs)))
        cache.update(x=x.copy(), r=residual, j=matrix)
        return residual, matrix

    result = least_squares(lambda x: evaluate(x)[0], np.clip(initial[indices], lower+1e-8, upper-1e-8),
                           jac=lambda x: evaluate(x)[1], bounds=(lower, upper), max_nfev=evaluations,
                           ftol=1e-8, xtol=1e-8, gtol=1e-7, tr_solver='lsmr')
    q[indices] = result.x
    scene.set_state(q, position, quaternion)
    tips, _ = scene.tips_and_jacobian()
    return q.copy(), dict(max_tip_error_mm=1000*float(np.max(np.linalg.norm(tips-targets, axis=1))),
                          max_penetration_mm=1000*max([max(0., -c.dist) for c in scene.data.contact], default=0.),
                          evaluations=result.nfev, cost=result.cost)
