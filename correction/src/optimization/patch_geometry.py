"""Static robot contact optimization with small surface-chart coordinates.

Both contact points may slide on their chosen collision triangles. This avoids
requiring one arbitrary human point to coincide with one arbitrary robot point.
No contact is moved off its actual triangle; dynamic sticking is a later gate.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco
from .surface_hypotheses import contact_kinematics, body_mesh
from ..dynamics.collision_distance import distance_row


def triangle_point(triangle, uv):
    a, b, c = np.asarray(triangle)
    u, v = uv
    direction = (1-v)*(b-a)+v*(c-a)
    return a+u*direction, np.column_stack((direction, u*(c-b)))


def triangle_coordinates(triangle, point):
    a, b, c = np.asarray(triangle)
    weights = np.linalg.lstsq(np.column_stack((b-a, c-a)), point-a, rcond=None)[0]
    u = float(weights.sum())
    return np.clip([u, weights[1]/u if u > 1e-10 else .5], 1e-8, 1-1e-8)


def normal_cone_gap(vertices, point, outward):
    """Distance to the supporting plane in a unit outward direction.

    Zero is valid for a face, edge or vertex. A single triangle face normal
    would falsely reject legitimate corner contacts on a convex fingertip.
    """
    direction = np.asarray(outward)/np.linalg.norm(outward)
    support = np.asarray(vertices)[np.argmax(np.asarray(vertices)@direction)]
    return float(direction@(support-point)), support


class _Deadline(Exception):
    pass


def matches_selected_patch(scene, points, geom1, geom2, row, object_position, object_rotation,
                           radius=.03, normal_cosine=.5):
    """Use the controller's local patch meaning across convex-piece seams.

    Same body alone is insufficient: require both closest surface points
    nearby and a compatible inward normal. This does not certify point-wrench
    accuracy across a finite patch.
    """
    first = geom1 in scene.object_geoms
    second = geom2 in scene.object_geoms
    if first == second:
        return False
    robot_geom = geom2 if first else geom1
    body = int(scene.model.geom_bodyid[robot_geom])
    robot_point, object_point = row['closest_points'][[1,0] if first else [0,1]]
    inward = row['normal_1_to_2']*(-1 if first else 1)
    for point in points:
        if point.body != body:
            continue
        rp = scene.data.xpos[body]+scene.data.xmat[body].reshape(3,3)@point.local
        op = object_position+object_rotation@point.object_local
        expected = -object_rotation@point.object_outward
        if (np.linalg.norm(robot_point-rp)<=radius and np.linalg.norm(object_point-op)<=radius
                and inward@expected>=normal_cosine):
            return True
    return False


def solve_patch_geometry(scene, points, nominal, position, quaternion, *, seconds=12., max_evaluations=250,
                         initial_q=None, non_contact_clearance_m=0., allow_patch_neighbours=False):
    started = perf_counter()
    points = deepcopy(points)
    q = (nominal if initial_q is None else initial_q).copy(); idx = 7+scene.movable
    n, count = len(idx), len(points)
    neighborhood = np.where(scene.hand, 1., .5)
    lo = np.maximum(scene.kin.robot.model.lowerPositionLimit[idx], nominal[idx]-neighborhood)
    hi = np.minimum(scene.kin.robot.model.upperPositionLimit[idx], nominal[idx]+neighborhood)
    if np.any(lo >= hi): raise ValueError('Invalid local joint neighborhood')
    lower, upper = np.r_[lo, np.zeros(4*count)], np.r_[hi, np.ones(4*count)]
    robot0, object0 = np.array([p.local for p in points]), np.array([p.object_local for p in points])
    uv0 = np.array([np.r_[triangle_coordinates(p.robot_triangle, p.local),
                         triangle_coordinates(p.object_triangle, p.object_local)] for p in points])
    initial = np.r_[np.clip(q[idx], lo+1e-8, hi-1e-8), uv0.ravel()]
    rotation = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    wanted_normals = -np.array([p.object_outward for p in points])@rotation.T
    robot_meshes = {p.geom: body_mesh(scene.model, p.geom) for p in points}
    size, pairs = len(initial), scene.model.ngeom**2
    best, cache = {'cost': np.inf, 'x': initial.copy()}, {}
    remembered=set()
    allowed={tuple(sorted((p.geom,p.object_geom))) for p in points}

    def place(x):
        q[idx] = x[:n]
        derivatives = []
        for i, p in enumerate(points):
            uv = x[n+4*i:n+4*i+4]
            p.local, dr = triangle_point(p.robot_triangle, uv[:2])
            p.object_local, do = triangle_point(p.object_triangle, uv[2:])
            derivatives.append((dr, do))
        scene.set_state(q, position, quaternion)
        return derivatives

    def evaluate(x):
        if 'x' in cache and np.array_equal(x, cache['x']): return cache['r'], cache['j']
        if perf_counter()-started > seconds: raise _Deadline()
        derivatives = place(x)
        p, normals, jp, jn = contact_kinematics(scene, points)
        target = position+np.array([point.object_local for point in points])@rotation.T
        jc = np.zeros((3*count, size)); jnormal = np.zeros((count, size))
        normal_gaps = np.zeros(count)
        jc[:, :n] = 100*jp.reshape(-1, n)
        prior, jprior = np.zeros(6*count), np.zeros((6*count, size))
        limits, jlimits = np.zeros(3*count), np.zeros((3*count, size))
        for i, (point, (dr, do)) in enumerate(zip(points, derivatives)):
            col = n+4*i; row = slice(3*i, 3*i+3)
            body_r = scene.data.xmat[point.body].reshape(3, 3)
            jc[row, col:col+2] = 100*body_r@dr
            jc[row, col+2:col+4] = -100*rotation@do
            gap, support = normal_cone_gap(robot_meshes[point.geom].vertices, point.local,
                                            body_r.T@wanted_normals[i])
            normal_gaps[i] = 150*gap
            support_world = scene.data.xpos[point.body]+body_r@support
            js = np.zeros((3, scene.model.nv))
            mujoco.mj_jac(scene.model, scene.data, js, None, support_world, point.body)
            jnormal[i, :n] = 150*wanted_normals[i]@(js[:, scene.active_v]-jp[i])
            jnormal[i, col:col+2] = -150*wanted_normals[i]@body_r@dr
            for j, (delta, derivative, radius) in enumerate([(point.local-robot0[i], dr, .02),
                                                           (point.object_local-object0[i], do, .035)]):
                k, c = 6*i+3*j, col+2*j
                prior[k:k+3], jprior[k:k+3, c:c+2] = 3*delta, 3*derivative
                distance = np.linalg.norm(delta)
                if distance > radius:
                    limits[3*i+j] = 500*(distance-radius)
                    jlimits[3*i+j, c:c+2] = 500*delta@derivative/distance
            if point.evidence_anchor is not None:
                delta = point.object_local-point.evidence_anchor
                distance = np.linalg.norm(delta)
                if distance > .1:
                    limits[3*i+2] = 500*(distance-.1)
                    jlimits[3*i+2, col+2:col+4] = 500*delta@do/distance
        collision, jcollision = np.zeros(pairs), np.zeros((pairs, size))
        if non_contact_clearance_m>0:
            remembered.update(tuple(sorted((int(c.geom1),int(c.geom2)))) for c in scene.data.contact)
            for a,b in remembered:
                row=distance_row(scene,a,b)
                if row is None: continue
                selected = ((a,b) in allowed or (allow_patch_neighbours and
                    matches_selected_patch(scene,points,a,b,row,position,rotation)))
                clearance_target=-.00025 if selected else non_contact_clearance_m
                key=a*scene.model.ngeom+b
                violation=max(0.,clearance_target-row['distance'])
                collision[key]=violation
                if violation: jcollision[key,:n]=-row['jacobian']
        else:
            for row, contact in zip(scene.contacts(scene.movable), scene.data.contact):
                key = int(contact.geom1)*scene.model.ngeom+int(contact.geom2)
                violation = max(0., -.00025-contact.dist)
                if violation > collision[key] and np.linalg.norm(row['jacobian']) > 1e-10:
                    collision[key] = violation
                    jcollision[key, :n] = -row['jacobian']
        posture_j = np.zeros((n, size)); posture_j[:, :n] = .15*np.eye(n)
        residual = np.r_[100*(p-target).ravel(), normal_gaps,
                         .15*(x[:n]-nominal[idx]), prior, limits, 500*collision]
        jacobian = sparse.csr_matrix(np.vstack((jc, jnormal, posture_j, jprior, jlimits, 500*jcollision)))
        cost = float(residual@residual)
        if cost < best['cost']: best.update(cost=cost, x=x.copy())
        cache.update(x=x.copy(), r=residual, j=jacobian)
        return residual, jacobian

    status, evaluations = 'not_started', 0
    try:
        fit = least_squares(lambda x: evaluate(x)[0], initial, jac=lambda x: evaluate(x)[1],
            bounds=(lower, upper), max_nfev=max_evaluations, tr_solver='lsmr',
            ftol=1e-7, xtol=1e-7, gtol=1e-6)
        status, evaluations = str(fit.message), int(fit.nfev)
    except _Deadline: status = 'wall_budget_exhausted'
    place(best['x'])
    p, normal, _, _ = contact_kinematics(scene, points)
    target = position+np.array([point.object_local for point in points])@rotation.T
    gaps = np.linalg.norm(p-target, axis=1)
    cosines = np.sum(normal*wanted_normals, axis=1)
    cone_gaps = [normal_cone_gap(robot_meshes[p.geom].vertices, p.local,
                  scene.data.xmat[p.body].reshape(3, 3).T@wanted_normals[i])[0] for i, p in enumerate(points)]
    collision = scene.contacts(scene.movable)
    depth = max([max(0., -row['distance']) for row in collision if np.linalg.norm(row['jacobian']) > 1e-10], default=0.)
    object_shift = np.linalg.norm(np.array([point.object_local for point in points])-object0, axis=1)
    robot_shift = np.linalg.norm(np.array([point.local for point in points])-robot0, axis=1)
    evidence_shift = [float(np.linalg.norm(p.object_local-p.evidence_anchor))
                      for p in points if p.evidence_anchor is not None]
    unwanted=[]; patch_neighbours=[]
    if non_contact_clearance_m>0:
        remembered.update(tuple(sorted((int(c.geom1),int(c.geom2)))) for c in scene.data.contact)
        for a,b in remembered-allowed:
            row=distance_row(scene,a,b)
            if row is not None and allow_patch_neighbours and matches_selected_patch(
                    scene,points,a,b,row,position,rotation):
                patch_neighbours.append(dict(pair=row['pair'],distance_m=float(row['distance'])))
                continue
            if row is not None and row['distance']<non_contact_clearance_m:
                unwanted.append(dict(pair=row['pair'],distance_m=float(row['distance'])))
    return q.copy(), points, dict(status=status, evaluations=evaluations, wall_seconds=perf_counter()-started,
        max_gap_m=float(gaps.max()), contact_gaps_m=gaps.tolist(),
        min_normal_cosine=float(cosines.min()), normal_cosines=cosines.tolist(),
        normal_cone_gaps_m=cone_gaps, max_normal_cone_gap_m=float(max(cone_gaps)),
        max_robot_penetration_m=float(depth), max_joint_change_rad=float(np.max(abs(q[idx]-nominal[idx]))),
        object_patch_shift_m=object_shift.tolist(), robot_patch_shift_m=robot_shift.tolist(),
        patch_neighborhood_satisfied=bool(object_shift.max() <= .0351 and robot_shift.max() <= .0201),
        evidence_displacement_m=evidence_shift,
        evidence_region_satisfied=bool(len(evidence_shift) == count and max(evidence_shift) <= .1001),
        non_contact_clearance_m=non_contact_clearance_m, unwanted_near_pairs=unwanted,
        allow_patch_neighbours=allow_patch_neighbours, selected_patch_neighbour_pairs=patch_neighbours,
        collision_pairs=[row['pair'] for row in collision if row['distance'] < -.002])
