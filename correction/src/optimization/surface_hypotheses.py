"""Bounded robot-surface hypotheses on the actual MuJoCo convex geometry.

This module solves static geometry only. Acquisition and dynamic execution
are separate gates. Object points are actual collision surfaces, not AABB sites.
"""
from dataclasses import dataclass
from time import perf_counter
import mujoco
import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@dataclass
class SurfacePoint:
    body: int
    geom: int
    local: np.ndarray
    normal_local: np.ndarray
    object_local: np.ndarray
    object_outward: np.ndarray
    object_geom: int
    robot_triangle: np.ndarray
    object_triangle: np.ndarray
    source_channel: int = -1
    evidence_anchor: np.ndarray | None = None


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def body_mesh(model, geom):
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError('Surface hypotheses currently require mesh collision geometry')
    mesh = int(model.geom_dataid[geom])
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start+count].copy()
    rot = Rotation.from_quat(model.geom_quat[geom], scalar_first=True)
    vertices = rot.apply(vertices)+model.geom_pos[geom]
    return trimesh.Trimesh(vertices=vertices, faces=ConvexHull(vertices).simplices, process=False).convex_hull


class SurfaceLibrary:
    def __init__(self, scene):
        self.scene = scene
        self.meshes = {g: body_mesh(scene.model, g) for g in range(scene.model.ngeom)
                       if scene.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
        pieces = [(g, self.meshes[g]) for g in sorted(scene.object_geoms)]
        self.object_triangles = np.concatenate([m.triangles for _, m in pieces])
        self.object_normals = np.concatenate([m.face_normals for _, m in pieces])
        self.object_ids = np.concatenate([np.full(len(m.faces), g, int) for g, m in pieces])
        self.object_halfspaces = [(g, ConvexHull(mesh.vertices).equations) for g, mesh in pieces]
        # Remove faces internal to overlapping convex pieces. This conservative
        # centroid test is followed by full collision checks at candidate poses.
        outward_probe = self.object_triangles.mean(axis=1)+.001*self.object_normals
        exposed = np.ones(len(outward_probe), bool)
        for geom, mesh in pieces:
            equations = ConvexHull(mesh.vertices).equations
            inside = np.all(outward_probe@equations[:, :3].T+equations[:, 3] < -1e-6, axis=1)
            exposed &= ~((self.object_ids != geom) & inside)
        self.exposed = exposed

    def object_patch(self, anchor, preferred_outward, radius=.055, normal_weight=.035):
        closest = trimesh.triangles.closest_point(self.object_triangles,
                                                 np.tile(anchor, (len(self.object_triangles), 1)))
        distance = np.linalg.norm(closest-anchor, axis=1)
        normal = np.asarray(preferred_outward, float)
        normal = normal/max(np.linalg.norm(normal), 1e-12)
        alignment = self.object_normals@normal
        keep = distance <= radius
        # Exposure must be checked at the proposed point. A large triangle can
        # have an internal centroid but an exposed rim near one of its edges.
        probes = closest+.001*self.object_normals
        for geom, equations in self.object_halfspaces:
            ids = np.flatnonzero(keep & (self.object_ids != geom))
            if not len(ids): continue
            inside = np.all(probes[ids]@equations[:, :3].T+equations[:, 3] < -1e-6, axis=1)
            keep[ids[inside]] = False
        if not keep.any():
            raise ValueError('No exposed object collision patch within demonstration neighborhood')
        # Prefer a compatible face but keep its alignment/uncertainty auditable.
        cost = distance+normal_weight*(1-alignment)
        index = np.flatnonzero(keep)[np.argmin(cost[keep])]
        return closest[index], self.object_normals[index], int(self.object_ids[index]), self.object_triangles[index]

    def robot_patch(self, body, object_point_world, inward_world, *, support_plane=False):
        s = self.scene
        rot = s.data.xmat[body].reshape(3, 3)
        target = rot.T@(object_point_world-s.data.xpos[body])
        inward_local = rot.T@inward_world
        candidates = []
        for geom, mesh in self.meshes.items():
            if int(s.model.geom_bodyid[geom]) != body:
                continue
            points = trimesh.triangles.closest_point(mesh.triangles, np.tile(target, (len(mesh.faces), 1)))
            alignment = mesh.face_normals@inward_local
            if support_plane:
                vertex = int(np.argmax(mesh.vertices@inward_local))
                adjacent = np.flatnonzero(np.any(mesh.faces == vertex, axis=1))
                index = adjacent[np.argmax(alignment[adjacent])]
                point = mesh.vertices[vertex]
                candidates.append((np.linalg.norm(point-target), geom, point,
                                   mesh.face_normals[index], mesh.triangles[index]))
                continue
            cost = np.linalg.norm(points-target, axis=1)+.025*(1-alignment)
            keep = alignment >= .2
            if not keep.any():
                continue
            index = np.flatnonzero(keep)[np.argmin(cost[keep])]
            candidates.append((cost[index], geom, points[index], mesh.face_normals[index], mesh.triangles[index]))
        if not candidates:
            raise ValueError('No robot surface faces the selected object region')
        _, geom, local, normal, triangle = min(candidates, key=lambda row: row[0])
        return geom, local, normal, triangle

    def hypothesis(self, raw, frame, family, region_normals):
        s = self.scene
        q, p, quat = raw['q'][frame], raw['object_positions'][frame], raw['object_wxyz'][frame]
        s.set_state(q, p, quat)
        rotation = Rotation.from_quat(quat, scalar_first=True)
        if family not in ('observed_six', 'coherent_four', 'support_six', 'support_any', 'rim_pinch',
                          'centered_four', 'centered_six', 'centered_proximal_four', 'centered_left_proximal_four'):
            raise ValueError('Unknown surface hypothesis family')
        columns = [1, 2, 4, 5] if family in ('coherent_four', 'support_any', 'centered_four',
                    'centered_proximal_four', 'centered_left_proximal_four') else list(range(6))
        result = []
        used_bodies = set()
        for i in columns:
            if family == 'observed_six':
                preference = raw['contact_normals_local'][frame, i]
            elif family == 'rim_pinch' or family.startswith('centered'):
                centers = [np.median(raw['contact_anchors_local'][frame, j:j+3], axis=0) for j in (0, 3)]
                axis = centers[1]-centers[0]
                vertical = rotation.inv().apply([0., 0., 1.])
                axis -= axis@vertical*vertical
                axis /= np.linalg.norm(axis)
                preference = axis*(-1 if i < 3 else 1)
                if family == 'rim_pinch' and i in (0, 3): preference *= -1
            elif family in ('support_six', 'support_any'):
                # A local alternative: underside/rim faces that can push upward.
                preference = rotation.inv().apply([0., 0., -1.])
            else:
                preference = region_normals['left' if i < 3 else 'right']
            anchor = raw['contact_anchors_local'][frame, i].copy()
            if family == 'rim_pinch':
                anchor += rotation.inv().apply([0., 0., -.02])
            if family.startswith('centered'):
                # Bring the grasp's support region toward the projected COM,
                # bounded to 55 mm from the human-derived anchor. This reduces
                # the moment arm rather than increasing arbitrary force caps.
                vertical = rotation.inv().apply([0., 0., 1.])
                shift = s.model.body_ipos[s.book_body]-np.mean(raw['contact_anchors_local'][frame], axis=0)
                shift -= shift@vertical*vertical
                shift *= min(1., .055/max(np.linalg.norm(shift), 1e-12))
                anchor += shift
            op, outward, og, object_triangle = self.object_patch(anchor, preference,
                normal_weight=.08 if family.startswith(('support', 'centered')) or family == 'rim_pinch' else .035)
            body = int(s.model.site_bodyid[s.tip_ids[i]])
            if (family=='centered_proximal_four' and i in (1,4)) or (family=='centered_left_proximal_four' and i==1):
                body=s.model.body(('left' if i<3 else 'right')+'_hand_index_0_link').id
            bodies = [body]
            if family == 'support_any':
                prefix = ('left' if i < 3 else 'right')+'_hand_'
                bodies = sorted({int(s.model.geom_bodyid[g]) for g in self.meshes
                                 if s.model.body(int(s.model.geom_bodyid[g])).name.startswith(prefix)}-used_bodies)
            options = []
            target, inward = p+rotation.apply(op), rotation.apply(-outward)
            for b in bodies:
                geom, local, normal, triangle = self.robot_patch(b, target, inward)
                rb = s.data.xmat[b].reshape(3, 3)
                cost = np.linalg.norm(s.data.xpos[b]+rb@local-target)+.015*(1-inward@rb@normal)
                options.append((cost, b, geom, local, normal, triangle))
            if not options: raise ValueError('No unused robot surface in the local hand')
            _, body, geom, local, normal, robot_triangle = min(options, key=lambda x: x[0])
            used_bodies.add(body)
            result.append(SurfacePoint(body, geom, local, normal, op, outward, og, robot_triangle,
                                       object_triangle, i, raw['contact_anchors_local'][frame, i].copy()))
        return result

    def rechart(self, points, position, quaternion):
        """Refresh a robot surface chart at the current q, keeping object intent."""
        from copy import deepcopy
        result = deepcopy(points)
        r = Rotation.from_quat(quaternion, scalar_first=True)
        for point in result:
            geom, local, normal, triangle = self.robot_patch(point.body,
                position+r.apply(point.object_local), r.apply(-point.object_outward), support_plane=True)
            point.geom, point.local, point.normal_local, point.robot_triangle = geom, local, normal, triangle
        return result


def contact_kinematics(scene, points):
    position, normal, jacobian, normal_jacobian = [], [], [], []
    for point in points:
        rotation = scene.data.xmat[point.body].reshape(3, 3)
        p = scene.data.xpos[point.body]+rotation@point.local
        n = rotation@point.normal_local
        jp, jr = np.zeros((3, scene.model.nv)), np.zeros((3, scene.model.nv))
        mujoco.mj_jac(scene.model, scene.data, jp, jr, p, point.body)
        position.append(p); normal.append(n)
        jacobian.append(jp[:, scene.active_v])
        normal_jacobian.append(-skew(n)@jr[:, scene.active_v])
    return tuple(np.asarray(v) for v in (position, normal, jacobian, normal_jacobian))


class _Deadline(Exception):
    pass


def solve_geometry(scene, points, nominal, position, quaternion, *, initial=None,
                   seconds=12., max_evaluations=120):
    """One bounded static solve; no claim of a collision-free approach."""
    started = perf_counter()
    q = np.array(nominal if initial is None else initial).copy()
    idx = 7+scene.movable
    lower = scene.kin.robot.model.lowerPositionLimit[idx]
    upper = scene.kin.robot.model.upperPositionLimit[idx]
    neighborhood = np.where(scene.hand, 1., .5)
    lower = np.maximum(lower, nominal[idx]-neighborhood)
    upper = np.minimum(upper, nominal[idx]+neighborhood)
    if np.any(lower >= upper):
        raise ValueError('Nominal pose lies outside the permitted local joint neighborhood')
    rotation = Rotation.from_quat(quaternion, scalar_first=True)
    targets = position+rotation.apply(np.array([p.object_local for p in points]))
    normals = rotation.apply(-np.array([p.object_outward for p in points]))
    n, pairs = len(idx), scene.model.ngeom**2
    cache, best = {}, {'cost': np.inf, 'x': np.clip(q[idx], lower+1e-8, upper-1e-8)}

    def evaluate(x):
        if 'x' in cache and np.array_equal(x, cache['x']):
            return cache['r'], cache['j']
        if perf_counter()-started > seconds:
            raise _Deadline()
        q[idx] = x
        scene.set_state(q, position, quaternion)
        p, normal, jp, jn = contact_kinematics(scene, points)
        gaps, jc = np.zeros(pairs), np.zeros((pairs, n))
        for row, contact in zip(scene.contacts(scene.movable), scene.data.contact):
            key = int(contact.geom1)*scene.model.ngeom+int(contact.geom2)
            violation = max(0., -.00025-contact.dist)
            if violation > gaps[key] and np.linalg.norm(row['jacobian']) > 1e-10:
                gaps[key], jc[key] = violation, -row['jacobian']
        residual = np.r_[100*(p-targets).ravel(), 2.*(normal-normals).ravel(),
                         .15*(x-nominal[idx]), 500*gaps]
        jac = sparse.csr_matrix(np.vstack((100*jp.reshape(-1, n), 2*jn.reshape(-1, n),
                                          .15*np.eye(n), 500*jc)))
        cost = float(residual@residual)
        if cost < best['cost']:
            best.update(cost=cost, x=x.copy())
        cache.update(x=x.copy(), r=residual, j=jac)
        return residual, jac

    status, evaluations = 'not_started', 0
    try:
        fit = least_squares(lambda x: evaluate(x)[0], best['x'], jac=lambda x: evaluate(x)[1],
                            bounds=(lower, upper), max_nfev=max_evaluations, tr_solver='lsmr',
                            ftol=1e-7, xtol=1e-7, gtol=1e-6)
        status, evaluations = str(fit.message), int(fit.nfev)
    except _Deadline:
        status = 'wall_budget_exhausted'
    q[idx] = best['x']
    scene.set_state(q, position, quaternion)
    p, normal, _, _ = contact_kinematics(scene, points)
    gaps = np.linalg.norm(p-targets, axis=1)
    cosines = np.sum(normal*normals, axis=1)
    collision_rows = scene.contacts(scene.movable)
    robot_depth = max([max(0., -row['distance']) for row in collision_rows
                       if np.linalg.norm(row['jacobian']) > 1e-10], default=0.)
    return q, dict(status=status, evaluations=evaluations, wall_seconds=perf_counter()-started,
                   max_gap_m=float(gaps.max()), contact_gaps_m=gaps.tolist(),
                   min_normal_cosine=float(cosines.min()), normal_cosines=cosines.tolist(),
                   max_robot_penetration_m=float(robot_depth),
                   max_joint_change_rad=float(np.max(abs(q[idx]-nominal[idx]))),
                   collision_pairs=[row['pair'] for row in collision_rows if row['distance'] < -.002])
