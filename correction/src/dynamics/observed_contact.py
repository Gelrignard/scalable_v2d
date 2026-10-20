"""Actual collision-patch observation and conditional object-wrench allocation."""
import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from .environment_contact import environment_contacts
from .sparse_forces import contact_basis, cross_matrix
from ..optimization.surface_hypotheses import SurfaceLibrary


def observe_patches(scene, points, chart):
    """No force is inferred from distance. Record gap, geometry and load separately."""
    s = scene; d = s.data; m = s.model
    rotation = Rotation.from_quat(d.qpos[s.book_q+3:s.book_q+7], scalar_first=True)
    anchors = d.qpos[s.book_q:s.book_q+3]+rotation.apply(chart['object_local'])
    inward = rotation.apply(chart['inward'])
    positions, normals, jacobians, rotation_jacobians = [], [], [], []
    for i, point in enumerate(points):
        r = d.xmat[point.body].reshape(3, 3)
        position = d.xpos[point.body]+r@chart['robot_local'][i]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, position, point.body)
        positions.append(position); normals.append(r@chart['robot_normal'][i])
        jacobians.append(jp[:, s.active_v]); rotation_jacobians.append(jr[:, s.active_v])
    positions, normals = np.array(positions), np.array(normals)
    measured = np.zeros_like(positions); touching = np.zeros(len(points), bool)
    signed_distance = np.full(len(points), np.inf)
    closest = np.zeros((len(points), 6))
    for i, point in enumerate(points):
        signed_distance[i] = mujoco.mj_geomDistance(m, d, int(chart['robot_geom'][i]),
            int(chart['object_geom'][i]), 1., closest[i])
        for ci, contact in enumerate(d.contact):
            first = contact.geom1 in s.object_geoms
            if not (first or contact.geom2 in s.object_geoms): continue
            other = contact.geom2 if first else contact.geom1
            normal = np.asarray(contact.frame).reshape(3, 3)[0]*(-1 if first else 1)
            if (int(m.geom_bodyid[other]) != point.body or contact.efc_address < 0 or
                    np.linalg.norm(contact.pos-positions[i]) > .025 or
                    np.linalg.norm(contact.pos-anchors[i]) > .025 or normal@inward[i] < .5):
                continue
            touching[i] = True
            wrench = np.zeros(6); mujoco.mj_contactForce(m, d, ci, wrench)
            measured[i] += (-1 if first else 1)*np.asarray(contact.frame).reshape(3, 3).T@wrench[:3]
    return dict(positions=positions, anchors=anchors, inward=inward, normals=normals,
                jac=np.asarray(jacobians), jac_rot=np.asarray(rotation_jacobians),
                gap=np.linalg.norm(positions-anchors, axis=1), signed_distance=signed_distance,
                closest=closest, touching=touching, measured=measured,
                normal_alignment=np.einsum('ij,ij->i', normals, inward))


class ContactCharts:
    """Cache exposed object and robot collision surfaces; keep patch identities stable."""
    def __init__(self, scene, points):
        self.scene, self.points = scene, points
        self.library = SurfaceLibrary(scene)
        self.cache = {}
        self.audit = []

    def get(self, robot_local, object_local, inward):
        key = np.r_[robot_local.ravel(), object_local.ravel(), inward.ravel()].round(6).tobytes()
        if key in self.cache: return self.cache[key]
        import trimesh
        out = dict(robot_local=[], robot_normal=[], object_local=[], inward=[], robot_geom=[], object_geom=[])
        changes = []
        for i, point in enumerate(self.points):
            op, normal, og, _ = self.library.object_patch(object_local[i], -inward[i], radius=.03)
            candidates = []
            for geom, mesh in self.library.meshes.items():
                if int(self.scene.model.geom_bodyid[geom]) != point.body: continue
                near = trimesh.triangles.closest_point(mesh.triangles,
                            np.tile(robot_local[i], (len(mesh.faces), 1)))
                j = np.argmin(np.linalg.norm(near-robot_local[i], axis=1))
                candidates.append((np.linalg.norm(near[j]-robot_local[i]), geom, near[j], mesh.face_normals[j]))
            distance, rg, rp, rn = min(candidates, key=lambda c: c[0])
            changes.append([float(np.linalg.norm(op-object_local[i])), float(distance)])
            for name, value in zip(out, (rp, rn, op, -normal, rg, og)):
                out[name].append(value)
        out = {k: np.array(v) for k, v in out.items()}
        self.audit.append(dict(chart=len(self.cache), projection_distances_m=changes,
                               valid=bool(max(np.ravel(changes)) <= .015)))
        if not self.audit[-1]['valid']:
            raise ValueError('Saved contact point is more than 15 mm from an exposed actual collision surface')
        self.cache[key] = out
        return out

    def propose_nearby(self, chart, prior, active, *, object_radius=.08, robot_radius=.04):
        """Propose new surface identities, bounded around the fixed human prior.

        A distant contact never counts as load on the old patch. It can instead
        seed an explicitly different patch, which the caller must validate in
        a physical rollout before adopting. Nearest-geometry points also seed
        acquisition while separated; they are never treated as measured load.
        """
        s = self.scene
        observed = observe_patches(s, self.points, chart)
        result = {k: v.copy() for k, v in chart.items()}
        rotation = Rotation.from_quat(s.data.qpos[s.book_q+3:s.book_q+7], scalar_first=True)
        position = s.data.qpos[s.book_q:s.book_q+3]
        changes = []
        for i in np.flatnonzero(active & ~observed['touching']):
            body = self.points[i].body
            seeds = []
            for contact in s.data.contact:
                first = contact.geom1 in s.object_geoms
                if not (first or contact.geom2 in s.object_geoms) or contact.efc_address < 0:
                    continue
                other = contact.geom2 if first else contact.geom1
                if int(s.model.geom_bodyid[other]) != body:
                    continue
                normal = np.asarray(contact.frame).reshape(3, 3)[0]*(-1 if first else 1)
                seeds.append((contact.pos.copy(), normal, 'physical_contact'))
            if np.isfinite(observed['signed_distance'][i]) and observed['signed_distance'][i] < .15:
                seeds.append((observed['closest'][i, 3:].copy(), observed['inward'][i], 'closest_geometry'))
            options = []
            for world, normal, source in seeds:
                anchor = rotation.inv().apply(world-position)
                inward = rotation.inv().apply(normal)
                if (np.linalg.norm(anchor-prior['object_local'][i]) > object_radius or
                        inward@prior['inward'][i] < .5):
                    continue
                try:
                    op, outward, og, _ = self.library.object_patch(anchor, -inward, radius=.006)
                    rg, rp, rn, _ = self.library.robot_patch(body, position+rotation.apply(op),
                                                            rotation.apply(-outward))
                except ValueError:
                    continue
                object_shift = np.linalg.norm(op-prior['object_local'][i])
                robot_shift = np.linalg.norm(rp-prior['robot_local'][i])
                alignment = float(-outward@prior['inward'][i])
                actual_gap = np.linalg.norm(s.data.xpos[body]+s.data.xmat[body].reshape(3,3)@rp-
                                            position-rotation.apply(op))
                if (object_shift > object_radius or robot_shift > robot_radius or alignment < .5 or
                        actual_gap > observed['gap'][i]-.001):
                    continue
                # Prefer closeness, retain the prior and avoid needless switching.
                score = actual_gap+.1*object_shift+.1*robot_shift
                options.append((score, op, outward, og, rg, rp, rn, source,
                                object_shift, robot_shift, actual_gap))
            if not options:
                continue
            _, op, outward, og, rg, rp, rn, source, os, rs, gap = min(options, key=lambda x: x[0])
            for key, value in [('object_local',op), ('inward',-outward), ('object_geom',og),
                               ('robot_geom',rg), ('robot_local',rp), ('robot_normal',rn)]:
                result[key][i] = value
            changes.append(dict(channel=int(i), source=source, object_shift_m=float(os),
                                robot_shift_m=float(rs), proposed_gap_m=float(gap)))
        return result, changes


def allocate_observed(scene, obs, active, desired_p, desired_r, qcmd, vcmd, kp, kd,
                      previous, period, normal_cap=15., friction=.8):
    """Conditional force QP from measured state; separated patches have zero load.

    Object acceleration is a bounded pose-error control target, not a derivative
    of the observed reference. This local allocation is validated in dynamics.
    """
    s = scene; m, d = s.model, s.data; count = len(active)
    support = environment_contacts(s, 8)
    p = np.vstack([obs['positions'], np.array([c['position'] for c in support]).reshape(-1, 3)])
    normals = np.vstack([obs['inward'], np.array([c['normal'] for c in support]).reshape(-1, 3)])
    width = 3*len(p); com = d.xipos[s.book_body]
    r = Rotation.from_quat(d.qpos[s.book_q+3:s.book_q+7], scalar_first=True)
    v = d.qvel[s.book_v:s.book_v+3]; omega = r.apply(d.qvel[s.book_v+3:s.book_v+6])
    error = desired_p-d.qpos[s.book_q:s.book_q+3]
    linear = np.clip(12*error-4*v, -.8, .8)
    angular = np.clip(8*(Rotation.from_quat(desired_r, scalar_first=True)*r.inv()).as_rotvec()-3*omega, -2., 2.)
    inertia_r = Rotation.from_quat(m.body_iquat[s.book_body], scalar_first=True)
    world_inertia = (r*inertia_r).as_matrix()@np.diag(m.body_inertia[s.book_body])@(r*inertia_r).as_matrix().T
    wanted = np.r_[m.body_mass[s.book_body]*(linear-m.opt.gravity),
                   world_inertia@angular+np.cross(omega, world_inertia@omega)]
    operator = np.column_stack([np.vstack((np.eye(3), cross_matrix(x-com))) for x in p])
    scale = np.array([10., 10., 10., 1., 1., 1.]); a = operator/scale[:, None]
    rows, lows, highs = [], [], []
    eligible = active & obs['touching'] & (obs['normal_alignment'] > .3)
    target = np.zeros(width)
    for i, n in enumerate(normals):
        basis = contact_basis(n)
        mu = friction if i < count else support[i-count]['friction']
        cap = normal_cap if i < count and eligible[i] else (0. if i < count else m.body_mass[s.book_body]*20)
        slip = (basis[:, 1:].T@(v+np.cross(omega, p[i]-com)) if i >= count else np.zeros(2))
        sliding = i >= count and np.linalg.norm(slip) > .002
        local = (np.array([[1.,0,0]]) if sliding else
                 np.array([[1, 0, 0]]+[[-.85*mu, x, y] for x, y in [(1,1),(1,-1),(-1,1),(-1,-1)]]))
        block = np.zeros((len(local), width)); block[:, 3*i:3*i+3] = local@basis.T
        rows.extend(block); lows.extend([0.]+[-np.inf]*(len(local)-1)); highs.extend([cap]+[0.]*(len(local)-1))
        if i < count and eligible[i]: target[3*i:3*i+3] = 2*n
        if sliding:
            block = np.zeros((2, width))
            block[:, 3*i:3*i+3] = (np.c_[mu*slip/np.linalg.norm(slip), np.eye(2)])@basis.T
            rows.extend(block); lows.extend([0., 0.]); highs.extend([0., 0.])
    jt = np.zeros((len(s.caps), width)); jt[:, :3*count] = obs['jac'].reshape(3*count, -1).T
    base = (kp*(qcmd-d.qpos[s.active_q])+kd*(vcmd-d.qvel[s.active_v])+
            d.qfrc_bias[s.active_v]-d.qfrc_passive[s.active_v])
    # A hard instantaneous reserve may be infeasible before contact. Return
    # zero with a recorded failure; never invent a feasible force solution.
    rows.extend(jt); lows.extend(-.9*s.caps-base); highs.extend(.9*s.caps-base)
    prior = np.r_[previous.ravel(), np.zeros(3*len(support))]
    hessian = a.T@a+.002*np.eye(width)+.002*np.eye(width)
    problem = osqp.OSQP()
    problem.setup(P=sparse.csc_matrix(2*hessian), q=-2*(a.T@(wanted/scale)+.002*target+.002*prior),
                  A=sparse.csc_matrix(np.array(rows)), l=np.array(lows), u=np.array(highs),
                  verbose=False, eps_abs=2e-5, eps_rel=2e-5, max_iter=3000, polishing=True)
    fit = problem.solve(raise_error=False)
    solved = fit.info.status_val in (1, 2) and fit.x is not None and np.isfinite(fit.x).all()
    forces = fit.x if solved else np.zeros(width)
    # Separated contacts are structurally absent, including solver roundoff.
    forces[:3*count].reshape(count,3)[~eligible]=0.
    value=np.array(rows)@forces
    violation=float(max(0.,np.max(np.array(lows)-value),np.max(value-np.array(highs))))
    if violation>1e-4:
        solved=False;forces[:]=0.
    return forces[:3*count].reshape(count, 3), dict(status=fit.info.status, solved=bool(solved),
        wrench_residual=(operator@forces-wanted).tolist(), eligible=eligible.tolist(),
        environment_contacts=len(support),constraint_violation=violation,
        environment_forces=forces[3*count:].reshape(-1,3).tolist(),physical_success=False)
