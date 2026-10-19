"""Independent attained-contact and temporal checks, including between samples."""
from itertools import product
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation

from .sparse_forces import allocate_forces


def box_contacts(local, bounds):
    """Nearest box surface point, inward normal, and exact signed distance."""
    center, half = bounds.mean(axis=0), (bounds[1]-bounds[0])/2
    offset = local-center
    face_distance = np.abs(offset)-half
    axes = np.argmax(face_distance, axis=-1)
    point = np.clip(local, bounds[0], bounds[1])
    normal = np.zeros_like(local)
    for idx in np.ndindex(local.shape[:-1]):
        axis = axes[idx]
        sign = 1 if offset[idx][axis] >= 0 else -1
        point[idx][axis] = center[axis]+sign*half[axis]
        normal[idx][axis] = -sign
    signed = np.linalg.norm(np.maximum(face_distance, 0), axis=-1)+np.minimum(np.max(face_distance, axis=-1), 0)
    return point, normal, signed


def support_contacts(scene, positions, quaternions):
    """Possible upward support at box vertices close to table top or floor.

    No arbitrary environment wrench: forces are unilateral and restricted to
    geometric support locations. Cylinder side/edge contact is not modeled.
    """
    corners = np.array(list(product(*zip(scene.bounds[0], scene.bounds[1]))))
    rotations = Rotation.from_quat(quaternions, scalar_first=True).as_matrix()
    world = np.einsum('tij,kj->tki', rotations, corners)+positions[:, None]
    near_table = (np.linalg.norm(world[:, :, :2]-scene.table_center[:2], axis=-1) <= scene.table_radius)
    near_table &= np.abs(world[:, :, 2]-scene.table_top) <= .003
    near_floor = np.abs(world[:, :, 2]) <= .003
    inward = np.einsum('tji,j->ti', rotations, np.array([0., 0., 1.]))
    return (np.broadcast_to(corners, world.shape).copy(),
            np.repeat(inward[:, None], len(corners), axis=1), near_table | near_floor)


def forces_with_support(scene, points, normals, active, required, positions, quaternions, times,
                        *, time_limit=2.):
    ep, en, ea = support_contacts(scene, positions, quaternions)
    count = points.shape[1]
    return allocate_forces(np.concatenate((points, ep), axis=1), np.concatenate((normals, en), axis=1),
                           np.concatenate((active, ea), axis=1), required, times, scene.bounds.mean(axis=0),
                           cap=np.r_[np.full(count, 15.), np.full(ep.shape[1], 100.)],
                           mu=scene.friction, time_limit=time_limit)


def attained(kin, scene, q, positions, quaternions, times, required, *, collision=True,
             midpoints=False, time_limit=2.):
    from v2d_retarget.core.contacts import TIP_NAMES
    started = perf_counter()
    rotations = Rotation.from_quat(quaternions, scalar_first=True)
    tips = np.array([[p.translation for p in kin.poses(qt, TIP_NAMES)] for qt in q])
    local = np.einsum('tji,tkj->tki', rotations.as_matrix(), tips-positions[:, None])
    points, normals, distance = box_contacts(local, scene.bounds)
    gap = distance-.002
    active = np.abs(gap) <= .003
    result = forces_with_support(scene, points, normals, active, required, positions, quaternions,
                                 times, time_limit=time_limit)
    maximum, violations, pair_worst, manipulation = [], [], {}, []
    samples = 0
    if collision:
        for i in range(len(q)):
            checks = [(q[i], positions[i], quaternions[i])]
            if midpoints and i+1 < len(q):
                middle = (q[i]+q[i+1])/2
                root = Rotation.from_quat(q[i, 3:7])
                middle[3:7] = (root*Rotation.from_rotvec(.5*(root.inv()*Rotation.from_quat(q[i+1, 3:7])).as_rotvec())).as_quat()
                rot = rotations[i]*Rotation.from_rotvec(.5*(rotations[i].inv()*rotations[i+1]).as_rotvec())
                checks.append((middle, (positions[i]+positions[i+1])/2, rot.as_quat(scalar_first=True)))
            for qt, pos, rot in checks:
                scene.set_state(qt, pos, rot)
                contacts = scene.contacts()
                worst = max([max(0., -c['distance']) for c in contacts], default=0.)
                maximum.append(worst)
                manipulation.append(max([max(0., -c['distance']) for c in contacts
                    if any(part in c['pair'] for part in ('hand', 'wrist', 'elbow', 'shoulder'))], default=0.))
                violations.append(any(c['distance'] < -.002 for c in contacts))
                samples += 1
                for c in contacts:
                    pair_worst[c['pair']] = max(pair_worst.get(c['pair'], 0.), max(0., -c['distance']))
    velocity = np.diff(q[:, 7:], axis=0)/np.diff(times)[:, None]
    accel = np.diff(velocity, axis=0)/((np.diff(times)[:-1]+np.diff(times)[1:])/2)[:, None]
    slip = np.linalg.norm(np.diff(local, axis=0), axis=-1)/np.diff(times)[:, None]
    stable = active[1:] & active[:-1] & (np.sum(normals[1:]*normals[:-1], axis=-1) > .99)
    report = dict(seconds=perf_counter()-started, frames=len(q),
                  contact_gap_mean_mm=float(np.abs(gap).mean()*1000),
                  wrench_feasible_frames=int(result.feasible.sum()),
                  max_force_residual_n=float(np.linalg.norm(result.residual[:, :3], axis=1).max()),
                  max_torque_residual_nm=float(np.linalg.norm(result.residual[:, 3:], axis=1).max()),
                  max_penetration_mm=max(maximum, default=0.)*1000 if collision else None,
                  max_manipulation_penetration_mm=max(manipulation, default=0.)*1000 if collision else None,
                  manipulation_penetration_squared_sum_m2=float(np.sum(np.square(manipulation))),
                  collision_samples=samples, collision_samples_over_2mm=sum(violations),
                  worst_collision_pairs_mm={k: v*1000 for k, v in sorted(pair_worst.items(), key=lambda kv: -kv[1])[:12]},
                  max_joint_speed_ratio=float((np.abs(velocity)/kin.robot.model.velocityLimit[6:]).max()),
                  max_joint_acceleration_rad_s2=float(np.abs(accel).max()) if len(accel) else 0.,
                  mean_contact_relative_speed_mm_s=float(slip[stable].mean()*1000) if stable.any() else None,
                  max_contact_relative_speed_mm_s=float(slip[stable].max()*1000) if stable.any() else None,
                  force_solver_status=result.status,
                  collision_model='URDF collision shapes (convex mesh contacts), rigid box book, table and floor; 2 mm reporting tolerance',
                  between_sample_check='midpoints only; not a continuous swept-volume guarantee' if midpoints else 'saved frames only')
    return report, dict(tips=tips, points=points, normals=normals, active=active, gap=gap,
                        forces=result.forces[:, :6], residual=result.residual, feasible=result.feasible)
