"""Build an explicit first-pickup hypothesis from the saved book retargeting.

This adapter currently supports the approximately box-shaped book and a planar
portion of its table. It does not pretend to infer arbitrary contact schedules.
The first pickup (frames 0..120) and contact times are declared before replay.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from .horizon_reference import HorizonReference, ContactPatch
from correction.src.dynamics.symbolic_mechanics import BodyPoint, SymbolicRobot


def surface_point(scene, body, target_world, inward_world):
    """Select the collision surface facing the object, even from a bad pose.

    A penetrated retargeted finger's nearest point can be its *back* surface.
    The robot surface must face the direction of compression into the object.
    MuJoCo collides the convex hull of a mesh, so use that same hull here.
    """
    model, data = scene.model, scene.data
    candidates = []
    for geom in np.flatnonzero(model.geom_bodyid == body):
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fstart, fcount = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        vertices = np.asarray(model.mesh_vert[start:start+count])
        vertices = vertices@data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]
        faces = np.asarray(model.mesh_face[fstart:fstart+fcount])
        hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
        alignment = hull.face_normals@np.asarray(inward_world)
        keep = alignment >= max(.65, float(alignment.max())-.10)
        if not keep.any(): keep[np.argmax(alignment)] = True
        surface = trimesh.Trimesh(vertices=hull.vertices, faces=hull.faces[keep], process=False)
        point, distance, triangle = trimesh.proximity.closest_point_naive(surface, np.asarray(target_world)[None])
        candidates.append((float(distance[0]), point[0], int(geom), surface.face_normals[int(triangle[0])]))
    if not candidates:
        raise ValueError(f'No collision mesh for contact body {model.body(body).name}')
    distance, world, geom, normal = min(candidates, key=lambda item: item[0])
    local = data.xmat[body].reshape(3, 3).T@(world-data.xpos[body])
    return BodyPoint(body, local), dict(body=model.body(body).name, geom=model.geom(geom).name,
                                       local=local.tolist(), reference_distance_m=distance,
                                       selected_surface_normal_world=normal.tolist(),
                                       inward_alignment=float(normal@inward_world))


def load_book_pickup(path, *, stride=6, mass=1., thumb_faces=(1, 1), hypothesis='pinch_thickness'):
    from correction.src.preprocessing import retarget_bridge
    from v2d_retarget.core.robot import G1Kinematics
    from correction.src.dynamics.supported_robot import SupportedRobotScene
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: archive[key].copy() for key in archive.files}
    if len(raw['q']) < 121:
        raise ValueError('The declared book pickup requires source frames 0..120')
    mesh = trimesh.load(str(raw['object_mesh']), force='mesh')
    meta = json.loads(str(raw['metadata_json']))
    ids = np.unique(np.r_[np.arange(0, 121, stride), 63, 81, 120])
    times = raw['timestamps'][ids]-raw['timestamps'][0]
    observed_times = raw['timestamps'][:121]-raw['timestamps'][0]
    kin = G1Kinematics()
    scene = SupportedRobotScene(kin, mesh.bounds, meta['source']['support_surface'], raw['q'][0],
                               mass=mass, friction=.6, move_waist=True, moving_fixture=True)
    scene.model.opt.noslip_iterations = 10
    com = scene.model.body_ipos[scene.book_body].copy()
    original_rotations = Rotation.from_quat(raw['object_wxyz'][:121], scalar_first=True)
    observed = raw['object_positions'][:121]+original_rotations.apply(np.tile(com, (121, 1)))
    half = np.ptp(mesh.bounds, axis=0)/2
    # A supported initial rest pose is estimated once from the table and first
    # observed orientation. The raw observations stay unchanged. Both replays
    # use this same state; no changes are made during execution.
    yaw = original_rotations[0].as_euler('xyz')[2]
    initial_rotation = Rotation.from_euler('z', yaw)*Rotation.from_euler('x', np.pi/2)
    initial_position = observed[0].copy()
    initial_position[2] = scene.table_top+half[1]
    position_shift = initial_position-observed[0]
    rotation_shift = (initial_rotation.inv()*original_rotations[0]).magnitude()
    if np.linalg.norm(position_shift) > .015 or rotation_shift > np.deg2rad(3):
        raise ValueError('Initial resting-pose registration exceeds the declared perception tolerance')
    qref = raw['q'][ids][:, 7+scene.movable]
    base = np.zeros((len(ids), 25))
    root_position = CubicSpline(times, raw['q'][ids, :3])
    root_rotation = RotationSpline(times, Rotation.from_quat(raw['q'][ids, 3:7]))
    rotations = root_rotation(times)
    base[:, :3], base[:, 3:7] = root_position(times), rotations.as_quat(scalar_first=True)
    base[:, 7:10], base[:, 13:16] = root_position(times, 1), root_position(times, 2)
    base[:, 10:13], base[:, 16:19] = rotations.apply(root_rotation(times, 1)), rotations.apply(root_rotation(times, 2))
    local_human = np.einsum('tji,tkj->tki', original_rotations[90:121].as_matrix(),
                           raw['source_tips'][90:121]-observed[90:121, None])
    median = np.median(local_human, axis=0)
    patches, points, point_info = [], [], []
    scene.set_state(raw['q'][90], raw['object_positions'][90], raw['object_wxyz'][90])
    for i in range(6):
        hand, finger = divmod(i, 3)
        axis = 1
        if hypothesis == 'pinch_thickness':
            sign = (1 if thumb_faces[hand] else -1)*(1 if finger == 0 else -1)
        elif hypothesis in ('two_hand_edges', 'opposed_faces'):
            axis = 0 if hypothesis == 'two_hand_edges' else 1
            sign = -1 if hand == 0 else 1
        else:
            raise ValueError('Unknown reviewed contact hypothesis')
        center = np.clip(median[i], -half+.012, half-.012)
        center[axis] = sign*half[axis]
        inward = np.zeros(3); inward[axis] = -sign
        tangents = np.eye(3)[:, [j for j in range(3) if j != axis]]
        patch = ContactPatch(scene.tip_names[i], center, inward,
                             tangents, ids >= 63,
                             robot_index=i, radius=.012, sigma=.02, force_limit=15., friction=.45,
                             observed_center=median[i])
        patches.append(patch)
        target = observed[90]+original_rotations[90].apply(center)
        point, info = surface_point(scene, int(scene.model.site_bodyid[scene.tip_ids[i]]), target,
                                    original_rotations[90].apply(inward))
        points.append(point); point_info.append(info)
    # Local -Y face rests on the horizontal table before lift-off. These are
    # sticking contacts; no fictitious sliding friction is available.
    for sx in (-1, 1):
        for sz in (-1, 1):
            patches.append(ContactPatch(f'table_{sx}_{sz}', np.array([sx*half[0], -half[1], sz*half[2]]),
                np.array([0., 1., 0.]), np.array([[1., 0.], [0., 0.], [0., 1.]]), ids <= 81,
                radius=0., force_limit=30., friction=.45, support_z=scene.table_top))
    robot = SymbolicRobot(scene, points)
    principal = Rotation.from_quat(scene.model.body_iquat[scene.book_body], scalar_first=True).as_matrix()
    inertia = principal@np.diag(scene.model.body_inertia[scene.book_body])@principal.T
    ref = HorizonReference(times, observed_times, observed, raw['object_wxyz'][:121], qref, base,
        qref[0], np.zeros(robot.n), initial_position, initial_rotation.as_quat(scalar_first=True),
        np.zeros(3), np.zeros(3), observed[-1], raw['object_wxyz'][120], patches, mass, inertia, half,
        position_sigma=.015, rotation_sigma=np.deg2rad(5), goal_position_tolerance=.02,
        goal_rotation_tolerance=np.deg2rad(6), metadata=dict(source=str(Path(path).resolve()),
            episode_index=11, source_frame_range_inclusive=[0, 120], time_scale=1.,
            contact_onset_source_frame=63, support_release_source_frame=81,
            hypothesis=hypothesis, thumb_faces=list(thumb_faces),
            assumption='1 kg homogeneous box; reviewed fixed contact schedule; table plane during pickup',
            initial_position_correction_m=position_shift.tolist(), initial_rotation_correction_deg=float(np.rad2deg(rotation_shift)),
            contact_surface_points=point_info, object_com_local=com.tolist(),
            automatically_inferred_contact_schedule=False, noslip_iterations=10))
    return ref, robot, scene, raw, ids
