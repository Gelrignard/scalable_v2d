"""Connect actual robot contact geometry to the actuator-bounded wrench LP."""
import numpy as np
from scipy.spatial.transform import Rotation
from .surface_hypotheses import contact_kinematics
from .wrench_screen import screen_wrench


def static_hold_screen(scene, library, points, floor_z, contract, *, allow_floor=True):
    position, _, jac, _ = contact_kinematics(scene, points)
    r = Rotation.from_quat(scene.data.qpos[scene.book_q+3:scene.book_q+7], scalar_first=True)
    inward = r.apply(-np.array([p.object_outward for p in points]))
    vertices = np.concatenate([library.meshes[g].vertices for g in scene.object_geoms])
    world = scene.data.qpos[scene.book_q:scene.book_q+3]+r.apply(vertices)
    support_gap = float(world[:, 2].min()-floor_z)
    normal_caps = np.full(len(points), contract.normal_limit_n)
    if allow_floor and support_gap <= .003:
        ids = np.flatnonzero(world[:, 2] <= world[:, 2].min()+.002)
        indices = [ids[np.argmin(world[ids, 0])], ids[np.argmax(world[ids, 0])],
                   ids[np.argmin(world[ids, 1])], ids[np.argmax(world[ids, 1])]]
        position = np.vstack((position, world[indices]))
        inward = np.vstack((inward, np.tile([0., 0., 1.], (4, 1))))
        jac = np.concatenate((jac, np.zeros((4, 3, len(scene.movable)))))
        normal_caps = np.r_[normal_caps, np.full(4, scene.model.body_mass[scene.book_body]*20)]
    required = np.r_[-scene.model.body_mass[scene.book_body]*scene.model.opt.gravity, np.zeros(3)]
    result = screen_wrench(position, inward, scene.data.xipos[scene.book_body], required, jac,
        scene.data.qfrc_bias[scene.active_v], scene.caps,
        friction=contract.friction, normal_caps=normal_caps)
    result['floor_contacts_included'] = len(position) > len(points)
    return result, support_gap
