"""Measured solver contact wrenches, expressed at the actual object COM."""
import mujoco
import numpy as np


def object_contact_wrench(model, data, index, object_geoms, object_body):
    """Return robot/environment-on-object [force, moment] in world coordinates.

    MuJoCo's contact wrench acts on geom2, in the contact frame. The moment
    below includes the contact couple and its actual moment arm; it does not
    use a planned anchor or an inferred observed acceleration.
    """
    contact = data.contact[index]
    first = contact.geom1 in object_geoms
    second = contact.geom2 in object_geoms
    if first == second:
        raise ValueError('Exactly one contact geom must belong to the object')
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, wrench)
    transform = (-1. if first else 1.) * np.asarray(contact.frame).reshape(3, 3).T
    force = transform @ wrench[:3]
    moment = transform @ wrench[3:] + np.cross(contact.pos-data.xipos[object_body], force)
    return np.r_[force, moment]


def object_wrench_partition(model, data, object_geoms, object_body, environment, *,
                           detail=False, patches=None):
    """Solver wrenches and unsigned robot-body force for interval diagnostics."""
    robot, support = np.zeros(6), np.zeros(6)
    body_force = np.zeros(model.nbody)
    body_wrench = np.zeros((model.nbody, 6)) if detail else None
    selected = np.zeros(6)
    active_selected = np.zeros(6)
    for index, contact in enumerate(data.contact):
        if not object_geoms.intersection((contact.geom1, contact.geom2)):
            continue
        other = contact.geom2 if contact.geom1 in object_geoms else contact.geom1
        wrench = object_contact_wrench(model, data, index, object_geoms, object_body)
        if other in environment:
            support += wrench
        else:
            robot += wrench
            body = int(model.geom_bodyid[other])
            body_force[body] += np.linalg.norm(wrench[:3])
            if detail:
                body_wrench[body] += wrench
                patch = None if patches is None else patches.get(body)
                if patch is not None:
                    point, normal, active = patch
                    sign = -1 if contact.geom1 in object_geoms else 1
                    if (sign*np.asarray(contact.frame).reshape(3,3)[0]@normal >= .5
                            and np.linalg.norm(contact.pos-point) <= .03):
                        selected += wrench
                        if active:
                            active_selected += wrench
    if detail:
        return robot, support, body_force, dict(body_wrench=body_wrench,
            selected_patch_wrench=selected, active_selected_patch_wrench=active_selected)
    return robot, support, body_force
