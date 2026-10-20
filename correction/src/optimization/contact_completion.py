"""Propose a missing robot contact from executed geometry and human regions.

This does not infer measured force from a reset scene. Saved physical impulses
identify which body to inspect; geometry supplies a candidate for a new plan.
"""
import numpy as np
from .surface_hypotheses import SurfacePoint


def extract_contact_patch(scene, library, body_name, evidence_anchors, *,
                          source_channels, region_radius=.1, max_gap=.002):
    body = scene.model.body(body_name).id
    candidates = []
    for contact in scene.data.contact:
        first = contact.geom1 in scene.object_geoms
        second = contact.geom2 in scene.object_geoms
        if first == second:
            continue
        other = int(contact.geom2 if first else contact.geom1)
        if int(scene.model.geom_bodyid[other]) != body or contact.dist > max_gap:
            continue
        inward = (-1. if first else 1.)*np.asarray(contact.frame)[:3]
        candidates.append((float(contact.dist), contact.pos.copy(), inward))
    if not candidates:
        raise ValueError('No collision-surface proximity on the requested body')
    distance, world, inward = min(candidates, key=lambda x:x[0])
    rotation = scene.data.xmat[scene.book_body].reshape(3,3)
    origin = scene.data.xpos[scene.book_body]
    observed_local = rotation.T@(world-origin)
    anchors = np.asarray(evidence_anchors, float)
    nearest = int(np.argmin(np.linalg.norm(anchors-observed_local, axis=1)))
    evidence_distance = float(np.linalg.norm(anchors[nearest]-observed_local))
    if evidence_distance > region_radius:
        raise ValueError(f'Executed contact outside human hand region: {evidence_distance:.6f}m')
    # Collision points can sit midway across a positive contact margin.
    # Snap independently to the two actual convex surface meshes.
    op, outward, og, ot = library.object_patch(observed_local, -rotation.T@inward,
                                              radius=.005, normal_weight=.08)
    geom, local, normal, rt = library.robot_patch(body, origin+rotation@op, -rotation@outward)
    actual_robot = scene.data.xpos[body]+scene.data.xmat[body].reshape(3,3)@local
    point = SurfacePoint(body,geom,local,normal,op,outward,og,rt,ot,
                         int(source_channels[nearest]),anchors[nearest].copy())
    return point, dict(body=body_name, collision_distance_m=distance,
        collision_point_world_m=world.tolist(), collision_inward_world=inward.tolist(),
        nearest_evidence_channel=int(source_channels[nearest]), evidence_distance_m=evidence_distance,
        patch_evidence_distance_m=float(np.linalg.norm(op-anchors[nearest])),
        snapped_surface_gap_m=float(np.linalg.norm(actual_robot-origin-rotation@op)),
        collision_to_object_patch_m=float(np.linalg.norm(observed_local-op)),
        scope='Geometric candidate only; no force measurement inferred from this reconstructed scene')
