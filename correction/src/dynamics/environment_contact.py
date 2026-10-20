"""Environment support from the same collision manifold used by execution.

Candidates are geometrically present simulator contacts, not a height-threshold
guess. They are possible supports, not promises of nonzero measured force.
"""
import numpy as np


def environment_contacts(scene, max_contacts=8):
    rows = []
    for contact in scene.data.contact:
        first = int(contact.geom1) in scene.object_geoms
        second = int(contact.geom2) in scene.object_geoms
        if first == second:
            continue
        other = int(contact.geom2 if first else contact.geom1)
        if other not in scene.environment or contact.efc_address < 0:
            continue
        normal = np.asarray(contact.frame).reshape(3, 3)[0]*(-1 if first else 1)
        point = np.asarray(contact.pos).copy()
        if any(np.linalg.norm(point-r['position']) < 1e-5 and normal@r['normal'] > .999
               for r in rows):
            continue
        rows.append(dict(position=point, normal=normal.copy(),
                         friction=float(contact.friction[0]), distance=float(contact.dist),
                         environment_geom=other))
    # Retain spatial coverage if the convex decomposition produces many contacts.
    if len(rows) > max_contacts:
        positions = np.array([r['position'] for r in rows])
        chosen = [int(np.argmin(positions[:, 0]))]
        while len(chosen) < max_contacts:
            distance = np.min(np.linalg.norm(positions[:, None]-positions[chosen], axis=2), axis=1)
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        rows = [rows[i] for i in chosen]
    return rows
