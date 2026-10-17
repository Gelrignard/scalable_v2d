from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree


class ObjectSurface:
    """Exact unsigned mesh proximity; conservative convex-hull collision proxy.

    Reconstructed meshes may be open. We deliberately do not pretend that an
    unsigned nearest-surface distance is a signed mesh penetration distance.
    """

    def __init__(self, path):
        self.mesh = trimesh.load(str(Path(path)), force="mesh", process=True)
        if not len(self.mesh.faces):
            raise ValueError("Object needs a triangle mesh")
        self.hull = ConvexHull(self.mesh.vertices)
        self.planes = self.hull.equations.copy()
        self.center = self.mesh.vertices[self.hull.vertices].mean(axis=0)
        self.vertex_tree = cKDTree(self.mesh.vertices)
        self.max_edge = float(self.mesh.edges_unique_length.max())

    def contact_candidates(self, points, cutoff):
        """Exact triangle distances for candidates; safe lower bounds far away.

        d_vertex - longest_edge <= true surface distance. Points whose lower
        bound exceeds the release threshold cannot be contacts. This avoids
        huge trimesh query boxes for hands metres away from a dense mesh.
        """
        shape = np.asarray(points).shape
        pts = np.asarray(points).reshape(-1, 3)
        vd, vi = self.vertex_tree.query(pts)
        distance = np.maximum(vd - self.max_edge, 0.0)
        closest = self.mesh.vertices[vi].copy()
        normals = np.zeros_like(pts)
        candidate = distance <= cutoff
        if candidate.any():
            closest[candidate], distance[candidate], normals[candidate] = self.nearest(
                pts[candidate]
            )
        return (
            closest.reshape(shape),
            distance.reshape(shape[:-1]),
            normals.reshape(shape),
        )

    def nearest(self, points):
        shape = np.asarray(points).shape
        pts = np.asarray(points).reshape(-1, 3)
        if not len(pts):
            return np.empty(shape), np.empty(shape[:-1]), np.empty(shape)
        result, distance, normals = [], [], []
        for offset in range(0, len(pts), 128):
            p, d, tri = trimesh.proximity.closest_point(
                self.mesh, pts[offset : offset + 128]
            )
            normal = self.mesh.face_normals[tri].copy()
            # Orient the local surface normal outward for the object's shell.
            flip = np.einsum("ij,ij->i", normal, p - self.center) < 0
            normal[flip] *= -1
            result.append(p)
            distance.append(d)
            normals.append(normal)
        return (
            np.concatenate(result).reshape(shape),
            np.concatenate(distance).reshape(shape[:-1]),
            np.concatenate(normals).reshape(shape),
        )

    def hull_distance_normal(self, points):
        points = np.asarray(points)
        distances = points @ self.planes[:, :3].T + self.planes[:, 3]
        face = np.argmax(distances, axis=-1)
        return np.take_along_axis(distances, face[..., None], axis=-1)[
            ..., 0
        ], self.planes[face, :3]
