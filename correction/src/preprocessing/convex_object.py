"""Cached convex collision pieces for supplied Tier 1 meshes (CPU V-HACD)."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh


def decompose(mesh_path, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest()
    settings = dict(maxConvexHulls=24, resolution=150000, maxNumVerticesPerCH=48)
    manifest = directory/'collision.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info['sha256'] == digest and info['settings'] == settings:
            return [(directory/p['file'], p['mass_fraction']) for p in info['parts']]
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.merge_vertices()
    pieces = trimesh.decomposition.convex_decomposition(mesh, **settings)
    meshes = [trimesh.Trimesh(**part).convex_hull for part in pieces]
    volumes = np.array([abs(m.volume) for m in meshes])
    if not len(meshes) or np.any(volumes <= 0):
        raise ValueError('Convex decomposition produced invalid solid pieces')
    parts = []
    for i, (m, fraction) in enumerate(zip(meshes, volumes/volumes.sum())):
        name = f'part_{i:03d}.obj'
        m.export(directory/name)
        parts.append(dict(file=name, mass_fraction=float(fraction)))
    manifest.write_text(json.dumps(dict(sha256=digest, settings=settings, parts=parts,
        assumption='Uniform density over convex pieces; overlapping volumes are counted separately.'), indent=2))
    return [(directory/p['file'], p['mass_fraction']) for p in parts]
