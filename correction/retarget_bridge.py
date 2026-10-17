"""CPU-only bridge to the separate, unchanged retargeting package."""
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "retargeting/src"))
from v2d_retarget.paths import enable_upstream
enable_upstream()


def inspect_motion(path):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation
    from v2d_retarget.dataprocessing import load_npz
    from v2d_retarget.core.contacts import build_contacts, TIP_NAMES
    from v2d_retarget.core.surface import ObjectSurface
    from .dynamics import contiguous_runs
    motion = load_npz(path)
    cfg = json.loads((WORKSPACE / 'retargeting/configs/contact_aware.json').read_text())
    surface = ObjectSurface(motion.object_mesh)
    contacts = build_contacts(motion, surface, cfg)
    rotation = Rotation.from_quat(motion.object_wxyz, scalar_first=True).as_matrix()
    vertices = surface.mesh.vertices
    floor_height = np.array([(vertices @ r.T)[:, 2].min() + p[2] for r, p in zip(rotation, motion.object_positions)])
    report = dict(frames=len(motion.timestamps), object_floor_height_range_m=[float(floor_height.min()),float(floor_height.max())], contacts={})
    for j,n in enumerate(TIP_NAMES):
        report['contacts'][n] = [{'indices':[int(run[0]),int(run[-1])], 'time_s':[float(motion.timestamps[run[0]]),float(motion.timestamps[run[-1]])], 'anchor':contacts.anchors_local[run[0],j].tolist(), 'normal':contacts.normals_local[run[0],j].tolist()} for run in contiguous_runs(contacts.active[:,j])]
    report['min_source_tip_distance_m'] = contacts.surface_distance.min(axis=0).tolist()
    target = Path(path).with_name('contact_inspection.json')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    inspect_motion(sys.argv[1])
