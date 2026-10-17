"""Independently re-evaluate final FK and contact metrics from an output artifact."""

import _bootstrap  # noqa: F401 -- initialize local source/runtime paths
import argparse
import json
from pathlib import Path
import numpy as np
from v2d_retarget.dataprocessing import load_npz
from v2d_retarget.output import load_result
from v2d_retarget.core.robot import G1Kinematics
from v2d_retarget.core.surface import ObjectSurface
from v2d_retarget.core.contacts import ContactTargets
from v2d_retarget.core.targets import BodyTargets
from v2d_retarget.core.diagnostics import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    a = p.parse_args()
    result, meta = load_result(a.result)
    motion = load_npz(a.result.parent / "human_motion.npz")
    kin = G1Kinematics()
    if meta["joint_names"] != kin.joint_names or not np.array_equal(
        result["timestamps"], motion.timestamps
    ):
        raise ValueError("Robot joint order or source timing mismatch")
    entries = [kin.config.ik_map[n] for n in meta["body_frame_names"]]
    source = motion.positions[:, [motion.index(e.soma_joint) for e in entries]]
    body = BodyTargets(
        tuple(meta["body_frame_names"]),
        result["body_targets"],
        result["body_rotations"],
        source,
        np.array([e.position_cost for e in entries]),
        np.array([e.orientation_cost for e in entries]),
    )
    contacts = ContactTargets(
        result["source_tips"],
        result["tip_targets"],
        result["contact_active"],
        result["contact_blend"],
        result["contact_anchors_local"],
        result["contact_normals_local"],
        np.empty((0, 6)),
        np.empty((0, 6)),
        meta["config"]["tip_clearance_m"],
    )
    report, _ = evaluate(
        kin, motion, result["q"], body, contacts, ObjectSurface(motion.object_mesh)
    )
    path = a.result.with_name(a.result.stem + "_reevaluated.json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(path)
