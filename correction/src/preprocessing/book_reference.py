"""Load the canonical motion and saved robot reference from retargeting."""
import json

from .retarget_bridge import WORKSPACE
from v2d_retarget.dataprocessing import load_npz
from v2d_retarget.core.contacts import build_contacts
from v2d_retarget.core.surface import ObjectSurface
from v2d_retarget.output import load_result


def load_book_reference(directory, preserve_saved_object_pose=False):
    motion = load_npz(directory / "human_motion.npz")
    robot, metadata = load_result(directory / "retargeted/contact_aware.npz")
    if preserve_saved_object_pose:
        # Retain even sub-ulp differences in the saved quaternion coefficients.
        motion.object_positions = robot["object_positions"].copy()
        motion.object_wxyz = robot["object_wxyz"].copy()
    config = json.loads((WORKSPACE / "retargeting/configs/contact_aware.json").read_text())
    surface = ObjectSurface(motion.object_mesh)
    contacts = build_contacts(motion, surface, config)
    return motion, robot, metadata, config, surface, contacts
