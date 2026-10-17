"""Compatibility entry point; implementation now lives in retargeting/."""

from pathlib import Path
import argparse
import os
import runpy
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retargeting/src"))
from v2d_retarget.paths import (
    UPSTREAM as RG,
    DEMO_PARQUET as MOTION,
    DEMO,
    enable_upstream,
)
from v2d_retarget.dataprocessing import load_demo

enable_upstream()
# Retain read-only imports used by the previous audit artifact.
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.whole_body_kinematics import WholeBodyKinematics
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "ROOT",
    "RG",
    "MOTION",
    "DEMO",
    "read_demo",
    "load_robot_config",
    "WholeBodyKinematics",
    "SOMA_JOINTS_ORDER",
]


def read_demo():
    motion = load_demo()
    world = np.asarray(load_robot_config("g1").r_world)
    q = motion.rotations_wxyz
    rotations = (
        Rotation.from_matrix(
            world.T
            @ Rotation.from_quat(q.reshape(-1, 4), scalar_first=True).as_matrix()
        )
        .as_quat(scalar_first=True)
        .reshape(q.shape)
    )
    return motion.upstream_motion, motion.positions @ world, rotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["retarget", "view"], default="view", nargs="?")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Compatibility option; viewer defaults to contact-aware motion",
    )
    args = parser.parse_args()
    script = (
        ROOT
        / "retargeting/scripts"
        / ("run.py" if args.mode == "retarget" else "visualize_viser.py")
    )
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)]
    if args.mode == "retarget":
        sys.argv += ["--max-iterations", str(args.max_iter)]
        if args.max_frames:
            sys.argv += [
                "--stop",
                str(args.max_frames),
                "--output",
                str(ROOT / f"retargeting/outputs/preview_{args.max_frames}"),
            ]
    else:
        sys.argv += ["--port", str(args.port)]
    runpy.run_path(str(script), run_name="__main__")
