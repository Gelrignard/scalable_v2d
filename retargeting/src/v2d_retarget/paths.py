from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[3]
MODULE = WORKSPACE / "retargeting"
UPSTREAM = WORKSPACE / "upstream/robotic_grounding"
UPSTREAM_COMMIT = "193382a184308f6800331f3db829f76dc4294cfe"
DEMO = "2026-03-06_10-24-18_snack_box_pick_and_place_01"
DEMO_PARQUET = (
    UPSTREAM
    / "source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma"
    / ("sequence_id=" + DEMO)
    / "robot_name=g1/data.parquet"
)


def enable_upstream():
    """Use the pinned source checkout without importing Isaac Lab or Torch."""
    source = UPSTREAM / "source/robotic_grounding"
    if not source.is_dir():
        raise FileNotFoundError(f"NVIDIA source checkout required: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
