"""Refine a saved robot reference and write the downstream NPZ/JSON artifacts."""
import argparse
from pathlib import Path
import json
from correction.src.paths import BOOK_OUTPUT
from correction.src.optimization.robot_trajectory import repair as refine_trajectory
from correction.src.pass_traj.robot_reference import save_robot_reference

def repair(directory):
    result = refine_trajectory(directory)
    path = save_robot_reference(directory, result)
    metrics = result.metrics
    print(json.dumps({k:v for k,v in metrics.items() if k.startswith('phase_') or k in ('elapsed_seconds','max_joint_change_rad','speed_violating_transitions')}, indent=2), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=BOOK_OUTPUT)
    repair(parser.parse_args().directory)


if __name__ == "__main__":
    main()
