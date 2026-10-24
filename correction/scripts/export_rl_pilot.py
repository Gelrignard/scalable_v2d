"""Package the completed paired pilot with the versioned RL artifact contract."""
import argparse
from pathlib import Path
from correction.src.pass_traj.rl_pilot import export_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_pilot(args.root, args.output)


if __name__ == '__main__':
    main()
