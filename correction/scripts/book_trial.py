"""Plan and save contact wrenches for the explicit Tier 1 book phase."""
import argparse
from pathlib import Path
import json
from correction.src.paths import BOOK_OUTPUT
from correction.src.optimization.book_wrench import plan as solve_plan
from correction.src.pass_traj.book_plan import save_wrench_plan

def plan(directory, start, stop):
    arrays, report = solve_plan(directory, start, stop)
    save_wrench_plan(directory, arrays, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=BOOK_OUTPUT)
    parser.add_argument('--start', type=int, default=135)
    parser.add_argument('--stop', type=int, default=176)
    args = parser.parse_args()
    plan(args.directory, args.start, args.stop)


if __name__ == "__main__":
    main()
