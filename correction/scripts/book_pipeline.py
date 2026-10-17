"""Run book wrench planning, robot refinement/export, and the final force audit."""
import argparse
from pathlib import Path
from correction.src.paths import BOOK_OUTPUT
from .book_trial import plan
from .repair_robot import repair
from correction.src.dynamics.contact_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=BOOK_OUTPUT)
    parser.add_argument("--start", type=int, default=135)
    parser.add_argument("--stop", type=int, default=176)
    args = parser.parse_args()
    plan(args.directory, args.start, args.stop)
    repair(args.directory)
    audit(args.directory)


if __name__ == "__main__":
    main()
