"""Inspect contact phases in the canonical retargeting input."""
import argparse
from pathlib import Path
from correction.src.preprocessing.retarget_bridge import inspect_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    inspect_motion(parser.parse_args().reference)


if __name__ == "__main__":
    main()
