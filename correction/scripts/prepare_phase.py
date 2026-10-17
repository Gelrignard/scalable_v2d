"""Prepare one correction phase from a retargeted geometric reference."""
import argparse
from pathlib import Path
import json
from correction.src.preprocessing.reference import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--stop-frame", type=int, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--window", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference, args.config, args.output,
                             args.start_frame, args.stop_frame, args.stride, args.window), indent=2))


if __name__ == "__main__":
    main()
