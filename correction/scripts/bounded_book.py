"""Run bounded CPU contact correction and optional physical diagnostics."""
import argparse
from pathlib import Path
from correction.src.optimization.bounded_book import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('correction/outputs/tier1_book'))
    parser.add_argument('--output', type=Path, default=Path('correction/outputs/bounded_book'))
    parser.add_argument('--start', type=int, default=400)
    parser.add_argument('--stop', type=int, default=601, help='exclusive')
    parser.add_argument('--padding', type=int, default=20)
    parser.add_argument('--budget', type=float, default=120., help='CPU correction wall seconds; audit/rollout timed separately')
    parser.add_argument('--window', type=int, default=31)
    parser.add_argument('--overlap', type=int, default=8)
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--robot-candidates', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=3, help='SQP steps per alternating round')
    parser.add_argument('--trust', type=float, default=.9, help='joint correction radius in radians')
    parser.add_argument('--mass', type=float, default=1., help='provisional book mass in kg')
    parser.add_argument('--friction', type=float, default=.6)
    parser.add_argument('--rollout', action='store_true', help='physical replay and bounded preload search')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
