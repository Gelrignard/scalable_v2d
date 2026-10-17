"""Audit wrench feasibility at the saved robot contacts."""
import argparse
from pathlib import Path
from correction.src.paths import BOOK_OUTPUT
from correction.src.dynamics.contact_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=BOOK_OUTPUT)
    audit(parser.parse_args().directory)


if __name__ == "__main__":
    main()
