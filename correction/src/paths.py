"""Checkout-relative paths shared by the correction stages."""
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
CORRECTION = WORKSPACE / "correction"
BOOK_OUTPUT = CORRECTION / "outputs/tier1_book"
