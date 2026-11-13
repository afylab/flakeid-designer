#!/usr/bin/env python3
"""Identify exfoliated 2D flakes in stitched optical microscope overviews.

Example:
    python find_flakes.py "example images/g31.jpg" graphite 90 --annotate
    python find_flakes.py "example images/bn24.jpg" hbn 285 --annotate --json

See `python find_flakes.py -h` for all options.
"""
import sys

from flakeid.cli import main

if __name__ == "__main__":
    sys.exit(main())
