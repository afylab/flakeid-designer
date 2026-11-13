#!/usr/bin/env python3
"""Moved to flakeid_designer/marks_to_klayout.py (the designer's "Export to KLayout" uses it); this runs that file."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, os.pardir, "flakeid_designer", "marks_to_klayout.py")
sys.argv[0] = TARGET
runpy.run_path(TARGET, run_name="__main__")
