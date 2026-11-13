"""Where the network-test data lives: FLAKEID_ML_DATA, else flakeid_ml_data next to the repository folder.
sets/ hold paths relative to that root (see README.md); results/ takes every output."""
import os
from pathlib import Path

DATA = Path(os.environ.get("FLAKEID_ML_DATA", Path(__file__).resolve().parents[2] / "flakeid_ml_data"))
SETS = DATA / "sets"
OUT = DATA / "results"
SHARE = Path(os.environ.get("FLAKEID_SHARE", "Z:/scanresults"))


def dpath(rel):
    """bundle-relative path (as stored in sets/) -> absolute path string."""
    return str(DATA / rel)
