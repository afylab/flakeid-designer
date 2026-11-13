"""The scan program's 100x thickness (flakeid.thick100, in each 100x picture's _regions.json as t100) carried into the
designer bundle's target terrace (tools/make_catalog_100x.t100x)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
pytest.importorskip("cv2")
import make_catalog_100x as MC  # noqa: E402


def test_t100x():
    R = dict(t100=dict(t=216.0, misfit=0.12, alts=[87.5], alt_misfits=[0.14], t10=85.0, agree="alt", poor=False,
                       contrast=[0.1, 0.2, 0.3], na=0.9, oxide_nm=290.0, n_px=5000), glue=0.04)
    assert MC.t100x(R) == dict(t=216.0, alts=[87.5], agree="alt", poor=False)
    assert MC.t100x({}) is None                                  # (a run from before py-2026-10-30a)
    assert MC.t100x(dict(t100=None, t100_error="ValueError: x")) is None
    assert MC.t100x(dict(t100=dict(t=3.5))) == dict(t=3.5, alts=[], agree=None, poor=False)


def test_thicker_than_wanted_at_100x_scores_0():
    """(2026-11-12) A flake thicker than the wanted range at 100x is kept with a 100x score of 0 (automatically
    discarded in the designer); the screening's own score is kept."""
    fl = [dict(terraces=[dict(id="a", p_keep=0.8, tier="keep", t_over=100.0)]),
          dict(terraces=[dict(id="b", p_keep=0.6, tier="keep")])]
    assert MC.want_max(fl) == 1
    a, b = fl[0]["terraces"][0], fl[1]["terraces"][0]
    assert (a["p_keep"], a["tier"], a["p_keep_screen"]) == (0.0, "reject", 0.8)
    assert (b["p_keep"], b["tier"]) == (0.6, "keep") and "p_keep_screen" not in b
