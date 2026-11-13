"""Graphite at 100x: the thickness, and the layer count of few-layer flakes, from all three channels (flakeid.thick100,
graphite's own fitted 100x model since 2026-11-12).  The colours are measured ones: GR13' 20261102_1 F796's staircase
(1L - 5L on one flake), thick graphite of the same chip, and GR14' 20261103_1's purple and blue flakes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from flakeid import thick100 as T  # noqa: E402

OX = 84.5
STAIRS = [(1, [-0.083, -0.102, -0.081]), (2, [-0.149, -0.183, -0.154]), (3, [-0.216, -0.264, -0.208]),
          (4, [-0.274, -0.325, -0.253]), (5, [-0.324, -0.386, -0.299])]


def test_staircase_counts_every_step():
    assert [T.layers_of(c, "graphite", OX) for _, c in STAIRS] == [n for n, _ in STAIRS]
    assert T.layers_of([-0.377, -0.437, -0.326], "graphite", OX) == 6        # (F171.b)


def test_three_layers_is_not_two():
    r = T.fit([-0.216, -0.264, -0.208], "graphite", OX, 1.01)
    assert r["t"] == 1.0 and r["alts"] == [] and r["agree"] == "yes" and not r["poor"]
    r2 = T.fit([-0.149, -0.183, -0.154], "graphite", OX, 0.67)
    assert r2["t"] == 0.7 and "layers" not in r and "layers" not in r2       # (thickness in nm only)


def test_thick_graphite_keeps_the_colour_fit():
    for c in ([-0.385, -0.25, -0.055], [0.042, 0.336, 0.469], [-0.638, -0.732, -0.685]):   # (~11 nm, ~18 nm, past 10L)
        assert T.layers_of(c, "graphite", OX) is None
        r = T.fit(c, "graphite", OX)
        assert r["t"] > T.LAYERS_MAX * T.LAYER_NM


def test_only_graphite_on_thin_oxide():
    c = STAIRS[2][1]
    assert T.layers_of(c, "hbn", OX) is None and T.layers_of(c, "graphite", 285.0) is None


def test_bundle_refits_graphite_from_the_stored_colour():
    import make_catalog_100x as MC
    stored = dict(t=0.8, alts=[], agree="yes", poor=False, contrast=STAIRS[2][1], oxide_nm=OX, t10=1.0)
    assert MC.t100x(dict(t100=stored), "graphite") == dict(t=1.0, alts=[], agree="yes", poor=False)
    assert MC.t100x(dict(t100=stored), "hbn")["t"] == 0.8                    # (other materials: as measured)
    assert MC.t100x(dict(t100=dict(t=0.8)), "graphite")["t"] == 0.8          # (no colour kept)


# GR14' 20261103_1 (10x labels): purple flakes 3 - 5 nm, blue ones 8 - 9 nm.  Before 2026-11-12 (the hBN-calibrated
# camera's model, the count from green) both read 10 - 13 nm.
PURPLE = {"F005_5nm": [-0.583, -0.621, -0.403], "F010_4nm": [-0.547, -0.609, -0.418], "F006_3nm": [-0.471, -0.549, -0.398]}
BLUE = {"F011_9nm": [-0.496, -0.415, -0.173], "F007_8nm": [-0.54, -0.489, -0.243]}


def test_purple_is_thin_and_blue_is_thick():
    for k, c in PURPLE.items():
        r = T.fit(c, "graphite", 84.0)
        lab = float(k.split("_")[1][:-2])
        assert abs(r["t"] - lab) <= 1.5 and r["alts"] == [] and r["misfit"] < 0.05, (k, r)
    for k, c in BLUE.items():
        r = T.fit(c, "graphite", 84.0)
        lab = float(k.split("_")[1][:-2])
        assert abs(r["t"] - lab) <= 1.5 and r["alts"] == [] and r["misfit"] < 0.05, (k, r)
    assert max(T.fit(c, "graphite", 84.0)["t"] for c in PURPLE.values()) < 6.0 < min(
        T.fit(c, "graphite", 84.0)["t"] for c in BLUE.values())


def test_every_reading_uses_all_three_channels():
    """Two colours with the same green but different red and blue are not the same thickness."""
    g = -0.55
    purple, blue = [-0.47, g, -0.40], [-0.56, g, -0.33]
    tp, tb = T.fit(purple, "graphite", 84.0)["t"], T.fit(blue, "graphite", 84.0)["t"]
    assert tp < 4.0 and tb > 6.0


def test_one_100x_model_everywhere():
    import terraces100 as TR
    from flakeid_designer import manual as MAN
    m = T.model100("graphite", 84.0)
    assert TR.contrast_model("graphite", 84.0) is m and MAN._model("graphite", 84.0) is m
    assert m.na == 0.95 and T.model100("hbn", 285.0).na == T.NA_100X
