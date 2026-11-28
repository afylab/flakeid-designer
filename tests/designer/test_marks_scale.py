"""Export to KLayout, placing pictures by the chip's alignment marks (2026-11-27, DQD15): the 10x picture was placed at
1/3 of its scale (the diagonal squares repeat at 150, 225, 300, 450 um, so four of them also fit as a smaller square),
which then threw out the right 50x and 100x fits and left the design out.  A synthetic picture of the same template at
the 10x scale with two of the four crosses under a flake.  python3 -m pytest tests/designer -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flakeid_designer"))

cv2 = pytest.importorskip("cv2")
gdstk = pytest.importorskip("gdstk")
import marks_to_klayout as MK  # noqa: E402

SUB = np.array([130, 70, 120], float)      # BGR: substrate (285 nm oxide)
GOLD = np.array([110, 180, 230], float)
FLAKE = np.array([200, 170, 90], float)


def cross(cx, cy, sx, sy):
    """The two-L mark of pads_template.gds at (cx, cy), its corner pointing (sx, sy)."""
    pts = [(0, -7), (0, 0), (-7, 0), (-7, 2), (-2, 2), (-2, 7), (0, 7), (0, 0), (7, 0), (7, -2), (2, -2), (2, -7)]
    return [(cx + sx * x, cy + sy * y) for x, y in pts]


def template(path):
    lib = gdstk.Library()
    top = lib.new_cell("TOP")
    for sx in (1, -1):
        for sy in (1, -1):
            top.add(gdstk.Polygon(cross(45 * sx, 45 * sy, sx, sy), layer=50))
            for d, size in [(150, 4), (225, 4), (300, 4), (375, 4), (450, 10), (550, 10), (700, 10), (850, 10)]:
                top.add(gdstk.rectangle((sx * d - size / 2, sy * d - size / 2), (sx * d + size / 2, sy * d + size / 2),
                                        layer=50))
    top.add(gdstk.rectangle((-2000, -2000), (2000, -1900), layer=50))      # (metal: the layer with the most)
    lib.write_gds(str(path))


def picture(tpl_path, um_px, deg, shift_px, covered=(), cover_alpha=0.6, w=1376, h=1038, seed=1, bits=0):
    """The template's marks drawn at um_px, rotated by deg; marks at the `covered` layout positions under a flake;
    `bits` flakes in the gold's colour, 10-40 px, away from the middle (they outrank the small marks)."""
    lib = gdstk.read_gds(str(tpl_path))
    top = [c for c in lib.top_level() if c.name == "TOP"][0]
    ss = 8
    gold = np.zeros((h * 1, w * 1), np.uint8)
    a = np.exp(1j * np.radians(deg)) / um_px
    cx, cy = w / 2.0 + shift_px[0], h / 2.0 + shift_px[1]
    flake = np.zeros_like(gold)
    for p in top.get_polygons():
        z = (p.points @ np.array([1, 1j])) * a
        px = np.stack([cx + z.real, cy - z.imag], axis=1)
        if px[:, 0].max() < -50 or px[:, 0].min() > w + 50 or px[:, 1].max() < -50 or px[:, 1].min() > h + 50:
            continue
        cv2.fillPoly(gold, [np.round(px * ss).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=3)
    for c in covered:
        z = complex(*c) * a
        cv2.circle(flake, (int(cx + z.real), int(cy - z.imag)), int(25 / um_px), 255, -1)
    rng = np.random.default_rng(seed)
    for _ in range(bits):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        if abs(x - w / 2) < 150 and abs(y - h / 2) < 150:
            continue
        r = rng.uniform(5, 20)
        pts = [[x + r * np.cos(t) * rng.uniform(0.6, 1), y + r * np.sin(t) * rng.uniform(0.6, 1)]
               for t in np.linspace(0, 2 * np.pi, 7)[:-1]]
        cv2.fillPoly(gold, [np.array(pts, np.int32)], 255)
    g = (gold / 255.0)[..., None]
    im = SUB * (1 - g) + GOLD * g
    f = (flake / 255.0)[..., None] * cover_alpha
    im = im * (1 - f) + FLAKE * f
    im = cv2.GaussianBlur(im, (0, 0), 0.7) + np.random.default_rng(seed).normal(0, 2.0, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def tpl(tmp_path_factory):
    p = tmp_path_factory.mktemp("marks") / "pads_template.gds"
    template(p)
    feats, _ = MK.template_fiducials(str(p), "TOP", None)
    return p, feats


@pytest.mark.parametrize("seed", [2, 3, 8, 11])
def test_10x_scale_not_a_third(tpl, seed):
    # (these seeds: the code before 2026-11-27 took 0.26-0.71 um/px)
    path, feats = tpl
    im = picture(path, 1.0886, 0.5, (20, -40), covered=[(45, 45), (-45, 45)], seed=seed, bits=70)
    fit = MK.fit_picture(im, feats, log=lambda *a: None)
    assert fit is not None and MK.fit_good(fit)
    assert abs(fit["sim"].scale / 1.0886 - 1) < 0.01, fit["sim"].scale
    assert abs(fit["sim"].angle + 0.5) < 0.3                 # (drawn turned by +0.5 deg: picture -> layout is -0.5)


def test_hidden_cross_found_by_outline(tpl):
    path, feats = tpl
    T = np.array([f["c"] for f in feats])
    im = picture(path, 0.2168, 0.5, (30, 10), covered=[(-45, 45)])
    fit = MK.fit_picture(im, feats, log=lambda *a: None)
    assert fit is not None and fit["n"] == 3 and abs(fit["sim"].scale / 0.2168 - 1) < 0.01
    got, miss = MK.find_hidden_marks(im, fit, feats, T)
    assert [tuple(t) for t, _ in got] == [(-45.0, 45.0)] and not miss
    assert fit["n"] == 4 and fit["rms"] < 0.2 and MK.fit_strong(fit)


def test_no_mark_invented_where_there_is_none(tpl):
    path, feats = tpl
    T = np.array([f["c"] for f in feats])
    im = picture(path, 0.2168, 0.5, (30, 10), covered=[(-45, 45)], cover_alpha=1.0)     # an opaque flake
    fit = MK.fit_picture(im, feats, log=lambda *a: None)
    got, miss = MK.find_hidden_marks(im, fit, feats, T)
    assert not got and [tuple(t) for t in miss] == [(-45.0, 45.0)]


def test_scale_consensus_and_names():
    assert MK.name_mag("DQD15_stack_10x.jpg") == 10 and MK.name_mag("a_2.5x_b.png") == 2.5 and MK.name_mag("x.jpg") is None

    class S(object):
        def __init__(self, s):
            self.scale = s
    fits = [dict(name="d_5x.jpg", how="marks", sim=S(2.1776)), dict(name="d_10x.jpg", how="marks", sim=S(0.3628)),
            dict(name="d_50x.jpg", how="marks", sim=S(0.2168)), dict(name="d_100x.jpg", how="marks", sim=S(0.1086))]
    k = MK.scale_consensus(fits)
    assert k is not None and abs(k / 10.86 - 1) < 0.01
    assert MK.scale_consensus(fits[:2]) is None              # two that disagree: no majority


def test_overlay_letters_are_not_marks():
    # Stream's yellow text (bottom left) and scale bar (bottom right) gave DQD14's 100x four "marks" (2026-11-27)
    w, h = 1376, 1038
    cand = [(120, 1022, 8, 0.5, "mark"), (84, 988, 12, 0.5, "mark"), (1300, 1010, 20, 0.9, "square"),
            (298, 933, 127, 0.2, "mark"), (120, 1000, 128, 0.2, "mark"), (700, 1020, 10, 0.9, "square")]
    kept = MK.not_overlay(cand, w, h)
    assert [c[:2] for c in kept] == [(298, 933), (120, 1000), (700, 1020)]
