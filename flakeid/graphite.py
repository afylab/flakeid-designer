"""Graphite / graphene 100x targets: layer number from measured contrast, thinnest terrace of each flake first.

Layer numbers come from a measured calibration (profiles/graphite_layers.json): the green contrast of 1-4 layer
graphene, read off the peaks of the terrace-contrast histogram of a real chip (the peaks are sharp: +-0.005-0.011),
not from the thin-film model, plus a near-neutral colour check (R ~ G, B a bit weaker) that rejects ~12-15 nm
graphite of the same green contrast.  The calibration holds for the wafers it was measured on: chips entered as its
nominal oxide (nominal_nm), or, without a nominal value, a fitted oxide within oxide_tol_nm of its oxide_nm.  (Until
2026-10-04 only the fitted oxide was compared: on pleasework_2 the fit never settled, the oxide stayed at 90 nm, the
table was not used, and 63 1-2 layer flakes fell into the "thicker" tier, where < 20 um or score < 0.75 drops them.)

Selection (unlike hBN, bigger is not better: large graphite layers are hard to place in a heterostructure without
shorting to other layers):
  tier 1  1-2 layer terraces, at least min_d_graphene um across
  tier 2  3-4 layer terraces, same size cut
  tier 3  thicker graphite with score >= min_score, at least min_d um across
Within a tier the flake score orders the flakes (1 and 2 layers together), except that terraces wider than max_d um go
last.
Each flake gives one target: its thinnest qualifying terrace.
Ribbons (is_ribbon: long, narrow, isolated terraces of a few nm, which many devices need) are targeted whatever their
inscribed diameter or score, with RIBBON_PLACES places of a capped list kept for them (their flake's target is the
ribbon terrace); the inscribed-diameter cut and the score, both driven by width, dropped nearly all of them before.
With a cap on the number of targets, THICK_FRACTION of it is kept for tier 3 (reserve): graphite of a few nm and more
is needed for gates as much as graphene is (on a graphite chip measured with the layer table, 1-4 layer terraces alone
filled 79 of 80 places, 2026-10-04); either group's unused places go to the other.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

LAYER_NM = 0.335
THICK_FRACTION = 0.6            # of a capped target list, for thicker graphite (tier 3); the user asked for 50-75 %
# ribbons, from Jordan's review of 48 long, narrow terraces of graphite_20261019 (2026-10-21): useful = an isolated
# ribbon (not a ribbon-shaped part of a larger flake), straight, nearly parallel edges, 0.7-10 nm, ~10-26 um wide (5-10
# would be even better).  With these limits 10 of his 11 useful ones pass and 8 of 31 not useful (curved edges or a
# neighbour attached: not visible in the 10x outline); 7-14 ribbon flakes per piece, 1-3 of them targeted before.
# max_iso: the flake's inscribed diameter (everything connected to the terrace) at most this many times the terrace's
RIBBON = dict(t_min=0.5, t_max=10.5, min_len=80.0, min_w=5.0, max_w=30.0, min_aspect=4.0, max_iso=4.5, min_solidity=0.45)
RIBBON_PLACES = 10              # places of a capped target list kept for ribbons (0: no ribbon rule)
CAL_PATH = Path(__file__).parent / "profiles" / "graphite_layers.json"


def load_calibration(path=CAL_PATH):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def layers_from_contrast(c, cal: dict):
    """Layer number (1..len(peaks)) for a contrast vector (R, G, B): green within the tolerance of a measured peak and
    the colour near-neutral as graphene is (R/G and B/G within the calibration's ranges), else None.  The colour check
    matters: ~12-15 nm graphite has the green contrast of 1-4 layers but a much darker red and a positive blue."""
    r, g, b = (float(v) for v in c[:3])
    if g >= 0:
        return None
    lo, hi = cal.get("r_over_g", [0.0, 99.0])
    if not lo <= r / g <= hi:
        return None
    lo, hi = cal.get("b_over_g", [-99.0, 99.0])
    if not lo <= b / g <= hi:
        return None
    return layers_from_green(g, cal)


def layers_from_green(g: float, cal: dict):
    """Layer number from the green contrast alone (use layers_from_contrast: green alone confuses 1-4 layers with
    ~12-15 nm graphite)."""
    for n, (mu, sd) in enumerate(zip(cal["green"], cal["sd"]), 1):
        if abs(g - mu) <= max(cal.get("k_sd", 3.0) * sd, cal.get("min_tol", 0.02)):
            return n
    return None


def _f(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def is_ribbon(t, t_nm, p=RIBBON):
    """A terrace row (*_terraces.csv) of thickness t_nm is a ribbon: long, narrow, isolated, not ragged."""
    L, W, d = _f(t.get("length_um"), 0.0), _f(t.get("width_um"), 0.0), _f(t.get("inscribed_diameter_um"), 0.0)
    fd, sol = _f(t.get("flake_inscribed_d_um"), 1e9), _f(t.get("solidity"), 0.0)
    return (p["t_min"] <= t_nm <= p["t_max"] and L >= p["min_len"] and p["min_w"] <= W <= p["max_w"] and L >= p["min_aspect"] * W
            and d > 0 and fd <= p["max_iso"] * d and sol >= p["min_solidity"])


def calibration_applies(cal, oxide_nm, oxide_nominal=None):
    if not cal:
        return False
    if oxide_nominal is not None and cal.get("nominal_nm") is not None:
        return abs(float(oxide_nominal) - float(cal["nominal_nm"])) <= 0.5
    return oxide_nm is not None and abs(float(oxide_nm) - cal["oxide_nm"]) <= cal.get("oxide_tol_nm", 6.0)


def select(terraces_csv, oxide_nm, min_score=0.75, max_targets=None, exclude=(), min_d=20.0,
           min_d_graphene=10.0, max_d=80.0, cal=None, log=print, oxide_nominal=None, ribbons=True):
    """Target rows (flake-table style dicts, one per flake) in priority order.  oxide_nominal: the oxide entered for
    the piece (the wafers' nominal thickness), which decides whether the layer calibration applies."""
    cal = cal or load_calibration()
    if cal and not calibration_applies(cal, oxide_nm, oxide_nominal):
        what = (f"entered as {float(oxide_nominal):g} nm" if oxide_nominal is not None else
                f"of unknown nominal oxide, fitted {float(oxide_nm or 0):.1f} nm (the table needs {cal['oxide_nm']:g} +- {cal.get('oxide_tol_nm', 6.0):g})")
        log(f"WARNING: the graphite layer calibration is for {cal.get('nominal_nm', cal['oxide_nm']):g} nm wafers, this piece is "
            f"{what}: layer numbers not used, so 1-4 layer flakes are only targeted if at least {min_d:g} um across with score "
            f">= {min_score:g} (measure the peaks on this oxide: tools/graphite_layers.py)")
        cal = None
    with open(terraces_csv, newline="") as fh:
        T = list(csv.DictReader(fh))
    best = {}
    counts = {}
    rib = {}                                            # flake -> (its ribbon terrace's rank key, row, layers)
    for t in T:
        flags = set(filter(None, (t.get("quality") or "").split(";")))
        if flags & set(exclude):
            continue
        d = _f(t.get("inscribed_diameter_um"), 0.0)
        n = layers_from_contrast([_f(t.get("contrast_r")), _f(t.get("contrast_g")), _f(t.get("contrast_b"))], cal) if cal else None
        if ribbons and is_ribbon(t, n * LAYER_NM if n else _f(t.get("thickness_nm"), -1.0)):
            # (the most isolated, then the longest ribbon of the flake)
            rk = (round(_f(t.get("flake_inscribed_d_um"), 1e9) / max(d, 1e-6), 1), -_f(t.get("length_um"), 0.0))
            if t["flake_id"] not in rib or rk < rib[t["flake_id"]][0]:
                rib[t["flake_id"]] = (rk, t, n)
        if n is not None:
            if d < min_d_graphene:
                continue
            tier = 1 if n <= 2 else 2
        else:
            if d < min_d or not _f(t.get("score"), -1) >= min_score:
                continue
            tier = 3
        counts[tier] = counts.get(tier, 0) + 1
        key = (tier, n or 99, d > max_d, -_f(t.get("score"), -1))       # a flake's own terraces: the thinnest first
        fid = t["flake_id"]
        if fid not in best or key < best[fid][0]:
            best[fid] = (key, t, n)
    rows = []

    def row(t, n, oversize=False, ribbon=False):
        return dict(flake_id=t["flake_id"], terrace_id=t.get("terrace_id"), score=t.get("score"), layers=n or "",
                    best_thickness_nm=round(n * LAYER_NM, 2) if n else t.get("thickness_nm"),
                    best_terrace_d_um=t.get("inscribed_diameter_um"), area_um2=t.get("flake_area_um2"),
                    target_x_px=t["target_x_px"], target_y_px=t["target_y_px"],
                    stage_x_um=t.get("stage_x_um"), stage_y_um=t.get("stage_y_um"),
                    quality=t.get("quality", "") + (";ribbon" if ribbon else ""), oversize=oversize, ribbon=ribbon,
                    ribbon_key=rib[t["flake_id"]][0] if ribbon else None)
    # between flakes: tier, then oversize last, then the score (1 and 2 layers rank together: a faint, low-score
    # monolayer must not push out a clean bilayer, as flake 34 of pleasework_2 was, 2026-10-04)
    for key, t, n in sorted(best.values(), key=lambda b: (b[0][0], b[0][2], b[0][3])):
        if t["flake_id"] in rib:                        # (a ribbon flake: its target is the ribbon)
            rows.append(row(rib[t["flake_id"]][1], rib[t["flake_id"]][2], ribbon=True))
        else:
            rows.append(row(t, n, oversize=key[2]))
    have = {r["flake_id"] for r in rows}
    for fid, (rk, t, n) in sorted(rib.items(), key=lambda kv: kv[1][0]):
        if fid not in have:
            rows.append(row(t, n, ribbon=True))
    nl = {k: sum(1 for r in rows if r["layers"] == k) for k in range(1, 5)}
    log("graphite targets: " + ", ".join(f"{v} x {k}L" for k, v in nl.items() if v)
        + f", {sum(1 for r in rows if not r['layers'])} thicker; {sum(r['oversize'] for r in rows)} wider than "
        f"{max_d:g} um (ranked last in their tier); {len(rib)} ribbons (long, narrow, isolated, "
        f"{RIBBON['t_min']:g}-{RIBBON['t_max']:g} nm), {len(rib) - sum(1 for f in rib if f in best)} of them only by the ribbon rule")
    return rows[:max_targets] if max_targets else rows


def reserve(rows, max_targets, thick_fraction=THICK_FRACTION, log=print, ribbon_places=RIBBON_PLACES):
    """At most max_targets of select()'s rows: first up to ribbon_places ribbons (the most isolated, then the longest),
    then thick_fraction of the other places for thicker graphite (rows without a layer number), the rest for 1-4
    layers, each group in its own priority order; places one group cannot fill go to the other.  Ribbons and layered
    rows first in the result (the microscope orders the visits by stage path anyway)."""
    if not max_targets or len(rows) <= max_targets:
        return rows
    rb = sorted((r for r in rows if r.get("ribbon")), key=lambda r: r["ribbon_key"])[:min(ribbon_places, max_targets)]
    if rb:
        taken = {r["flake_id"] for r in rb}
        rest = reserve([r for r in rows if r["flake_id"] not in taken], max_targets - len(rb), thick_fraction,
                       log=log, ribbon_places=0)
        log(f"graphite targets: {len(rb)} places for ribbons (of {sum(1 for r in rows if r.get('ribbon'))})")
        return rb + rest
    thin = [r for r in rows if r.get("layers")]
    thick = [r for r in rows if not r.get("layers")]
    n_thick = int(round(thick_fraction * max_targets))
    n_thin = max_targets - n_thick
    if len(thick) < n_thick:
        n_thin, n_thick = max_targets - len(thick), len(thick)
    if len(thin) < n_thin:
        n_thick, n_thin = min(len(thick), max_targets - len(thin)), len(thin)
    log(f"graphite targets capped at {max_targets}: {n_thin} of {len(thin)} 1-4 layer, {n_thick} of {len(thick)} thicker "
        f"({100 * thick_fraction:.0f} % of the places kept for thicker graphite)")
    return thin[:n_thin] + thick[:n_thick]
