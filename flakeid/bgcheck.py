"""Substrate colour / brightness check against blank-substrate references (the flat-field measurements).

Every flat field (python -m flakeid.mm flat) records the bare-substrate counts it was measured on (info
substrate_counts, dark subtracted, with exposure_ms and bit_depth).  The colour of bare SiO2/Si depends on the oxide
thickness, so references are kept per objective and oxide in profiles/substrate_refs.json.  A run's substrate (its
most common colour, flat-field corrected, dark subtracted) is compared with the reference of the same objective and
oxide (+-OXIDE_TOL_NM):
  colour      R/G and B/G within COLOUR_TOL (8 %): a coloured filter left in, a lamp / white-balance change
  brightness  counts per ms within BRIGHT_TOL (25 %) of the reference: aperture, lamp intensity, a neutral filter
              (the morning chip_1 100x run, 2026-10-02: colour right, ~9x too dark)
The check only warns; results go to the log and a <name>_bgcheck.json next to the results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REFS = Path(__file__).parent / "profiles" / "substrate_refs.json"
COLOUR_TOL = 0.08          # bare-substrate colour moves with the oxide (the model: tens of % per 20 nm near 300 nm), so
                           # wafer-to-wafer oxide spread shifts it by several %; a filter / lamp change is 10-20 %
BRIGHT_TOL = 0.25
OXIDE_TOL_NM = 25.0


LOCAL_REFS = Path.home() / ".flakeid" / "substrate_refs.json"     # recorded with flat fields of measured wafers


def load_refs():
    out = []
    for f in (LOCAL_REFS, REFS):                  # (measured ones first: the nearest oxide wins, ties go to the first)
        try:
            out += json.loads(f.read_text(encoding="utf-8"))["refs"] if f.is_file() else []
        except (OSError, ValueError, KeyError):
            pass
    return out


def reference(objective, oxide_nm, flat_info=None):
    """The reference for this objective and oxide: from substrate_refs.json, else the flat field used (if its oxide is
    recorded and matches), else None."""
    best = None
    for r in load_refs():
        if r["objective"] == objective and oxide_nm is not None and abs(r["oxide_nm"] - float(oxide_nm)) <= OXIDE_TOL_NM:
            if best is None or abs(r["oxide_nm"] - oxide_nm) < abs(best["oxide_nm"] - oxide_nm):
                best = r
    if best is None and flat_info and flat_info.get("oxide_nm") is not None and oxide_nm is not None \
            and abs(float(flat_info["oxide_nm"]) - float(oxide_nm)) <= OXIDE_TOL_NM:
        best = dict(objective=objective, oxide_nm=flat_info["oxide_nm"], substrate_counts=flat_info["substrate_counts"],
                    exposure_ms=flat_info.get("exposure_ms"), bit_depth=flat_info.get("bit_depth", 12), source="flat field")
    return best


def check(substrate, objective, oxide_nm, exposure_ms=None, bit_depth=12, flat_info=None, frames=None):
    """Compare a measured substrate (dark-subtracted counts, R G B) with the reference.  frames: optional per-frame
    substrates (with their exposure) for a brightness check frame by frame.  Returns a dict with 'warnings'."""
    s = np.asarray(substrate, float)
    out = dict(objective=objective, oxide_nm=None if oxide_nm is None else round(float(oxide_nm), 1),
               substrate=[round(float(v), 1) for v in s], rg=round(float(s[0] / s[1]), 4), bg=round(float(s[2] / s[1]), 4),
               exposure_ms=exposure_ms, warnings=[])
    ref = reference(objective, oxide_nm, flat_info)
    if ref is None:
        out["reference"] = None
        out["warnings"].append(f"no {objective} substrate reference for {oxide_nm} nm oxide: colour / brightness not checked "
                               "(add one from a blank-chip flat field to profiles/substrate_refs.json)")
        return out
    r = np.asarray(ref["substrate_counts"], float) * 2.0 ** (bit_depth - ref.get("bit_depth", 12))
    out["reference"] = dict(oxide_nm=ref["oxide_nm"], source=ref.get("source"), rg=round(float(r[0] / r[1]), 4),
                            bg=round(float(r[2] / r[1]), 4), exposure_ms=ref.get("exposure_ms"))
    d_rg = out["rg"] / out["reference"]["rg"] - 1
    d_bg = out["bg"] / out["reference"]["bg"] - 1
    out.update(d_rg=round(d_rg, 4), d_bg=round(d_bg, 4))
    if abs(d_rg) > COLOUR_TOL or abs(d_bg) > COLOUR_TOL:
        out["warnings"].append(f"substrate colour off the {objective} reference ({ref['oxide_nm']} nm oxide): R/G {d_rg:+.1%}, "
                               f"B/G {d_bg:+.1%} (tolerance {COLOUR_TOL:.0%}): coloured filter in the light path, lamp or "
                               "white-balance change, or a different oxide than entered")
    ref_exp = ref.get("exposure_ms")
    exp = exposure_ms if exposure_ms is not None else ref_exp
    if ref_exp and exp:
        b = float(s[1] / exp) / float(r[1] / ref_exp)
        out["brightness"] = round(b, 3)
        if exposure_ms is None:
            out["brightness_note"] = "exposure not recorded: assumed equal to the reference's"
        if abs(b - 1) > BRIGHT_TOL:
            out["warnings"].append(f"substrate brightness {b:.2f}x the {objective} reference per ms of exposure "
                                   f"(tolerance ±{BRIGHT_TOL:.0%}): aperture, lamp intensity or a neutral filter changed"
                                   + ("; exposure not recorded, assumed the reference's" if exposure_ms is None else ""))
        if frames:
            # only frames whose most common colour is substrate (colour close to the reference): in a frame filled by
            # a flake (e.g. thick dark graphite) the most common colour is the flake
            rr = (r[0] / r[1], r[2] / r[1])
            sub_like = [(np.asarray(f, float), e) for f, e in frames
                        if abs(f[0] / f[1] / rr[0] - 1) <= 2 * COLOUR_TOL and abs(f[2] / f[1] / rr[1] - 1) <= 2 * COLOUR_TOL]
            bf = [float(f[1] / (e or exp)) / float(r[1] / ref_exp) for f, e in sub_like]
            bad = [i for i, v in enumerate(bf) if abs(v - 1) > BRIGHT_TOL]
            out["frames_checked"] = len(bf)
            out["frames_not_substrate"] = len(frames) - len(bf)
            out["frames_off"] = len(bad)
            if bad and len(bad) < len(bf):
                out["warnings"].append(f"{len(bad)} of {len(bf)} frames have the substrate brightness off by more than "
                                       f"{BRIGHT_TOL:.0%} (first: frame {bad[0] + 1}, {bf[bad[0]]:.2f}x): the light level "
                                       "changed during the run")
    return out


def report(res, log=print, prefix=""):
    if res.get("reference") is not None:
        log(f"{prefix}substrate check ({res['objective']}, ref {res['reference']['oxide_nm']} nm): R/G {res['rg']} "
            f"({res.get('d_rg', 0):+.1%}), B/G {res['bg']} ({res.get('d_bg', 0):+.1%})"
            + (f", brightness {res['brightness']}x" if "brightness" in res else ""))
    for w in res["warnings"]:
        log(f"{prefix}WARNING: {w}")
