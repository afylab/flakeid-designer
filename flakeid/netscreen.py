"""Graphite 100x screening by an image network, in place of the graphite pick model when its model file is there.

The network (ml/setupc.py, ml/export_onnx.py; ml/README.md "Graphite screening with the network", 2026-10-26) is a
ResNet-50 pretrained to predict the rule measurements from the 100x picture, then fine-tuned on the designer's
keep / discard decisions: 3 such models on 4 flips each, averaged, in one ONNX file run by OpenCV's dnn module (no
PyTorch).  Held out by run on 486 decided graphite flakes: AUC 0.866 against the pick model's 0.793; with the same
95 %-of-keeps cutoff 180 instead of 113 discards screened out for 10 instead of 8 keeps, and a keep tier of 103 keeps
/ 37 discards instead of 127 / 95.  Combined with the pick model the pick model's weight is 0, so the network
decides alone.

The profile's "net" block (profiles/screen100_graphite.json) gives the file, the calibration P(keep) =
sigmoid(coef * logit + intercept) and the tiers: "reject" (screened out, shown in the designer) below reject_below,
"keep" from keep_above, else "review"; a flake whose 100x terrace does not fit the 10x one stays "review".  The model
file is large (~280 MB) and not in the repository: models/<file> next to flakeid/, or FLAKEID_NET_DIR.  Without it
(or when OpenCV cannot read it) nothing changes: the pick model's p_keep / tier stay.  A scored terrace keeps the
pick model's as p_rules / tier_rules and gets p_net (the network's logit) and screen="net".

    python -m flakeid.netscreen <designer folder or bundle.json> [--pictures DIR ...] [-o out.json]

re-scores the graphite flakes of a built bundle from its own natural 100x pictures (x100_nat_* files or the older
WebP sheets).  Run it on a copy on this computer, not on the share.  Python 3.8.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROFILE = Path(__file__).parent / "profiles" / "screen100_graphite.json"
MODEL_DIRS = [Path(__file__).resolve().parents[1] / "models"]
_NET = {}


def model_dirs():
    d = os.environ.get("FLAKEID_NET_DIR")
    return ([Path(d)] if d else []) + MODEL_DIRS


class NetScreen:
    """The graphite network and its calibration; ok False (with why) when the profile has no "net" block, the file is
    missing or OpenCV cannot read it."""

    def __init__(self, profile=PROFILE):
        self.ok, self.why, self.cfg, self.net = False, "", {}, None
        try:
            self.cfg = json.loads(Path(profile).read_text(encoding="utf-8")).get("net") or {}
        except (OSError, ValueError) as e:
            self.why = f"no profile ({e})"
            return
        if not self.cfg:
            self.why = f"no network in {Path(profile).name}"
            return
        path = next((d / self.cfg["file"] for d in model_dirs() if (d / self.cfg["file"]).is_file()), None)
        if path is None:
            self.why = f"model file {self.cfg['file']} not found (in {', '.join(str(d) for d in model_dirs())})"
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as e:
            self.why = f"OpenCV cannot read {path}: {str(e).strip().splitlines()[-1]}"
            return
        self.path, self.ok = path, True

    def logit(self, bgr):
        """The network's keep logit of one natural 100x picture (BGR uint8, any size)."""
        w, h = self.cfg.get("size", [512, 384])
        x = cv2.resize(bgr, (int(w), int(h)), interpolation=cv2.INTER_AREA).astype(np.float32)
        self.net.setInput(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        return float(self.net.forward().reshape(-1)[0])

    def p(self, z):
        v = self.cfg["coef"] * z + self.cfg["intercept"]
        return 1.0 / (1.0 + math.exp(-max(min(v, 50.0), -50.0)))

    def tier(self, p):
        return "reject" if p < self.cfg["reject_below"] else ("keep" if p >= self.cfg["keep_above"] else "review")


def get():
    """The loaded network (one per process)."""
    if "g" not in _NET:
        _NET["g"] = NetScreen()
    return _NET["g"]


def score_flakes(flakes, picture, screen=None, log=None):
    """p_keep / tier of every graphite flake (terrace a) whose picture(f) (BGR or None) can be read, from the network;
    nothing when the network is not available.  Returns {tier: count} of the flakes scored."""
    ns = screen or get()
    gs = [f for f in flakes if f.get("material") == "graphite" and f.get("terraces")]
    if not gs:
        return {}
    if not ns.ok:
        if log:
            log(f"graphite screening: the pick model ({ns.why})")
        return {}
    n, miss, ms = {}, 0, []
    for f in gs:
        try:
            bgr = picture(f)
        except (OSError, ValueError, cv2.error):
            bgr = None
        if bgr is None:
            miss += 1
            continue
        t0 = time.perf_counter()
        z = ns.logit(bgr)
        ms.append((time.perf_counter() - t0) * 1000)
        t = f["terraces"][0]
        if "tier_rules" not in t:                       # (the pick model's, kept once: a re-score does not overwrite it)
            t["p_rules"], t["tier_rules"] = t.get("p_keep"), t.get("tier")
        p = ns.p(z)
        tier = ns.tier(p)
        if (t.get("seed") or {}).get("fits") is False:
            tier = "review"                 # (its terrace does not fit the 10x one: the screen may have judged the wrong thing)
        t.update(p_keep=round(p, 4), tier=tier, p_net=round(z, 3), screen="net")
        n[tier] = n.get(tier, 0) + 1
    if log:
        log(f"graphite screening (network {ns.path.name}, {np.median(ms) if ms else 0:.0f} ms per picture): "
            + ", ".join(f"{v} {k}" for k, v in sorted(n.items())) + f" of {len(gs)} flakes"
            + (f"; {miss} without a readable picture keep the pick model's tier" if miss else ""))
    return n


def bundle_picture(folder, pictures=()):
    """picture(f) for a built bundle: its natural 100x picture, a file in the bundle's folder (or one of pictures) or a
    piece of an older bundle's WebP sheet."""
    dirs = [Path(folder)] + [Path(q) for q in pictures]
    sheets = {}

    def read(p):
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

    def picture(f):
        pic = f.get("img")
        if isinstance(pic, dict):
            if pic["s"] not in sheets:
                sheets[pic["s"]] = read(Path(folder) / pic["s"])
            s = sheets[pic["s"]]
            return None if s is None else s[pic["y"]:pic["y"] + pic["h"], pic["x"]:pic["x"] + pic["w"]]
        if isinstance(pic, str):
            name = Path(pic).name
            p = next((q for d in dirs for q in (d / pic, d / name, d / "100x" / name) if q.is_file()), None)
            return read(p) if p is not None else None
        return None
    return picture


def _replace(tmp, dst, tries=6):
    """os.replace with retries (Windows: a virus scanner or indexer may hold the file a moment)."""
    for i in range(tries):
        try:
            os.replace(str(tmp), str(dst))
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def rescore(bundle_path, pictures=(), out=None, log=print):
    """The graphite flakes of a built bundle scored by the network; bundle.json written (out, default in place)."""
    p = Path(bundle_path)
    p = p / "bundle.json" if p.is_dir() else p
    b = json.loads(p.read_text(encoding="utf-8"))
    n = score_flakes(b["flakes"], bundle_picture(p.parent, pictures), log=log)
    if not n:
        log(f"{p}: nothing scored, bundle unchanged")
        return n
    dst = Path(out) if out else p
    tmp = dst.with_name("." + dst.name + ".part")
    tmp.write_text(json.dumps(b, separators=(",", ":")), encoding="utf-8")
    _replace(tmp, dst)
    log(f"-> {dst}")
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="a designer folder (bundle.json in it) or a bundle.json, on this computer")
    ap.add_argument("--pictures", action="append", default=[], help="a folder holding the bundle's x100_nat_* pictures "
                    "(also <folder>/100x; repeat)")
    ap.add_argument("-o", "--out", help="write the re-scored bundle.json here (default: in place)")
    a = ap.parse_args(argv)
    rescore(a.bundle, a.pictures, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
