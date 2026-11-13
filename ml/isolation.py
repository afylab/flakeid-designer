"""Isolation test: share of non-substrate pixels around each 10x target (c400 crops, 2.18 um/px), in rings beyond the
flake (its 10x size), vs keep / wasted.  AUC of "less material around = keep", within runs and pooled."""
import json
from pathlib import Path
import numpy as np, cv2
from sklearn.metrics import roc_auc_score
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent
S = json.load(open(SETS / "set10.json"))
PX = 2.182
rows = []
for s in S:
    im = cv2.imread(dpath(s["c400"])).astype(np.float32)
    valid = im.sum(2) > 0
    sub = np.median(im[valid].reshape(-1, 3), axis=0)
    fg = (np.linalg.norm(im - sub, axis=2) > 14) & valid
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    H, W = fg.shape
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - H / 2, xx - W / 2) * PX
    fd = float(np.exp(s["hand"].get("log_flake_d", np.log(40))))
    r0 = fd / 2 + 15                                   # (the flake itself and its edge left out; flake_d: inscribed)
    f = {}
    for nm, a, b in (("near", r0, r0 + 60), ("far", r0 + 60, 200)):
        m = (r >= a) & (r < b) & valid
        f[nm] = float(fg[m].mean()) if m.sum() > 200 else np.nan
    rows.append(dict(material=s["material"], run=s["run"], y=s["y"], **f))
for mat in ("hbn", "graphite"):
    R = [r for r in rows if r["material"] == mat]
    y = np.array([r["y"] for r in R]); runs = np.array([r["run"] for r in R])
    for nm in ("near", "far"):
        v = np.array([r[nm] for r in R]); ok = np.isfinite(v)
        per = [roc_auc_score(y[ok & (runs == q)], -v[ok & (runs == q)]) for q in np.unique(runs)
               if 0 < y[ok & (runs == q)].sum() < (ok & (runs == q)).sum()]
        kept, wasted = np.nanmedian(v[y == 1]), np.nanmedian(v[y == 0])
        print(f"{mat:8s} {nm:4s} ring: material around kept {kept:.3f} vs wasted {wasted:.3f} (median share);  "
              f"AUC 'cleaner surroundings = keep' pooled {roc_auc_score(y[ok], -v[ok]):.3f}, within runs {np.mean(per):.3f}")
json.dump(rows, open(OUT / "isolation.json", "w"))
