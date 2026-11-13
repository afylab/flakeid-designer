"""10x crops around every past 100x target (hBN and graphite) with its outcome (keep vs discard / dropped), from the
viewer tile pyramids (local full/<run> copy, else the share): c150/<key>.png (level max, 1.09 um/px, 144 px = 157 um)
and c400/<key>.png (level max-1, 2.18 um/px, 192 px = 419 um).  Writes set10.json (+ the 10x hand features)."""
import csv, glob, json, os, sys, collections
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).parent
TENX = HERE.parent / "tenx"
sys.path.insert(0, str(TENX))
from common10 import rcsv, derive, LIVE, FLAKEDATA

LOCAL = Path(FLAKEDATA) / "full"
SHARE = Path(os.environ.get("FLAKEID_SHARE", "Z:/scanresults"))
for d in ("c150", "c400"):
    (HERE / d).mkdir(exist_ok=True)

# (pool_hbn holds the derived features already; outcome / pool bookkeeping and the 100x h_* columns left out)
HBN_HAND = ["score", "log_best_d", "best_t", "log_flake_area", "log_flake_d", "aspect", "n_terraces", "n_ter_big",
            "best_area_frac", "ter_t_spread", "focus_um", "crowding", "frac_window", "frac_thick", "bt_solidity", "bt_fill",
            "bt_fullres_solidity", "bt_ring_similar", "bt_fit_residual", "bt_uniformity", "bt_edge_width_um",
            "bt_contrast_mag", "bt_log_edge_thick", "bt_log_thick_nb", "min_uniformity", "max_ring"]
J = {r["key"]: r for r in rcsv(TENX / "joined.csv")}
items = []
for r in rcsv(TENX / "pool_hbn.csv"):
    if r["targeted"] == "1" and r["outcome"] in ("keep", "discard", "dropped"):
        items.append(dict(material="hbn", run=r["run"], piece=r["piece"], fid=int(r["flake_id"]), outcome=r["outcome"],
                          hand={k: float(v) for k, v in r.items() if k in HBN_HAND and v not in ("", "nan")}))
for r in rcsv(TENX / "pool.csv"):
    if r["targeted"] != "1":
        continue
    tg = glob.glob(os.path.join(LIVE, r["run"], r["piece"], "*_tiles_live_targets.csv"))
    if not tg or "layers" not in open(tg[0], encoding="utf-8").readline():
        continue                                                     # (hBN pieces: in pool_hbn)
    key = f"{r['run']}:{r['piece']}-f{int(r['flake_id']):04d}"
    j = J.get(key)
    oc = "dropped" if j is None else {"1": "keep", "0": "discard"}.get(j.get("y"), "undecided")
    if oc == "undecided":
        continue
    items.append(dict(material="graphite", run=r["run"], piece=r["piece"], fid=int(r["flake_id"]), outcome=oc,
                      hand={k: v for k, v in derive(r).items() if isinstance(v, float)}))
print(collections.Counter((i["material"], i["outcome"]) for i in items))

pos = {}
for run, pc in {(i["run"], i["piece"]) for i in items}:
    for r in rcsv(glob.glob(os.path.join(LIVE, run, pc, "*_tiles_live_flakes.csv"))[0]):
        pos[(run, pc, int(r["flake_id"]))] = (float(r["x_px"]), float(r["y_px"]))


class Pyr:
    def __init__(self, run, pc):
        d = LOCAL / run / pc / "tiles" / "image"
        self.d = d if d.is_dir() else SHARE / run / pc / "tiles" / "image"
        self.top = max(int(p.name) for p in self.d.iterdir() if p.name.isdigit())
        self.cache = {}

    def tile(self, L, c, r):
        k = (L, c, r)
        if k not in self.cache:
            p = self.d / str(L) / f"{c}_{r}.jpg"
            self.cache[k] = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p.is_file() else None
        return self.cache[k]

    def crop(self, L, x, y, n):
        """n x n px centred on (x, y) in level-L pixels (black outside)."""
        out = np.zeros((n, n, 3), np.uint8)
        x0, y0 = int(round(x - n / 2)), int(round(y - n / 2))
        for r in range(max(0, y0 // 256), (y0 + n - 1) // 256 + 1):
            for c in range(max(0, x0 // 256), (x0 + n - 1) // 256 + 1):
                t = self.tile(L, c, r)
                if t is None:
                    continue
                tx, ty = c * 256, r * 256
                ax0, ay0 = max(x0, tx), max(y0, ty)
                ax1, ay1 = min(x0 + n, tx + t.shape[1]), min(y0 + n, ty + t.shape[0])
                if ax1 > ax0 and ay1 > ay0:
                    out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = t[ay0 - ty:ay1 - ty, ax0 - tx:ax1 - tx]
        return out


S, done = [], 0
for (run, pc), its in sorted(collections.defaultdict(list, {k: [i for i in items if (i["run"], i["piece"]) == k]
                                                       for k in {(i["run"], i["piece"]) for i in items}}).items()):
    P = Pyr(run, pc)
    for i in its:
        key = f"{run}:{pc}-f{i['fid']:04d}"
        x, y = pos[(run, pc, i["fid"])]
        a = HERE / "c150" / (key.replace(":", "__") + ".png")
        b = HERE / "c400" / (key.replace(":", "__") + ".png")
        if not a.is_file():
            cv2.imwrite(str(a), P.crop(P.top, x, y, 144))
        if not b.is_file():
            cv2.imwrite(str(b), P.crop(P.top - 1, x / 2, y / 2, 192))
        S.append(dict(i, key=key, y=int(i["outcome"] == "keep"), c150=str(a), c400=str(b)))
        done += 1
    P.cache.clear()
    print(run, pc, len(its), "local" if str(P.d).startswith(str(LOCAL)) else "share", done, flush=True)
json.dump(S, open(HERE / "set10.json", "w"), indent=0)
print(len(S), "crops")
