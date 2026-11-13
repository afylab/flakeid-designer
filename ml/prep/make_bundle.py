"""Portable copy of the network-test data for another PC: flakeid_ml_data next to the repository (FLAKEID_ML_DATA) with
sets/ (paths relative to the bundle root, plus each picture's place on the share), images/, results/."""
import csv, json, os, shutil
from pathlib import Path

HERE = Path(__file__).parent
BESIDE = Path(__file__).resolve().parents[3]                   # (the folder holding the repository)
OUT = Path(os.environ.get("FLAKEID_ML_DATA", BESIDE / "flakeid_ml_data"))
LOCAL = Path(os.environ.get("FLAKEID_FLAKEDATA", BESIDE / "flakedata"))
for d in ("sets", "images/graphite100", "images/hbn100", "images/c150", "images/c400", "results"):
    (OUT / d).mkdir(parents=True, exist_ok=True)


def cp(src, dst):
    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)


def share_rel(p):           # local flakedata (or full/) path -> path under scanresults/ on the share
    p = Path(p)
    for base in (LOCAL / "full", LOCAL):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            pass
    return ""


# graphite 100x
G = json.load(open(HERE / "graphite_set.json"))
for s in G:
    if s["ok"]:
        n = Path(s["img"]).name
        cp(Path(s["img"]), OUT / "images/graphite100" / n)
        s["img"] = f"images/graphite100/{n}"
json.dump(G, open(OUT / "sets/graphite100.json", "w", encoding="utf-8"), indent=0)

# hBN 100x (designer defects v2 features + decision)
src = LOCAL / "screen_features_view_v2.csv"
R = list(csv.DictReader(open(src, encoding="utf-8")))
for r in R:
    p = Path(r["img"])
    n = r["key"].replace(":", "__") + p.suffix
    cp(p, OUT / "images/hbn100" / n)
    r["share_path"] = share_rel(p)
    r["img"] = f"images/hbn100/{n}"
with open(OUT / "sets/hbn100.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(R[0]))
    w.writeheader(); w.writerows(R)

# 10x crops
S = json.load(open(HERE / "set10.json"))
for s in S:
    for c in ("c150", "c400"):
        n = Path(s[c]).name
        cp(Path(s[c]), OUT / "images" / c / n)
        s[c] = f"images/{c}/{n}"
json.dump(S, open(OUT / "sets/set10.json", "w", encoding="utf-8"), indent=0)

# results of the first test
for p in list(HERE.glob("*.npz")) + list(HERE.glob("log_*.txt")) + [HERE / n for n in
          ("examples.json", "isolation.json", "nn_summary.html", "crops10.log")]:
    cp(p, OUT / "results" / p.name)
print("graphite", sum(s["ok"] for s in G), "hbn", len(R), "10x", len(S),
      "missing share paths", sum(not r["share_path"] for r in R))
