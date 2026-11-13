"""key -> 100x picture of the graphite flakes in the pick model's fit (fit_screen100.decision_rows' choice of shot),
with label, run and the hand model's held-out score (focus_refit/scores_omit5b.csv = v3)."""
import csv, json, os, re, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[2]                     # (the repository)
FLAKEDATA = Path(os.environ.get("FLAKEID_FLAKEDATA", ROOT.parent / "flakedata"))
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
from make_afm_set import load_viewer_data, viewer_dir
from make_catalog_100x import h100, noise
from fit_screen100 import viewer_pieces

SCR = Path(__file__).parent.parent
RUNS = ["20261004_run_1", "20261005_run_1", "20261007_run_2", "graphite_20261019_run_1", "graphite_20261019_run_5", "pleasework_run_1"]
sc = {r["key"]: r for r in csv.DictReader(open(SCR / "focus_refit" / "scores_omit5b.csv", encoding="utf-8"))}
out = []
for run in RUNS:
    for piece in viewer_pieces(FLAKEDATA / run):
        D = load_viewer_data(piece)
        if D["material"] != "graphite":
            continue
        stem = D["name"].replace("_tiles", "")
        vd = viewer_dir(piece)
        for fid_s, imgs in h100(vd).items():
            key = f"{run}:{stem}-f{int(fid_s):04d}"
            if key not in sc:
                continue
            imgs = [e for e in imgs if not e.get("domain")]
            if len(imgs) > 1 and not all((vd / e["f"]).is_file() for e in imgs):
                imgs = imgs[:1]
            e = imgs[0] if len(imgs) == 1 else min(imgs, key=lambda e: (e.get("focus") not in ("auto", "manual", None),
                                                                        noise(vd / e["f"], cv2.IMREAD_REDUCED_GRAYSCALE_2)))
            p = vd / e["f"]
            s = sc[key]
            out.append(dict(key=key, run=run, y=int(s["decision"] != "discard"), img=str(p), ok=p.is_file(),
                            hand_z=float(s["held_out_z"])))
print(len(out), "of", len(sc), "found;", sum(o["ok"] for o in out), "pictures on disk;", sum(o["y"] for o in out), "kept")
json.dump(out, open(Path(__file__).parent / "graphite_set.json", "w"), indent=0)
