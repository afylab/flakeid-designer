"""The graphite set's natural 100x pictures from the designer bundles (local designer_g, else the share), as gimg/<key>.png
(lossless: older bundles hold them in WebP atlases, newer as JPEG files).  Updates graphite_set.json's img / src."""
import json, os
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).parent
OUT = HERE / "gimg"; OUT.mkdir(exist_ok=True)
LOCAL = Path(os.environ.get("FLAKEID_FLAKEDATA", Path(__file__).resolve().parents[3] / "flakedata"))
SHARE = Path(os.environ.get("FLAKEID_SHARE", "Z:/scanresults"))
S = json.load(open(HERE / "graphite_set.json"))
by_run = {}
for s in S:
    by_run.setdefault(s["run"], []).append(s)
for run, items in by_run.items():
    d = LOCAL / run / "designer_g"
    if not (d / "bundle.json").is_file():
        d = SHARE / run / "designer"
    F = {f["id"]: f for f in json.load(open(d / "bundle.json", encoding="utf-8"))["flakes"]}
    atl, n = {}, 0
    for s in items:
        f = F.get(s["key"])
        s["ok"] = False
        if f is None:
            continue
        pic = f.get("img")
        if isinstance(pic, dict):
            if pic["s"] not in atl:
                atl[pic["s"]] = cv2.imdecode(np.fromfile(str(d / pic["s"]), np.uint8), cv2.IMREAD_COLOR)
            im = atl[pic["s"]][pic["y"]:pic["y"] + pic["h"], pic["x"]:pic["x"] + pic["w"]]
            src = "atlas " + pic["s"].rsplit(".", 1)[-1]
        elif pic:
            im = cv2.imdecode(np.fromfile(str(d / pic), np.uint8), cv2.IMREAD_COLOR)
            src = "file " + pic.rsplit(".", 1)[-1]
        else:
            continue
        p = OUT / (s["key"].replace(":", "__") + ".png")
        cv2.imwrite(str(p), im)
        s.update(img=str(p), ok=True, src=src, shape=list(im.shape[:2])); n += 1
    print(run, d, n, "of", len(items), {x.get("src") for x in items})
json.dump(S, open(HERE / "graphite_set.json", "w"), indent=0)
print(sum(s["ok"] for s in S), "pictures")
