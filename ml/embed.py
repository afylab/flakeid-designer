"""Frozen pretrained-network features of the 100x pictures: python embed.py hbn|graphite resnet50|dinov2s
-> emb_<set>_<net>.npz (keys, X).  Whole picture, long side 512 px (ResNet: global average pool; DINOv2: CLS + mean
patch token), plus the same for the centre 50 % (the flake is near the centre)."""
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch, cv2
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent
which, net = sys.argv[1], sys.argv[2]
if which == "hbn":
    R = list(csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8")))
    items = [(r["key"], dpath(r["img"])) for r in R]
else:
    items = [(s["key"], dpath(s["img"])) for s in json.load(open(SETS / "graphite100.json")) if s["ok"]]
torch.set_num_threads(8)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
if net == "resnet50":
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity(); mult = 32
    def feat(x):
        return m(x)
else:
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"); mult = 14
    def feat(x):
        o = m.forward_features(x)
        return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)
m.eval()


def prep(im, long=512):
    h, w = im.shape[:2]
    s = long / max(h, w)
    H, W = max(mult, int(round(h * s / mult)) * mult), max(mult, int(round(w * s / mult)) * mult)
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1] / 255.0
    return torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))


keys, X, t0 = [], [], time.time()
with torch.no_grad():
    for i, (k, p) in enumerate(items):
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        c = im[h // 4:h - h // 4, w // 4:w - w // 4]
        v = torch.cat([feat(prep(im)), feat(prep(c, 384))], 1)[0].numpy()
        keys.append(k); X.append(v)
        if i % 100 == 0:
            print(i, len(items), f"{time.time() - t0:.0f} s", flush=True)
np.savez(OUT / f"emb_{which}_{net}.npz", keys=np.array(keys), X=np.array(X, np.float32))
print("done", len(keys), np.array(X).shape, f"{time.time() - t0:.0f} s")
