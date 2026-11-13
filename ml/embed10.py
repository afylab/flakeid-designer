"""Frozen network features of the 10x crops (set10.json): python embed10.py resnet50|dinov2s -> emb10_<net>.npz
(keys, X150, X400): each crop scaled to 224 px (ResNet: pooled; DINOv2: CLS + mean patch token)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, cv2
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent
net = sys.argv[1]
S = json.load(open(SETS / "set10.json"))
torch.set_num_threads(8)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
if net == "resnet50":
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    feat = m
else:
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    def feat(x):
        o = m.forward_features(x)
        return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)
m.eval()


def prep(p):
    im = cv2.imread(p)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)[..., ::-1] / 255.0
    return torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))


keys, A, B, t0 = [], [], [], time.time()
with torch.no_grad():
    for i in range(0, len(S), 16):
        ch = S[i:i + 16]
        A.append(feat(torch.cat([prep(dpath(s["c150"])) for s in ch])).numpy())
        B.append(feat(torch.cat([prep(dpath(s["c400"])) for s in ch])).numpy())
        keys += [s["key"] for s in ch]
        if i % 320 == 0:
            print(i, len(S), f"{time.time() - t0:.0f} s", flush=True)
np.savez(OUT / f"emb10_{net}.npz", keys=np.array(keys), X150=np.concatenate(A), X400=np.concatenate(B))
print("done", len(keys), f"{time.time() - t0:.0f} s")
