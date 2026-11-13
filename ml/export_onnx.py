"""The graphite 100x keep network as one ONNX file for OpenCV's dnn module (no PyTorch: the designer build's Python 3.8).
python export_onnx.py -> results/graphite_net.onnx (+ .json: input size, the combination, the check against PyTorch)

In the graph: the 3 final setup C models (setupc_final_bundle_s*.pt), material fixed to graphite; input a BGR picture as
float 0-255 at 384 x 512 (cv2.resize of the bundle's natural 100x picture, INTER_AREA), 1 x 3 x 384 x 512; output the
keep logit averaged over the models and the four flips (as the held-out scores were made)."""
import json, time
import numpy as np, torch, torch.nn as nn, cv2
import setupc as S
from paths import OUT


class Deploy(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("onehot", torch.tensor([[1.0, 0.0]]))       # graphite

    def forward(self, bgr):
        x = (bgr[:, [2, 1, 0]] / 255.0 - self.mean) / self.std
        out = 0
        for xf in (x, x.flip(3), x.flip(2), x.flip(2).flip(3)):
            for n in self.nets:
                f = torch.cat([torch.flatten(n.b.avgpool(n.b.layer4(n.b.layer3(n.b.layer2(n.b.layer1(
                    n.b.maxpool(n.b.relu(n.b.bn1(n.b.conv1(xf))))))))), 1), self.onehot.expand(xf.shape[0], 2)], 1)
                out = out + n.keep(f)
        return out / (4 * len(self.nets))


nets = []
for s in (0, 1, 2):
    n = S.Net(len(S.TNAMES)).cpu()
    n.load_state_dict(torch.load(OUT / f"setupc_final_bundle_s{s}.pt", map_location="cpu"), strict=False)
    nets.append(n.eval())
D = Deploy(nets).eval()
x = torch.from_numpy(np.load(str(OUT / "cache_c.npy"), mmap_mode="r")[:4].astype(np.float32)).permute(0, 3, 1, 2).contiguous()
path = OUT / "graphite_net.onnx"
torch.onnx.export(D, x[:1], str(path), input_names=["bgr"], output_names=["keep_logit"], opset_version=13, dynamo=False)
with torch.no_grad():
    ref = D(x).numpy()[:, 0]
net = cv2.dnn.readNetFromONNX(str(path))
got, t0 = [], time.time()
for i in range(4):
    net.setInput(x[i:i + 1].numpy())
    got.append(float(net.forward()[0, 0]))
dt = (time.time() - t0) / 4
print("PyTorch", ref.round(4), "\nOpenCV ", np.round(got, 4), f"\nmax difference {np.abs(ref - got).max():.2e}; OpenCV CPU {dt:.2f} s per picture")
combo = json.load(open(OUT / "graphite_combo.json"))
json.dump(dict(input="BGR float 0-255, 1 x 3 x 384 x 512 (cv2.resize INTER_AREA of the natural 100x picture)",
               output="keep logit, mean of 3 models x 4 flips", models=[f"setupc_final_bundle_s{s}.pt" for s in (0, 1, 2)],
               combo=combo, check=dict(max_diff=float(np.abs(ref - got).max()), opencv=cv2.__version__, s_per_picture=round(dt, 2))),
          open(OUT / "graphite_net.json", "w"), indent=1)
