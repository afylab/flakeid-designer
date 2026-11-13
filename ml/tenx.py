"""10x targeting test: a network on the 10x crop around a target, taught by what the 100x picture of that flake showed,
against the deployed-style target model (flakeid/target_model.py, refit leave one run out by tools/fit_target_model.py).

    python tenx.py VARIANT [SEEDS]        VARIANT: none | rule | teacher -> results/z10n_<variant>.npz, then the report
    python tenx.py report VARIANT...      AUCs, paired bootstrap against the target model, keeps in the top K per piece

Flakes: every past 100x target (sets/target_labels_10x.csv: keep vs discard / dropped) with its 10x crop (images/c150:
144 px, 1.09 um/px, 157 um), plus the imaged flakes without an outcome (prep/crops10_share.py), which only take part
in stage 1.  One ResNet-50 for both materials (crop upsampled to 288 px, material one-hot next to the pooled features).
Per held-out run, nothing of that run is used (at targeting time its 100x pictures do not exist yet):
  stage 1 (rule, teacher): predict from the 10x crop the 100x screen features of the flake's shot (_regions.json) and
          the hBN defects v2 features; teacher: also the setup C keep score of the shot, from the setup C model fitted
          without the held-out run (setupc.py teacher -> teacher_bundle.npz).  Rank-gaussianised per material.
  stage 2: keep vs not, balanced per material, the stage-1 loss kept as an auxiliary (none: stage 2 only, from ImageNet).
Combined = mean of the per-run standardised network and target-model scores, as for the 100x tests."""
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, cv2, torchvision
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from paths import SETS, OUT, DATA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DEV = "cuda"
N = 288
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
TSCR = ["bubble_frac", "bubble_per_1000", "clean_d_um", "clean_frac", "clean_um2", "cut_by_frame", "debris_frac",
        "debris_n", "debris_per_1000", "dt_iqr", "dt_p95", "fine_frac", "fine_n", "focus", "g_norm", "orth_p90",
        "ridge_density", "roughness", "solidity", "steps_frac", "t_nm", "terrace_um2", "wrinkle_coherence",
        "wrinkle_free_ratio", "wrinkle_len_density", "wrinkle_n"]
TDEF = ["clean_d", "edge_um_per_1000", "off_frac", "tex_frac", "defect_frac", "tex_p90", "a_d"]
NT = len(TSCR) + len(TDEF) + 1                         # + the teacher score (last)


def num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def gauss(T, mat, sel=None):
    """rank-gaussianise each column within each material, using only the rows in sel (others: NaN)."""
    sel = np.ones(len(T), bool) if sel is None else sel
    O = np.full(T.shape, np.nan, np.float32)
    for m in (0, 1):
        for j in range(T.shape[1]):
            s = sel & (mat == m) & np.isfinite(T[:, j])
            if s.sum() > 20 and np.unique(T[s, j]).size > 1:
                O[s, j] = norm.ppf((rankdata(T[s, j]) - 0.5) / s.sum())
    return O


def load():
    L = list(csv.DictReader(open(SETS / "target_labels_10x.csv", encoding="utf-8")))
    share = json.load(open(DATA / "share100" / "index.json", encoding="utf-8"))
    scr = {}
    for s in share:                                    # a flake shot twice: the first shot's features
        scr.setdefault(s["key"], s)
    defs = {r["key"]: r for r in csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8"))}
    TS = np.load(OUT / "teacher_bundle.npz")
    tidx = {}
    for i, k in enumerate(TS["keys"]):
        tidx.setdefault(str(k), []).append(i)
    tmat = {str(k): int(m) for k, m in zip(TS["keys"], TS["mat"])}
    rows = [(r["key"], r["run"], r["piece"], int(r["material"] == "hbn"), int(r["outcome"] == "keep")) for r in L]
    have = {r[0] for r in rows}
    for k, s in scr.items():                           # imaged, no outcome: stage 1 only (y = -1)
        if k not in have and k in tmat and (DATA / "images" / "c150" / (k.replace(":", "__") + ".png")).is_file():
            rows.append((k, s["run"], s["piece"], tmat[k], -1))
    keys, runs, pieces, mat, y = (np.array(v) for v in zip(*rows))
    R = np.array([[num(scr.get(k, {}).get("screen", {}).get(c)) for c in TSCR]
                  + [num(defs.get(k, {}).get(c)) for c in TDEF] for k in keys], np.float64)
    fold = {str(f): i for i, f in enumerate(TS["folds"])}
    Tch = np.full((len(TS["folds"]), len(keys)), np.nan)
    for j, k in enumerate(keys):
        if k in tidx:
            Tch[:, j] = TS["S"][:, tidx[k]].mean(1)
    X = np.zeros((len(keys), 144, 144, 3), np.uint8)
    for i, k in enumerate(keys):
        im = cv2.imread(str(DATA / "images" / "c150" / (k.replace(":", "__") + ".png")))
        if im is not None:
            X[i] = im
    return keys, runs, pieces, mat, y, R, Tch, fold, X


def target_model_scores(keys):
    """held-out scores of the target model (tools/fit_target_model.py's own loading and leave-one-run-out fit)."""
    import fit_target_model as F
    data, _ = F.load(str(DATA / "live10"), str(SETS / "target_labels_10x.csv"))
    z = {}
    for mat, D in data.items():
        Xm = np.array([[d["f"][c] for c in F.FEATURES[mat]] for d in D], float)
        ym = np.array([d["y"] for d in D]); rm = np.array([d["run"] for d in D])
        for d, v in zip(D, F.loro(Xm, ym, rm, F.CS)):
            z[d["key"]] = v
    return np.array([z.get(k, np.nan) for k in keys])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.b.fc = nn.Identity()
        self.aux = nn.Linear(2050, NT)
        self.keep = nn.Linear(2050, 1)

    def forward(self, x, m):
        f = torch.cat([self.b(x), Fn.one_hot(m, 2).float()], 1)
        return self.aux(f), self.keep(f)[:, 0]


def gpu(X, b, train):
    x = torch.from_numpy(X[b]).to(DEV).permute(0, 3, 1, 2).float().flip(1) / 255.0
    if train:                                          # the 8 rotations / flips of the square crop, 8 px shifts
        k = np.random.randint(4)
        x = torch.rot90(x, k, (2, 3))
        if np.random.rand() < 0.5:
            x = x.flip(3)
        dy, dx = np.random.randint(-8, 9, 2)
        x = torch.roll(x, (int(dy), int(dx)), (2, 3))
    x = Fn.interpolate(x, (N, N), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


def run_stage(net, X, T, mat, y, idx, epochs, lr_b, lr_h, aux_w, keep_w, bs=64, log=""):
    opt = torch.optim.AdamW([{"params": net.b.parameters(), "lr": lr_b},
                             {"params": list(net.aux.parameters()) + list(net.keep.parameters()), "lr": lr_h}],
                            weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [lr_b, lr_h], total_steps=epochs * int(np.ceil(len(idx) / bs)),
                                              pct_start=0.15)
    pw = {m: (((mat[idx] == m) & (y[idx] == 0)).sum() / max(1, ((mat[idx] == m) & (y[idx] == 1)).sum())) for m in (0, 1)}
    Tt, mt, yt = torch.from_numpy(T).to(DEV), torch.from_numpy(mat).to(DEV), torch.from_numpy(y).float().to(DEV)
    net.train()
    for ep in range(epochs):
        perm, tot, t0 = np.random.permutation(idx), 0.0, time.time()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, k = net(gpu(X, b, True), mt[b])
            loss = torch.zeros((), device=DEV)
            if aux_w:
                t = Tt[b]; ok = torch.isfinite(t)
                loss = loss + aux_w * ((a.float() - torch.nan_to_num(t)) ** 2 * ok).sum() / ok.sum().clamp(min=1)
            if keep_w:
                lab = yt[b] >= 0
                if lab.any():
                    w = torch.where(yt[b] > 0, torch.tensor([pw[int(m)] for m in mat[b]], device=DEV, dtype=torch.float32), 1.0)[lab]
                    loss = loss + keep_w * (Fn.binary_cross_entropy_with_logits(k.float()[lab], yt[b][lab], reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sch.step()
            tot += float(loss) * len(b)
        if ep == epochs - 1:
            print(f"  {log} {epochs} epochs, loss {tot / len(idx):.4f}, {time.time() - t0:.0f} s per epoch", flush=True)


@torch.no_grad()
def predict(net, X, mat, idx):
    net.eval()
    mt = torch.from_numpy(mat).to(DEV)
    out = []
    for i in range(0, len(idx), 128):
        b = idx[i:i + 128]
        x = gpu(X, b, False)
        acc = 0
        for k in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + net(torch.rot90(x, k, (2, 3)), mt[b])[1].float()
        out.append((acc / 4).cpu())
    return torch.cat(out).numpy()


def std_per_run(z, runs):
    o = np.full(len(z), np.nan)
    for r in np.unique(runs):
        s = runs == r
        o[s] = (z[s] - z[s].mean()) / (z[s].std() + 1e-9)
    return o


def train_variant(variant, seeds):
    keys, runs, pieces, mat, y, R, Tch, fold, X = load()
    lab = y >= 0
    print(f"{len(keys)} flakes ({lab.sum()} with an outcome, {(y == 1).sum()} kept; {(mat == 1).sum()} hBN), "
          f"{len(set(runs[lab]))} runs; variant {variant}", flush=True)
    Z = np.full((len(seeds), len(keys)), np.nan)
    for si, seed in enumerate(seeds):
        for r in sorted(set(runs[lab])):
            torch.manual_seed(seed); np.random.seed(seed)
            tr = np.where(runs != r)[0]
            te = np.where((runs == r) & lab)[0]
            T = np.full((len(keys), NT), np.nan, np.float32)
            if variant != "none":                      # targets rank-gaussianised on the training runs only
                T[:, :-1] = gauss(R, mat, runs != r)
                if variant == "teacher":
                    T[:, -1] = gauss(Tch[fold.get(r, fold["ALL"])][:, None], mat, runs != r)[:, 0]
            net = Net().to(DEV)
            if variant != "none":
                run_stage(net, X, T, mat, y, tr, 10, 3e-4, 1e-3, 1.0, 0.0, log=f"{variant} seed {seed} {r} stage 1")
            trl = tr[y[tr] >= 0]
            run_stage(net, X, T, mat, y, trl, 8, 1e-4, 1e-3, 0.5 if variant != "none" else 0.0, 1.0,
                      log=f"{variant} seed {seed} {r} stage 2")
            Z[si, te] = predict(net, X, mat, te)
    np.savez(OUT / f"z10n_{variant}.npz", keys=keys[lab], runs=runs[lab], pieces=pieces[lab], mat=mat[lab], y=y[lab],
             z=Z[:, lab].mean(0), seeds=Z[:, lab])


def report(variants):
    base = None
    for v in variants:
        D = np.load(OUT / f"z10n_{v}.npz")
        if base is None:
            base = D
            tm = target_model_scores(D["keys"])
            print("flakes without a target-model score (no live table):", int(np.isnan(tm).sum()))
        assert (D["keys"] == base["keys"]).all()
        for m, name in ((1, "hBN"), (0, "graphite")):
            s = (D["mat"] == m) & np.isfinite(tm)
            y, runs, pieces = D["y"][s], D["runs"][s], D["pieces"][s]
            t = std_per_run(tm[s], runs); n = std_per_run(D["z"][s], runs); c = n + t
            rng = np.random.default_rng(0); g, gc = [], []
            for _ in range(2000):
                idx = np.concatenate([rng.choice(np.where(runs == r)[0], (runs == r).sum()) for r in np.unique(runs)])
                at = roc_auc_score(y[idx], t[idx])
                g.append(roc_auc_score(y[idx], n[idx]) - at); gc.append(roc_auc_score(y[idx], c[idx]) - at)
            at = roc_auc_score(y, t)
            print(f"{name} {v}: {s.sum()} targets, {y.sum()} kept.  AUC target model {at:.3f}, network {roc_auc_score(y, n):.3f} "
                  f"(gain {roc_auc_score(y, n) - at:+.3f}, CI {np.percentile(g, 2.5):+.3f} .. {np.percentile(g, 97.5):+.3f}), "
                  f"combined {roc_auc_score(y, c):.3f} (gain {roc_auc_score(y, c) - at:+.3f}, CI {np.percentile(gc, 2.5):+.3f} .. "
                  f"{np.percentile(gc, 97.5):+.3f})")
            for K in (5, 10, 20):
                kk = []
                for z in (t, n, c):
                    k = 0
                    for p in np.unique(pieces):
                        i = np.where(pieces == p)[0]
                        k += int(y[i[np.argsort(-z[i])][:K]].sum())
                    kk.append(k)
                print(f"    keeps among the top {K} per piece: target model {kk[0]}, network {kk[1]}, combined {kk[2]}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    torch.backends.cudnn.benchmark = True
    if sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        train_variant(sys.argv[1], [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0])
        report([sys.argv[1]])
