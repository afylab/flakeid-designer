"""Setup C: pretrain a network to predict the rule measurements from the 100x picture, then fine-tune keep / discard.
One ResNet-50 (ImageNet V2 start) for both materials, the material as an input (one-hot next to the pooled features).

    python setupc.py cache                    labelled pictures -> results/cache_c.npy (384 x 512, uint8, both sets)
    python setupc.py pretrain bundle|share [--check]
                                              rule heads -> results/setupc_pre_<pool>.pt.  bundle: the 1495 labelled
                                              pictures; share: every 100x shot on the share (prep/share100.py) and the
                                              labelled pictures not among them, with the rule masks as dense targets.
                                              --check: first hold out one run per material and print its R2 per target
    python setupc.py embed POOL               pretrained pooled features -> emb_<set>_setupc_<pool>.npz (evaluate.py)
    python setupc.py final POOL [SEEDS]       fine-tuned on every run (no held-out run) -> setupc_final_<pool>_s<seed>.pt
    python setupc.py finetune POOL|imagenet [SEEDS]
                                              leave one run out, end to end, from those weights (+ the rule loss as an
                                              auxiliary) or from ImageNet only -> z_<set>_<pool>.npz

Rule targets (label-free), each rank-gaussianised per material, missing ones masked:
  - the 10x scan's measurements of the flake (set10.json "hand": terrace thickness and size, flake area, uniformity,
    contrast, ...; not its score, which was fitted on labels);
  - the designer defects v2 features (hBN, hbn100.csv);
  - the 100x screen features of the shot's _regions.json (terrace and clean area, clean_d, debris, steps, fine
    terraces, wrinkles, bubbles, focus, t_nm);
  - share pool: the shot's terrace, debris, step and fine-terrace masks (96 x 128) from a small head on layers 2 + 4.
The pretraining uses the held-out runs' pictures too: it sees no decision, only what the rules measure.
Fine-tuning: fixed schedule (no tuning on the held-out run), balanced BCE per material, flips and crops; held-out
scores with the four flips averaged.  The z files hold the same y / runs / hand_s as z_<set>.npz, so
boot.py <set>_<pool> compares with the hand model exactly as for the frozen networks."""
import csv, json, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, cv2, torchvision
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from paths import SETS, OUT, DATA, dpath

H, W = 384, 512
DEV = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
T10 = ["log_best_d", "best_t", "log_flake_area", "log_flake_d", "aspect", "n_terraces", "n_ter_big", "best_area_frac",
       "ter_t_spread", "focus_um", "crowding", "frac_thick", "bt_solidity", "bt_fill", "bt_fit_residual", "bt_uniformity",
       "bt_edge_width_um", "bt_contrast_mag", "bt_log_edge_thick", "bt_log_thick_nb", "bt_t_range", "bt_ext_vs_inscr"]
TDEF = ["clean_d", "edge_um_per_1000", "off_frac", "tex_frac", "defect_frac", "tex_p90", "a_d", "res_a_frac",
        "res_touch_frac", "res_bleed_um"]
TSCR = ["bubble_frac", "bubble_per_1000", "clean_d_um", "clean_frac", "clean_um2", "cut_by_frame", "debris_frac",
        "debris_n", "debris_per_1000", "dt_iqr", "dt_p95", "fine_frac", "fine_n", "focus", "g_norm", "orth_p90",
        "ridge_density", "roughness", "solidity", "steps_frac", "t_nm", "terrace_um2", "wrinkle_coherence",
        "wrinkle_free_ratio", "wrinkle_len_density", "wrinkle_n"]
TNAMES = T10 + TDEF + TSCR
MASKS = ["terrace", "debris", "steps", "fine"]
SETNAME = {0: "graphite", 1: "hbn"}
SHARE = DATA / "share100"


def num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def sources():
    """targets by flake key: 10x hand features, hBN defects v2, 100x screen features (share index, when prepared)."""
    hand10 = {s["key"]: s["hand"] for s in json.load(open(SETS / "set10.json"))}
    defs = {r["key"]: r for r in csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8"))}
    share = json.load(open(SHARE / "index.json", encoding="utf-8")) if (SHARE / "index.json").is_file() else []
    scr = {s["key"]: s["screen"] for s in share}

    def targets(key):
        h, d, s = hand10.get(key, {}), defs.get(key, {}), scr.get(key, {})
        return [num(h.get(c)) for c in T10] + [num(d.get(c)) for c in TDEF] + [num(s.get(c)) for c in TSCR]
    return targets, share


def load():
    """the labelled pictures of both sets: keys, runs, material (0 graphite, 1 hBN), y, image paths, raw targets."""
    targets, _ = sources()
    rows = []
    for s in json.load(open(SETS / "graphite100.json")):
        if s["ok"]:
            rows.append((s["key"], s["run"], 0, s["y"], dpath(s["img"]), targets(s["key"])))
    for r in csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8")):
        rows.append((r["key"], r["run"], 1, int(r["decision"] != "discard"), dpath(r["img"]), targets(r["key"])))
    keys, runs, mat, y, imgs, T = map(list, zip(*rows))
    return np.array(keys), np.array(runs), np.array(mat), np.array(y), imgs, np.array(T, np.float64)


def gauss(T, mat):
    """rank-gaussianise each target within each material (NaN where too few or constant)."""
    T = T.copy()
    for m in (0, 1):
        for j in range(T.shape[1]):
            sel = (mat == m) & np.isfinite(T[:, j])
            if sel.sum() > 20 and np.unique(T[sel, j]).size > 1:
                T[sel, j] = norm.ppf((rankdata(T[sel, j]) - 0.5) / sel.sum())
            else:
                T[sel, j] = np.nan
    return T.astype(np.float32)


class Net(nn.Module):
    def __init__(self, n_t, seg=False):
        super().__init__()
        b = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        b.fc = nn.Identity()
        self.b = b
        self.rule = nn.Linear(2050, n_t)
        self.keep = nn.Linear(2050, 1)
        self.seg = nn.ModuleDict(dict(l4=nn.Conv2d(2048, 256, 1), l2=nn.Conv2d(512, 256, 1),
                                      out=nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                                        nn.Conv2d(128, len(MASKS), 1)))) if seg else None

    def forward(self, x, m):
        b = self.b
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        l2 = b.layer2(b.layer1(x))
        l4 = b.layer4(b.layer3(l2))
        f = torch.cat([torch.flatten(b.avgpool(l4), 1), Fn.one_hot(m, 2).float()], 1)
        s = None
        if self.seg is not None:
            u = self.seg["l2"](l2) + Fn.interpolate(self.seg["l4"](l4), size=l2.shape[2:], mode="bilinear", align_corners=False)
            s = Fn.interpolate(self.seg["out"](u), scale_factor=2, mode="bilinear", align_corners=False)
        return f, self.rule(f), self.keep(f)[:, 0], s


def augment(x, msk=None):
    """the same random crop of 70 - 100 % (resized back) and flips for the pictures and their masks."""
    xo, mo = [], []
    for i in range(len(x)):
        s = np.random.uniform(0.7, 1.0)
        fy, fx, h, w = np.random.rand(), np.random.rand(), s, s
        fl = [d for d, p in ((3, 0.5), (2, 0.5)) if np.random.rand() < p]
        for t, out in ((x, xo), (msk, mo)):
            if t is None:
                continue
            ti = t[i:i + 1]
            Hh, Ww = ti.shape[2:]
            hh, ww = int(Hh * h), int(Ww * w)
            y0, x0 = int(fy * (Hh - hh)), int(fx * (Ww - ww))
            ti = Fn.interpolate(ti[:, :, y0:y0 + hh, x0:x0 + ww], (Hh, Ww), mode="bilinear", align_corners=False)
            out.append(ti.flip(fl) if fl else ti)
    return torch.cat(xo), (torch.cat(mo) if msk is not None else None)


def to_gpu(X, idx):
    x = torch.from_numpy(np.ascontiguousarray(X[np.sort(idx)][np.argsort(np.argsort(idx))])).to(DEV)
    return x.permute(0, 3, 1, 2).float().flip(1) / 255.0       # BGR -> RGB


def masked_mse(p, t):
    ok = torch.isfinite(t)
    return ((p - torch.nan_to_num(t)) ** 2 * ok).sum() / ok.sum().clamp(min=1)


def train(net, X, T, mat, y, idx, epochs, lr_b, lr_h, rule_w, keep_w, M=None, hasm=None, bs=32, log=""):
    """X, T, mat, y (and M, hasm: masks and which rows have them) indexed by idx; keep_w > 0 needs y."""
    heads = [p for n, p in net.named_parameters() if not n.startswith("b.")]
    opt = torch.optim.AdamW([{"params": net.b.parameters(), "lr": lr_b}, {"params": heads, "lr": lr_h}], weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(idx) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [lr_b, lr_h], total_steps=steps, pct_start=0.15)
    pw = {}                                            # balanced keep loss within each material
    for m in (0, 1):
        s = idx[mat[idx] == m]
        if len(s) and keep_w:
            pw[m] = (len(s) - y[s].sum()) / max(1, y[s].sum())
    if M is not None:                                  # rare masks (steps, fine terraces) weighted up, at most 20x
        fr = M[idx[hasm[idx]][:500]].reshape(-1, len(MASKS), M.shape[2] * M.shape[3]).mean((0, 2))
        mpw = torch.tensor(np.clip((1 - fr) / np.maximum(fr, 1e-4), 1, 20), device=DEV, dtype=torch.float32).view(1, -1, 1, 1)
    Tt, mt = torch.from_numpy(T).to(DEV), torch.from_numpy(mat).to(DEV)
    yt = torch.from_numpy(y).float().to(DEV) if keep_w else None
    net.train()
    for ep in range(epochs):
        perm, tot, t0 = np.random.permutation(idx), 0.0, time.time()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            mb = torch.from_numpy(M[b]).to(DEV).float() if M is not None else None
            x, mb = augment(to_gpu(X, b), mb)
            x = (x - MEAN) / STD
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, r, k, s = net(x, mt[b])
            loss = rule_w * masked_mse(r.float(), Tt[b]) if rule_w else torch.zeros((), device=DEV)
            if mb is not None and s is not None:
                hm = torch.from_numpy(hasm[b]).to(DEV)
                if hm.any():
                    loss = loss + rule_w * Fn.binary_cross_entropy_with_logits(
                        s.float()[hm], (mb[hm] > 0.5).float(), pos_weight=mpw)
            if keep_w:
                wt = torch.where(yt[b] > 0, torch.tensor([pw.get(int(m), 1.0) for m in mat[b]], device=DEV), 1.0)
                loss = loss + keep_w * Fn.binary_cross_entropy_with_logits(k.float(), yt[b], weight=wt, reduction="sum") / wt.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sch.step()
            tot += float(loss) * len(b)
        print(f"  {log} epoch {ep + 1}/{epochs} loss {tot / len(idx):.4f} {time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def predict(net, X, mat, idx, bs=64):
    """pooled features, rule predictions and keep logits, averaged over the four flips."""
    net.eval()
    mt = torch.from_numpy(mat).to(DEV)
    F, R, K = [], [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = (to_gpu(X, b) - MEAN) / STD
        acc = None
        for fl in ((), (3,), (2,), (2, 3)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x.flip(fl) if fl else x, mt[b])[:3]
            o = [v.float() for v in o]
            acc = o if acc is None else [a + v for a, v in zip(acc, o)]
        F.append(acc[0].cpu() / 4); R.append(acc[1].cpu() / 4); K.append(acc[2].cpu() / 4)
    return torch.cat(F).numpy(), torch.cat(R).numpy(), torch.cat(K).numpy()


def r2_report(R, T, mat):
    R = np.clip(R, -4, 4)                              # the targets are gaussianised; a few odd pictures (dark, thick)
    for m in (0, 1):                                   # get predictions of 20 - 50 that would swamp the R2
        s = mat == m
        line = []
        for j, n in enumerate(TNAMES):
            ok = np.isfinite(T[s, j])
            if ok.sum() > 10:
                t, p = T[s, j][ok], R[s, j][ok]
                line.append(f"{n} {1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum():.2f}")
        print(f"  held-out R2, {SETNAME[m]}: " + ", ".join(line), flush=True)


def std_per_run(z, runs):
    o = z.copy()
    for r in np.unique(runs):
        o[runs == r] = (z[runs == r] - z[runs == r].mean()) / (z[runs == r].std() + 1e-9)
    return o


def pool(pname, keys, runs, mat, X, Traw):
    """the pretraining pool: (X, T raw, mat, runs, M, hasm, keys).  share: every share shot, then the labelled pictures that
    are not among them (no masks)."""
    if pname == "bundle":
        return X, Traw, mat, runs, None, None, keys
    targets, share = sources()
    by_piece = {}                                      # shots left out of the designer: their piece's material, else
    for s in share:                                    # by thickness (graphite terraces are a few nm, hBN 10 - 100)
        if s["material"]:
            by_piece.setdefault((s["run"], s["piece"]), []).append(s["material"])
    for s in share:
        if not s["material"]:
            ms = by_piece.get((s["run"], s["piece"]))
            t = num(s["screen"].get("t_nm"))
            s["material"] = max(set(ms), key=ms.count) if ms else "graphite" if t < 8 else "hbn" if t >= 8 else None
    share = [s for s in share if s["material"] in ("graphite", "hbn")]
    rows = np.array([s["i"] for s in share])
    Xs = np.load(str(SHARE / "X.npy"), mmap_mode="r")
    Ms = np.load(str(SHARE / "M.npy"), mmap_mode="r")
    have = {s["key"] for s in share}
    extra = np.array([k not in have for k in keys])
    Xp = np.concatenate([Xs[rows], X[extra]])
    Tp = np.concatenate([np.array([targets(s["key"]) for s in share]), Traw[extra]])
    mp = np.concatenate([np.array([int(s["material"] == "hbn") for s in share]), mat[extra]])
    rp = np.concatenate([np.array([s["run"] for s in share]), runs[extra]])
    Mp = np.concatenate([Ms[rows], np.zeros((extra.sum(),) + Ms.shape[1:], np.uint8)])
    hasm = np.r_[np.ones(len(rows), bool), np.zeros(extra.sum(), bool)]
    if pname == "sharenm":                             # the same pool without the dense mask targets
        Mp, hasm = None, None
    print(f"pool {pname}: {len(rows)} share shots ({(mp[:len(rows)] == 0).sum()} graphite) + {extra.sum()} labelled "
          f"pictures not on the share", flush=True)
    return Xp, Tp, mp, rp, Mp, hasm, np.r_[[s["key"] for s in share], keys[extra]]


def main():
    cmd = sys.argv[1]
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    keys, runs, mat, y, imgs, Traw = load()
    print(f"{len(keys)} labelled pictures ({(mat == 0).sum()} graphite, {(mat == 1).sum()} hBN), {len(set(runs))} runs",
          flush=True)
    if cmd == "cache":
        X = np.zeros((len(imgs), H, W, 3), np.uint8)
        for i, p in enumerate(imgs):
            im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            X[i] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        np.save(str(OUT / "cache_c.npy"), X)
        json.dump(list(keys), open(OUT / "cache_c_keys.json", "w"))
        return
    X = np.load(str(OUT / "cache_c.npy"))
    assert json.load(open(OUT / "cache_c_keys.json")) == list(keys)
    allidx = np.arange(len(keys))
    pname = {"b": "imagenet"}.get(sys.argv[2], sys.argv[2])
    if cmd == "pretrain":
        Xp, Tp, mp, rp, Mp, hasm, _ = pool(pname, keys, runs, mat, X, Traw)
        Tp = gauss(Tp, mp)
        print(f"targets present: {np.isfinite(Tp).mean(0).round(2).tolist()}", flush=True)
        pidx = np.arange(len(Tp))
        ep = 15 if pname == "bundle" else 12
        if "--check" in sys.argv:                      # one run per material held out: does the network learn the rules?
            ho = np.isin(rp, ["graphite_20261019_run_5", "bn_20261020_run_5"])
            net = Net(len(TNAMES), Mp is not None).to(DEV)
            train(net, Xp, Tp, mp, None, pidx[~ho], ep, 3e-4, 1e-3, 1.0, 0.0, Mp, hasm, log="check")
            r2_report(predict(net, Xp, mp, pidx[ho])[1], Tp[ho], mp[ho])
        net = Net(len(TNAMES), Mp is not None).to(DEV)
        train(net, Xp, Tp, mp, None, pidx, ep, 3e-4, 1e-3, 1.0, 0.0, Mp, hasm, log="pretrain")
        torch.save(net.state_dict(), OUT / f"setupc_pre_{pname}.pt")
        return
    T = gauss(Traw, mat)                               # the labelled pictures' own targets, for the auxiliary loss
    if cmd == "embed":
        net = Net(len(TNAMES), pname == "share").to(DEV)
        net.load_state_dict(torch.load(OUT / f"setupc_pre_{pname}.pt"))
        F = predict(net, X, mat, allidx)[0][:, :2048]
        for m in (0, 1):
            np.savez(OUT / f"emb_{SETNAME[m]}_setupc_{pname}.npz", keys=keys[mat == m], X=F[mat == m])
    elif cmd == "final":                               # the deployable model: fine-tuned on every run, one per seed
        for seed in ([int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]):
            torch.manual_seed(seed); np.random.seed(seed)
            net = Net(len(TNAMES)).to(DEV)
            net.load_state_dict(torch.load(OUT / f"setupc_pre_{pname}.pt"), strict=False)
            train(net, X, T, mat, y, allidx, 8, 1e-4, 1e-3, 0.5, 1.0, log=f"final seed {seed}")
            torch.save(net.state_dict(), OUT / f"setupc_final_{pname}_s{seed}.pt")
    elif cmd == "teacher":                             # keep scores of every share shot for the 10x test (tenx.py):
        Xs, _, ms, _, _, _, ks = pool("share", keys, runs, mat, X, Traw)     # one model per left-out run, + all runs
        ns = len(Xs)                                   # share shots, then the labelled pictures not on the share
        folds = sorted(set(runs)) + ["ALL"]
        S = np.zeros((len(folds), ns), np.float32)
        for fi, r in enumerate(folds):
            torch.manual_seed(0); np.random.seed(0)
            net = Net(len(TNAMES)).to(DEV)
            net.load_state_dict(torch.load(OUT / f"setupc_pre_{pname}.pt"), strict=False)
            train(net, X, T, mat, y, allidx[runs != r], 8, 1e-4, 1e-3, 0.5, 1.0, log=f"teacher {r}")
            S[fi] = predict(net, Xs, ms, np.arange(ns))[2]
        np.savez(OUT / f"teacher_{pname}.npz", keys=ks[:ns], folds=np.array(folds), S=S, mat=ms[:ns])
    elif cmd == "finetune":
        seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
        Z = np.zeros((len(seeds), len(keys)))
        for si, seed in enumerate(seeds):
            for r in sorted(set(runs)):
                torch.manual_seed(seed); np.random.seed(seed)
                te, tr = allidx[runs == r], allidx[runs != r]
                net = Net(len(TNAMES)).to(DEV)
                if pname != "imagenet":
                    net.load_state_dict(torch.load(OUT / f"setupc_pre_{pname}.pt"), strict=False)
                train(net, X, T, mat, y, tr, 8, 1e-4, 1e-3, 0.5 if pname != "imagenet" else 0.0, 1.0,
                      log=f"{pname} seed {seed} {r}")
                Z[si, te] = predict(net, X, mat, te)[2]
        z = Z.mean(0)
        for m in (0, 1):
            ref = np.load(OUT / f"z_{SETNAME[m]}.npz")
            sel = np.where(mat == m)[0]
            order = sel[np.argsort(keys[sel])]         # evaluate.py's order: sorted keys
            assert (ref["runs"] == runs[order]).all() and (ref["y"] == y[order]).all()
            zm, rm = z[order], runs[order]
            k = f"ft_{pname}"
            both = std_per_run(zm, rm) + ref["hand_s"]
            print(f"{SETNAME[m]} {k}: AUC {roc_auc_score(ref['y'], zm):.3f}, + hand {roc_auc_score(ref['y'], both):.3f}, "
                  f"hand {roc_auc_score(ref['y'], ref['hand_s']):.3f}", flush=True)
            np.savez(OUT / f"z_{SETNAME[m]}_{pname}.npz", y=ref["y"], runs=rm, hand_s=ref["hand_s"],
                     **{k: zm, f"{k}_both": both, f"{k}_seeds": Z[:, order]})


if __name__ == "__main__":
    main()
