# Neural-network screening experiments

This is research code, not part of the scan program. It tests whether trained image models can
improve on the hand-built scores:
- **100x pick model** (keep or discard a flake from its 100x picture);
- **10x targeting score** (which flakes the scope images at 100x).

The first test ran on the lab Windows PC (CPU only) on 2026-10-23. Setup C (pretrain on the rule
measurements, then fine-tune) ran on the RTX 5060 PC on 2026-10-24.

## Setup C: pretrain on rule measurements, then fine-tune (2026-10-24)

**Result: the best network so far for both materials.** Leave one run out, 3 seeds averaged, gain in AUC
over the hand model (paired bootstrap, `boot.py`):

| | Graphite | hBN |
|---|---|---|
| First test: frozen ImageNet ResNet-50 / DINOv2 + hand | +0.075 (+0.045 .. +0.106) | +0.009 (−0.014 .. +0.034) |
| Fine-tuned from ImageNet only (1 seed) + hand | +0.033 (+0.004 .. +0.062) | +0.019 (−0.004 .. +0.042) |
| **Rule-pretrained, fine-tuned, + hand** (`z_<set>_bundle.npz`, `ft_bundle_both`) | **+0.078 (+0.052 .. +0.104)** | **+0.036 (+0.019 .. +0.054)** |
| … the network alone (`ft_bundle`) | +0.087 (+0.043 .. +0.131) | +0.022 (−0.008 .. +0.052) |
| Pretrained on all share shots + masks, fine-tuned, + hand (`z_<set>_share.npz`) | +0.067 (+0.041 .. +0.093) | +0.037 (+0.021 .. +0.053) |
| Frozen rule-pretrained features, PCA 128, + hand (`setupc_share_128_both`) | +0.070 (+0.042 .. +0.096) | +0.018 (+0.005 .. +0.031) |

AUCs (pooled, per-run standardised): graphite hand 0.781, network alone 0.868, combined 0.859; hBN hand
0.837, network alone 0.859, combined 0.874.

- **The pretraining is what makes fine-tuning work.** From ImageNet weights alone, fine-tuning on
  1,500 labelled pictures does worse than the frozen features. Starting from the rule-pretrained
  weights, it does better.
- **hBN gains significantly for the first time.** The network adds to the hand model even though the
  hand model uses the same defects features the network was pretrained on.
- **Graphite:** the network alone beats the hand model (+0.087). Combining with the hand model adds
  nothing more.
- **The share shots don't help (yet).** The paired difference, share pool minus labelled pool, combined:
  - graphite −0.011 (−0.024 .. +0.002);
  - hBN +0.001 (−0.009 .. +0.011).

  The pool grew from 1,495 to 2,674 pictures (about 1,100 unlabelled shots), and the terrace, debris,
  step and fine-terrace masks were added. The masks don't matter: a check without them gives the same
  held-out R².
- **What the network learns.** Held-out-run R² of the rule measurements, labelled pool, predicted
  from the picture alone:
  - thickness 0.82 (graphite), 0.91 (hBN);
  - terrace size 0.74 / 0.48 and flake size 0.63 / 0.49;
  - hBN defect fractions about 0.6–0.7.

  The share-pool check gives lower R² (hBN thickness 0.2). It holds out a whole run's shots
  (unlabelled pieces and a few dark, thick-flake pictures with extreme outputs included), so it isn't
  comparable. Open question; it doesn't show in the fine-tune AUCs.
- **Caveats:**
  - one architecture (ResNet-50 at 384 × 512) and one fixed schedule (8 epochs, no tuning on the
    held-out run);
  - seed spread of about ±0.01 in AUC;
  - the pretraining sees the held-out runs' pictures and rule measurements, but never a decision.
- **Run times on the RTX 5060:** pretraining about 3 min (labelled pool) or 5 min (share pool).
  Leave-one-run-out fine-tuning is 15 min per seed (11 folds × 8 epochs).

**Setup details** (`setupc.py`):
- one ResNet-50 for both materials, the material as a one-hot next to the pooled features;
- pretraining targets:
  - the 10x scan's flake measurements (`set10.json` `hand`, not its score, which was fitted on labels);
  - the hBN defects v2 features;
  - the 26 screen features of each shot's `_regions.json`;
- each target rank-gaussianised per material, masked where missing, 12–15 epochs;
- fine-tuning: balanced BCE per material, the rule loss kept as an auxiliary (weight 0.5), flips and
  70–100 % crops; held-out scores averaged over the four flips.

## 10x targeting network taught by the 100x pictures (2026-10-24)

**Question:** can a network on the 10x crop around a target rank the targets better than the target
model (`flakeid/target_model.py`)?

**Test:** `tenx.py`, leave one run out over 10 runs and 2,352 past targets (279 kept), 2 seeds.
- **Input:** the 10x crop, 144 px at 1.09 µm/px (157 µm), upsampled to 288 px for a ResNet-50.
- **Teaching:** before learning keep/discard, the network learns to predict from the 10x crop what the
  flake's 100x picture showed:
  - `rule`: the 100x screen features of the shot and the hBN defects v2 features;
  - `teacher`: those, plus the setup C keep score from a setup C model fitted without the held-out run.
- **No leakage:** nothing of the held-out run is used, neither its 100x pictures nor its measurements.
  At targeting time they don't exist yet.
- **Unlabelled shots:** 152 imaged flakes without an outcome also take part in the teaching
  (`prep/crops10_share.py`).

| | hBN (1,646 targets, 123 kept) | Graphite (706, 156 kept) |
|---|---|---|
| Target model (refit leave one run out, deployed code) | 0.734 | 0.685 |
| Network, ImageNet start (`none`) | 0.737 | 0.719 |
| Network taught the 100x rules (`rule`) | 0.757 (+0.022, −0.021 .. +0.063) | **0.753 (+0.067, +0.017 .. +0.120)** |
| `rule` network + target model | **0.770 (+0.035, +0.010 .. +0.059)** | **0.746 (+0.061, +0.036 .. +0.087)** |
| Network taught rules + setup C score (`teacher`), + target model | 0.764 (+0.030) | 0.748 (+0.063) |
| Ceiling: setup C score of the flake's 100x picture (held out) | 0.86 | 0.82 |

AUC is pooled over per-run standardised scores; gains are over the target model with the bootstrap
95 % CI.

**Keeps among the top 10 targets per piece:**
- hBN: target model 35, `rule` + target model 43;
- graphite: target model 40, `rule` network alone 50.

**What the table shows:**
- **There is a gain:**
  - graphite about +0.06;
  - hBN about +0.035, combined only.
- **The first test missed it** because the network was frozen and only ImageNet-trained.
- **What does the work is teaching the 100x rule measurements.** The setup C score adds nothing on
  top of them.
- **Most of the gap remains:** a 10x network closes about a third of the way to what the 100x picture
  shows.

**Caveats:**
- **Selection bias:** every target was already chosen by the 10x score, so this measures re-ranking
  among chosen flakes, not finding flakes the score skips.
- 2 seeds and one fixed schedule.
- A new camera changes the 10x pictures. The model would need retraining, but the rule measurements
  for the teaching come for free with any scan.

**Cost:** one forward pass of ResNet-50 at 288 px per candidate.
- About 35 ms on this PC's CPU (4 threads) and under 1 ms on the GPU.
- A few hundred candidates per piece is about 10 s on a modern CPU. A smaller network
  (ResNet-18) would be about 3× faster.
- Training: about 25 min on the RTX 5060.

## Shot-quality check (2026-10-25)

**Question:** how many 100x shots don't show the targeted flake well, and how much of the screening
error do they cause? Run with `shotcheck.py`, over all 2,566 share shots.

**Faults checked** (from `_regions.json` and the picture):
- the terrace fills the whole frame;
- no flake of the 10x colour near the target;
- the terrace doesn't match the 10x terrace;
- the flake is more than 25 µm off-centre;
- the picture is defocused: its strongest edges are soft relative to its brightness range, below 0.25.

The scan's own `focus` number can't be used for defocus. It is low for large, clean flakes with few
edges in view, not only for blurred pictures.

**Results:**
- **13 % of shots have a fault.**
  - Graphite runs of Oct 4–7: 34–93 % (`20261004_run_5`: 70 of 76 shots are empty frames).
  - The Oct 13–20 runs: 3–12 %.
  - Defocused: 40 shots, 15 % of `20261007_run_2` graphite and 5 % of `20261014_run_1`.
  - Flake not found, not matching or off-centre: only recorded since Oct 19.
- **The scope already fixes most of this.** Since the Oct 19 runs, the scan does three things
  (`centre_on_target`, `retake_failed` in `flakeid/scope/scan.py`):
  - matches each 100x frame to the target's 10x crop and re-centres (175–206 per run);
  - searches a ring around the target when the centre is bare substrate;
  - retakes unfocused flakes at the end of the piece.

  Faulty shots fell from 18 % (Oct 4–14 runs: 254 empty frames, 42 defocused) to 5.5 % (Oct 19–20:
  11 empty, 1 defocused). The rest is mostly a flake that doesn't match the 10x terrace or sits
  off-centre. The not-found / no-match / off-centre flags only exist since Oct 19.

  Still open: frames where the 10x match is unsure and the picture is taken as it is. The logs show
  12–74 per run, the most in `graphite_20261019_run_1`, probably faint thin graphite.
- **They cost scope time, not screening accuracy.**
  - 82 % were dropped before review, 15 % discarded, 7 kept.
  - They are 2 of graphite's 10 confident screening errors and 0 of hBN's 41. A confident error is a
    discard in the run's top 10 % or a keep in its bottom 10 %, by the held-out combined score.
  - Leaving them out changes the screening AUC by under 0.01.
  - So the screening errors are about the flakes and the decisions.
- **Thin graphite breaks the terrace finder.**
  - Faint, low-contrast graphite shots are kept more often than the rest: 45 % vs 33 % of reviewed ones.
  - Several kept 1–2 layer flakes got a whole-frame terrace.
  - So the rule measurements are wrong exactly for the most wanted graphite.

The flags are in `results/shotcheck.json` and the report in `results/log_shotcheck.txt`.

## Graphite screening with the network (2026-10-26)

**Scripts:** `graphite_tiers.py`, `setupc.py final`, `export_onnx.py`, `score_bundles.py`.

**Held-out comparison** (all scores and cutoffs held out by run; 486 decided flakes, 166 kept). The
tiers are as now: screened out below the cutoff that passes 95 % of the training runs' keeps, keep
from P = 0.5, review in between.

| | AUC | Screened out | Review | Keep tier |
|---|---|---|---|---|
| Pick v3 (now) | 0.793 | 113 discards, 8 keeps | 143 (31 keeps) | 127 keeps, 95 discards |
| Network | 0.866 | 180 discards, 10 keeps | 156 (53 keeps) | 103 keeps, 37 discards |
| Pick + network | 0.862 | 149 discards, 10 keeps | 188 (54 keeps) | 102 keeps, 37 discards |

**The pick model adds nothing once the network is there.** Fitted over all runs, the combination's
weight on the pick logit is 0.000. So the deployable rule is the network alone, calibrated:
- P(keep) = sigmoid(0.550 × net logit − 0.394);
- screened out below 0.120, keep from 0.5;
- stored in `results/graphite_combo.json`.

**The deployable model.**
- 3 final models, each fine-tuned on every run, run on 4 flips each and averaged
  (`setupc_final_bundle_s*.pt`).
- Exported as `results/graphite_net.onnx` (282 MB), with the BGR 0–255 input at 384 × 512.
- OpenCV's dnn module reproduces PyTorch to within 3 × 10⁻⁶ and takes 0.39 s per picture on this
  PC's CPU. That check used OpenCV 5.0; the kit's 4.8 still has to be checked on the build PC.

**All built graphite flakes, scored** (578 in 7 runs on the share; `results/graphite_bundle_scores.json`;
nothing written to the share):
- **Tier counts:** now 230 screened out / 109 review / 239 keep; with the network 241 / 153 / 184.
- **Decided flakes:** their tier moves are optimistic, because the final model was trained on those
  decisions. The held-out table above is the fair comparison.
- **Undecided flakes (92, 74 of them in `20261004_run_1`):**
  - 24 move out of "screened out" (23 to review, 1 to keep);
  - 4 move in;
  - 10 drop from keep to review.

## Graphite layer counts and colour ambiguity (2026-10-26)

**Question:** do we need a model-based per-pixel layer fit, or a network, for few-layer graphite and
for colours that occur at two thicknesses?

**How the 100x thickness is stored.** Terrace a's thickness and layer count in the designer bundles
are copied from the 10x analysis: 327 of 330 graphite terraces carry the identical value, and the 100x
region terraces are measured relative to terrace a. So the bundles can't serve as an independent check.

**Independent check with `layers100.py`.** It measures terrace a's contrast in its own 100x JPEG,
inside the scope's terrace polygon, against the frame's bare substrate. The recent graphite runs
(Oct 19) were used.
- **1L and 2L graphite form tight, separate groups:**
  - 1L: green −10.1 %, interquartile spread 0.2 %, n 87;
  - 2L: green −19.2 %, spread 0.8 %, n 102;
  - JPEG gamma makes these larger than the model's linear contrast.
- **186 of 189 10x layer calls match.** In the 3 disagreements the 100x terrace isn't on the flake
  (substrate or a different piece).
- **So counting thin graphite layers at 10x (contrast peaks) is reliable.** A per-pixel layer fit
  could only help with finding faint flakes and drawing their boundaries, not with counting.
- **Thick graphite (8–12 nm at 10x) is noisy.** The 100x green contrast spreads widely (interquartile
  range up to 8 %) and gets lighter with thickness, past the contrast maximum. Spearman between 10x nm
  and 100x darkness is −0.64.

**Colour that occurs at two thicknesses (hBN ~20 nm vs ~150 nm):**
- At 10x the calibrated model puts them within about 10 % in colour. At 100x (NA 0.9) they differ by
  about 38 %, so the 100x picture resolves the branch physically.
- The 10x branch classifier (`flakeid/branch.py`, 148 user labels) is 93 % accurate in cross-validation.
- Only 16 of 1,646 past hBN targets were ambiguous at 10x, too few to train or test a network on. If
  the problem grows, the 100x colour can label ambiguous 10x terraces automatically. That would give a
  10x model (rules or network) many more examples than the 148 hand labels.

## Results of the first test (2026-10-23)

Every number is leave-one-run-out:
- each run is scored by a model fitted on the other runs only;
- the regularisation strength is chosen in a nested loop over the training runs.

AUC is the chance that a random kept flake ranks above a random discarded one (0.5 = no skill).
"Combined" is the mean of the per-run-standardised scores of the network and hand models.

### 100x picture → keep or discard

| Set | Flakes | Kept | Hand model | Network alone | Combined | Gain over hand, 95 % CI |
|---|---|---|---|---|---|---|
| Graphite | 486 | 166 | 0.793 (deployed v3) | ResNet-50, PCA 32: 0.804 | 0.856 | +0.075 (+0.045 .. +0.106) |
| hBN | 1009 | 124 | 0.824 (refit on defects v2) | DINOv2-S, PCA 128: 0.771 | 0.846 | +0.009 (−0.014 .. +0.034) |

- **Network:** a frozen ImageNet network supplies the picture features (no training on our pictures).
  On top: PCA, then an L2 logistic model with balanced class weights.
- **Graphite:** the combination helps. At 95 % of keeps passing, it screens out 171 of 320 discards
  instead of 132.
- **hBN:** the gain is within noise.

### 10x crop around a target → kept at 100x or not

| Set | Targets | Kept | 10x score + terrace size | All 10x numbers, refit on outcome | Network + numbers |
|---|---|---|---|---|---|
| hBN | 1646 | 123 | 0.698 | 0.731 | 0.737 (within noise of numbers) |
| Graphite | 706 | 156 | 0.577 | 0.697 | 0.718 (within noise of numbers) |

**Crops** (made by `prep/crops10.py`):
- `c150`: 144 px at 1.09 µm/px, 157 µm across;
- `c400`: 192 px at 2.18 µm/px, 419 µm across, showing the neighbourhood.

**What the 10x results show:**
- **The network adds nothing at 10x,** and the wide crop is worse (graphite −0.043, significant).
- **Refitting the 10x numbers on the 100x outcome beats the deployed score.** The deployed score was
  fitted on 10x user labels. Gain from refitting:
  - graphite: CI +0.071 .. +0.170;
  - hBN: CI +0.003 .. +0.061.
- This refit needs no network and is the most direct way to use 100x decisions to improve 10x
  targeting. It is not deployed yet: the scorer runs on the scope PC (Python 3.8).

### Isolation

**Measure:** the share of non-substrate pixels in rings around the target, computed on the c400
crops by `isolation.py`.

| Material | Ring | AUC for "cleaner surroundings = keep" |
|---|---|---|
| hBN | near | about 0.49 |
| hBN | far | about 0.59 (weak) |
| Graphite | far | 0.39–0.45 |

For graphite this is below 0.5: kept graphite has *more* material around it. Thin flakes sit next to
thicker pieces from the same peel.

The summary page with example pictures is `results/nn_summary.html` in the data bundle.

## Data bundle

`flakeid_ml_data/` (about 1 GB) is a portable copy of everything labelled. Point the scripts at it
with the environment variable `FLAKEID_ML_DATA`. Without it they look for `flakeid_ml_data` next to
the repository folder. Paths inside `sets/` are relative to the bundle root.

| Path | Contents |
|---|---|
| `sets/graphite100.json` | 486 graphite flakes in the v3 pick-model fit: `key`, `run`, `y` (1 = kept), `img`, `hand_z` (held-out v3 score), `ok` |
| `sets/hbn100.csv` | 1009 hBN flakes with the designer defects v2 features and `decision` (keep / used / discard; y = not discard), `img`, `share_path` |
| `sets/set10.json` | 2352 past 100x targets (both materials): `outcome` (keep / discard / dropped; y = keep), `c150`, `c400`, `hand` (10x features) |
| `images/graphite100/` | Graphite 100x pictures, lossless PNG taken from the designer bundles. Older runs stored them as WebP, so some carry its compression. |
| `images/hbn100/` | hBN 100x pictures (JPEG, as saved by the scan) |
| `images/c150/`, `images/c400/` | 10x crops cut from the viewer tile pyramids |
| `results/` | Feature files (`emb*.npz`), held-out scores (`z_*.npz`, `z10_*.npz`), logs, `examples.json`, `isolation.json`, `nn_summary.html`; setup C: `cache_c.npy` (labelled pictures at 384 × 512), `setupc_pre_<pool>.pt`, `z_<set>_<pool>.npz`, `log_setupc_*.txt` |
| `share100/` | Every 100x shot on the share that has a `_regions.json` (2,566, made by `prep/share100.py`): `X.npy` (384 × 512 pictures), `M.npy` (terrace, debris, step and fine-terrace masks, 96 × 128), `index.json` (key, run, piece, material, file, screen features) |

**hBN `share_path`:** where the picture sits under `scanresults/` on the share. The analysis files
saved next to each shot are there too:
- `_sg` (steps), `_debris`, `_dt`, `_c8`, `_avg*` pictures;
- `_regions.json`.

The 208 pictures of `bn_20261020_run_1` came from a catalog rebuild, and their `share_path` is empty.

**Unlabelled data for pretraining:**
- every 100x shot of every run on the share;
- the 10x tile pyramids (`<run>/<piece>/tiles/image/<level>/<col>_<row>.jpg`) with the live analysis
  tables (`data/<piece>_tiles_live_flakes.csv`).

Five runs (about 1,300 shots and 17,000 10x regions) were also copied locally on the lab PC. The share
has about 14 runs. The share is slow, so copy what you need once.

## Scripts

| Script | What it does |
|---|---|
| `embed.py hbn\|graphite resnet50\|dinov2s` | Frozen network features of the 100x pictures → `results/emb_<set>_<net>.npz` |
| `evaluate.py hbn\|graphite NET...` | Leave-one-run-out AUC: hand model, network, combined → `results/z_<set>.npz` (env `PCAS`, default `32,128`) |
| `boot.py SET KEY...` | Paired bootstrap of the AUC gain over the hand model (keys of `z_<set>.npz`, e.g. `resnet50_32_both`) |
| `embed10.py NET`, `evaluate10.py MAT NET...`, `boot10.py MAT KEY...` | The same for the 10x crops |
| `isolation.py` | The surroundings test |
| `setupc.py cache\|pretrain\|embed\|finetune` | Setup C (see its docstring). Pools: `bundle` (labelled pictures), `share` (all share shots + masks), `sharenm` (same without masks); `finetune imagenet` is the no-pretraining baseline. `boot.py <set>_<pool> ft_<pool>_both` for the gain |
| `prep/share100.py` | Copies every 100x shot and its rule outputs from the share into `share100/` (about 6 min) |
| `setupc.py teacher bundle` | Setup C keep scores of every share shot, one model per left-out run (+ all runs) → `teacher_bundle.npz` (20 min) |
| `tenx.py none\|rule\|teacher [SEEDS]`, `tenx.py report V...` | The 10x targeting network test (about 25 min per variant and 2 seeds) |
| `prep/crops10_share.py` | 10x crops for imaged flakes without an outcome |
| `layers100.py` | Graphite terrace contrast measured in each 100x picture, grouped by the 10x layer call → `results/layers100.json` (reads the share, about 3 min) |
| `shotcheck.py` | Shot-quality flags for every share shot, per-run counts and their share of the screening error (reads the share, about 5 min) |
| `report/examples.py`, `report/gen_page.py` | Example pictures and the summary page |
| `prep/*` | How the sets were made on the lab PC. Kept as a record; they use that PC's paths and scratch tools and would need adapting. |

`paths.py` holds the data locations.

**Check that a new setup reproduces the first test** (seconds):
`python boot.py graphite resnet50_32_both`
should print a gain of +0.075, CI +0.045 .. +0.106.

## Setting up the RTX 5060 PC

- Python 3.11 or 3.12 (64-bit).
- The RTX 50 series (Blackwell) needs PyTorch built for CUDA 12.8 or later (PyTorch 2.7 or later):
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install numpy scipy scikit-learn opencv-python
  ```
- Check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- 8 GB of graphics memory is enough for ResNet-18/50 or a small ViT at 224–512 px with batch sizes of
  16–64. Use mixed precision (`torch.autocast`).
- The first-test scripts set `torch.set_num_threads(8)` and run on the CPU. `setupc.py` runs on the GPU
  (bfloat16 autocast, about 6.6 GB at batch 32).
- On the RTX 5060 PC: the venv is `Software\mlenv` in the user folder (Python 3.13, torch 2.11 cu128). The
  data bundle is next to the repository. Set `FLAKEID_SHARE` to your `scanresults` folder on the share (the scripts' default is `Z:/scanresults`).

## Next steps (agreed with the user, in order)

1. **Pretrain on rule measurements, then fine-tune (setup C).** Done 2026-10-24; see the results above.
   Follow-ups:
   - Graphite: put the fine-tuned network's score into the designer build's screening, alone or
     combined. It needs one model fitted on all runs and a GPU, or about 1 s per picture on a CPU.
   - Larger inputs or a crop on terrace a, and a learning-rate/epoch check, nested over the training
     runs.
   - The share pool didn't beat the labelled pool. More runs (step 3) may matter more than more shots
     of the same runs.
2. **Self-supervised pretraining (setup D)** on the same unlabelled pictures (`share100/`, DINO-style) as a
   comparison.
3. **Learning curve:** train on 2, 4 and 6 runs.
   - Still rising at 6: more runs (each adds about 20–30 kept flakes with no extra labelling) are the
     best investment.
   - Flat: reason tags on discards are.
4. **Reason cards:** pre-tag discards whose rule measurements are clearly out of range and queue the
   uncertain ones for the user to review.
   - hBN: 397 of about 900 discards have at least one measurement worse than 98 % of kept flakes;
     505 look like keeps by the rules.
   - Aim for 200–300 reviewed cards, picked where the model is least sure and spread across runs,
     with a suggested reason that one key accepts.
   - Include a "fine but not needed" tag: those discards currently teach the model that a good
     flake is bad.
   - Labelling time is the user's bottleneck, so keep the queue short.
5. **10x score refit on 100x outcomes.** Done 2026-10-23: `flakeid/target_model.py` and
   `tools/fit_target_model.py`, used for graphite target ranking; see TODO.md. The labels
   (`sets/target_labels_10x.csv`) and the live tables (`live10/`) are in the data bundle, so it can be
   refitted there as runs accumulate.

**Selection bias:** only flakes the current 10x score chose were ever imaged at 100x. Every 10x model
learns only among those. Imaging a few randomly chosen low-ranked flakes per piece would remove that
blind spot.

**Where a network could run:** only in the 100x screening, which runs during the designer build on a
lab PC. Not in the live scan on the scope PC.
