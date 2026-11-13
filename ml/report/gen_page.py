"""nn_summary.html: the network test summary (examples.json thumbnails embedded)."""
import json, html
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent
E = json.load(open(OUT / "examples.json"))

# (task, n, kept, hand, network alone, combined, verdict, verdict class, note)
ROWS = [
    ("100x screening, graphite", 486, 166, 0.793, 0.804, 0.856, "Helps", "good",
     "Combined gain +0.075 (95 % range +0.045 to +0.106). 171 instead of 132 of 320 discards screened out while 95 % of keeps pass."),
    ("100x screening, hBN", 1009, 124, 0.824, 0.771, 0.846, "Within noise", "neutral",
     "Combined gain +0.009 (range −0.014 to +0.034): within noise. Network alone is clearly worse."),
    ("10x targeting, hBN", 1646, 123, 0.727, 0.682, 0.737, "No gain", "neutral",
     "Combined gain +0.006 (range −0.031 to +0.041). Wider 400 µm crops did worse than 150 µm."),
    ("10x targeting, graphite", 706, 156, 0.703, 0.649, 0.718, "No gain", "neutral",
     "Combined gain +0.022 (range −0.015 to +0.057). With 400 µm crops: −0.043, significantly worse."),
]


def bar(v, cls):
    lo, hi = 0.5, 0.9
    w = max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100
    return f'<div class="bar {cls}"><div class="tr"><span style="width:{w:.1f}%"></span></div><em>{v:.3f}</em></div>'


def meter(label, v):
    return (f'<div class="m"><span class="ml">{label}</span><span class="mt"><i style="left:{v}%"></i></span>'
            f'<span class="mv">{v}</span></div>')


def card(r, caption, wide=False, ten=False):
    out = {"keep": "kept", "discard": "discarded", "dropped": "dropped (catalog)"}
    verdict = out.get(r.get("outcome"), "kept" if r["y"] else "discarded")
    cls = "k" if r["y"] else "d"
    meters = (meter("10x numbers", r["hand"]) + meter("network", r["net"])) if ten else (
        meter("hand model", r["hand"]) + meter("network", r["net"]) + meter("combined", r["both"]))
    name = r["key"].split(":")[1]
    return (f'<figure class="card{" wide" if wide else ""}"><img src="{r["img"]}" alt="{html.escape(name)}" loading="lazy">'
            f'<figcaption><div class="row"><b>{html.escape(name)}</b><span class="tag {cls}">{verdict}</span></div>'
            f'{meters}<p>{caption}</p></figcaption></figure>')


g = {k: [r for r in E["graphite100"] if r["kind"] == k] for k in ("rescued", "caught", "missed", "fooled")}
h = {k: [r for r in E["hbn100"] if r["kind"] == k] for k in ("net_misses", "net_sees")}
t = {(m, k): [r for r in E[f"{m}10"] if r["kind"] == k] for m in ("hbn", "graphite")
     for k in ("net_high_kept", "net_high_wasted", "net_low_kept")}

CAP = {
    "rescued": "Kept. The hand model ranked it low; the network ranks it near the top. Faint, thin few-layer pieces like this are what the combination recovers.",
    "caught": "Discarded. The hand model ranked it high; the network ranks it low. Thicker dark flakes with debris or attached pieces nearby.",
    "missed": "Kept, but both rank it low. Large, dark (thick) graphite: the decisions keep some of these and the models have too few examples.",
    "fooled": "Discarded, but both rank it at the top. A nearly featureless frame: a very faint flake or a shot that missed it. Bad input fools both.",
    "net_misses": "Kept and ranked high by the hand model, ranked low by the network. A clean terrace on a dirty substrate: the network reacts to the speckles around it.",
    "net_sees": "Discarded, ranked high by the hand model, low by the network. Thick attached pieces, colour steps or folds at the edges that the terrace-based hand features do not weigh.",
}
CAP10 = {
    "net_high_kept": "Top network score, kept.",
    "net_high_wasted": "Top network score, wasted. At 10x it looks like the kept ones.",
    "net_low_kept": "Bottom network score, kept. Crowded scene or a large thick crystal beside it: the network learned 'busy = bad', which is wrong here.",
}

grid = lambda items: '<div class="grid">' + "".join(items) + "</div>"
rows_html = "".join(
    f'<tr><th scope="row">{a}<small>{n:,} flakes, {k} kept</small></th><td>{bar(hd, "h")}</td><td>{bar(nn, "n")}</td>'
    f'<td>{bar(bo, "b")}</td><td><span class="chip {vc}">{v}</span><small>{note}</small></td></tr>'
    for a, n, k, hd, nn, bo, v, vc, note in ROWS)

page = f"""<title>Flake Network Trial</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column (~68ch) for the argument; image grids break out to the full width for the evidence */
:root {{
  --bg: #f6f4f7; --surface: #ffffff; --ink: #221d26; --muted: #625a68; --line: #ddd6e1;
  --oxide: #6b4a78;          /* 285 nm SiO2 purple: the accent */
  --hbn: #2f7f95; --good: #2e7d4f; --warn: #a3501f; --neutral: #6d6577;
  --hand: #9b8aa6; --net: #2f7f95; --both: #6b4a78;
  --serif: "Literata", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", Consolas, "Courier New", monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #17141a; --surface: #211d25; --ink: #ece6ef; --muted: #a79fae; --line: #3a3340;
  --oxide: #c39bd3; --hbn: #6cc0d6; --good: #6cc48e; --warn: #e59b6b; --neutral: #a79fae;
  --hand: #8d7d98; --net: #6cc0d6; --both: #c39bd3; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #17141a; --surface: #211d25; --ink: #ece6ef; --muted: #a79fae; --line: #3a3340;
  --oxide: #c39bd3; --hbn: #6cc0d6; --good: #6cc48e; --warn: #e59b6b; --neutral: #a79fae;
  --hand: #8d7d98; --net: #6cc0d6; --both: #c39bd3; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.6 var(--sans); }}
main {{ max-width: 1120px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: flex; flex-direction: column; gap: 22px }}
.col {{ max-width: 68ch }}
h1, h2, h3 {{ font-family: var(--serif); font-weight: 650; text-wrap: balance; line-height: 1.2; margin: 0 }}
h1 {{ font-size: clamp(1.9rem, 4vw, 2.6rem) }}
h2 {{ font-size: 1.55rem; margin-top: 26px; padding-top: 22px; border-top: 1px solid var(--line) }}
h3 {{ font-size: 1.15rem; margin-top: 8px }}
p {{ margin: 0 }} .col p + p {{ margin-top: 10px }}
.lede {{ font-size: 1.12rem; color: var(--muted) }}
.eyebrow {{ font: 500 .78rem var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--oxide) }}
.verdicts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px }}
.verdicts div {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px }}
.verdicts b {{ display: block; font-family: var(--serif); font-size: 1.05rem; margin-bottom: 4px }}
.flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; align-items: stretch }}
.flow div {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: var(--surface); font-size: .92rem }}
.flow div b {{ display: block; font: 600 .74rem var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted) }}
.flow .net {{ border-color: var(--net) }}
.flowlabel {{ font: 500 .85rem var(--mono); color: var(--muted); margin-top: 6px }}
.tablewrap {{ overflow-x: auto }}
table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-variant-numeric: tabular-nums }}
th, td {{ text-align: left; vertical-align: top; padding: 12px 10px 12px 0; border-bottom: 1px solid var(--line) }}
thead th {{ font: 500 .74rem var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted) }}
th[scope=row] {{ font-weight: 600; width: 22% }}
th small, td small {{ display: block; font-weight: 400; color: var(--muted); font-size: .84rem; margin-top: 4px }}
.bar {{ display: flex; align-items: center; gap: 8px; min-width: 130px }}
.bar .tr {{ position: relative; flex: 1; height: 14px; background: color-mix(in srgb, var(--line) 60%, transparent); border-radius: 3px }}
.bar span {{ position: absolute; inset: 0 auto 0 0; border-radius: 3px }}
.bar.h span {{ background: var(--hand) }} .bar.n span {{ background: var(--net) }} .bar.b span {{ background: var(--both) }}
.bar em {{ font: 500 .82rem var(--mono); font-style: normal; color: var(--ink); width: 3.2em }}
.scale {{ font: .78rem var(--mono); color: var(--muted) }}
.chip {{ display: inline-block; font: 600 .76rem var(--mono); letter-spacing: .04em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; border: 1px solid }}
.chip.good {{ color: var(--good); border-color: var(--good) }} .chip.neutral {{ color: var(--neutral); border-color: var(--neutral) }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px }}
.grid.ten {{ grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) }}
.card {{ margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column }}
.card img {{ width: 100%; display: block; aspect-ratio: 4 / 3; object-fit: cover; background: var(--line) }}
.grid.ten .card img {{ aspect-ratio: 1 }}
figcaption {{ padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 6px; font-size: .88rem }}
figcaption .row {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; flex-wrap: wrap }}
figcaption b {{ font: 500 .8rem var(--mono); overflow-wrap: anywhere }}
figcaption p {{ color: var(--muted) }}
.tag {{ font: 600 .72rem var(--mono); text-transform: uppercase; letter-spacing: .05em }}
.tag.k {{ color: var(--good) }} .tag.d {{ color: var(--warn) }}
.m {{ display: grid; grid-template-columns: 86px 1fr 26px; gap: 8px; align-items: center; font: .76rem var(--mono); color: var(--muted) }}
.mt {{ position: relative; height: 6px; border-radius: 3px; background: linear-gradient(90deg, color-mix(in srgb, var(--warn) 45%, transparent), color-mix(in srgb, var(--good) 45%, transparent)) }}
.mt i {{ position: absolute; top: -4px; width: 3px; height: 14px; margin-left: -1px; background: var(--ink); border-radius: 1px }}
.mv {{ text-align: right; color: var(--ink) }}
.legend {{ font-size: .88rem; color: var(--muted) }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px }}
.two > div {{ min-width: 0 }}
ul {{ margin: 6px 0 0; padding-left: 1.2em }} li + li {{ margin-top: 6px }}
.steps li::marker {{ font-family: var(--mono); color: var(--oxide) }}
code {{ font: .88em var(--mono) }}
@media (max-width: 560px) {{ h2 {{ font-size: 1.3rem }} }}
</style>
<main>
<header class="col" style="display:flex;flex-direction:column;gap:10px">
  <span class="eyebrow">Flake Finder · 2026-10-23 · first test</span>
  <h1>Can a neural network pick flakes better than our hand-built measures?</h1>
  <p class="lede">A test on the decisions already made in the designer: 1,495 flakes judged at 100x and 2,352 past 100x targets with a known outcome, run on the lab PC's processor.</p>
</header>

<section class="verdicts" aria-label="Findings">
  <div><b>Graphite at 100x: yes, combined</b>The network and the current pick model together rank flakes clearly better (AUC 0.793 → 0.856). The gain is well outside the noise.</div>
  <div><b>hBN at 100x: not yet</b>The hand-built defect measures already capture what matters. The network alone is worse, and together the gain is within noise.</div>
  <div><b>10x targeting: no</b>On 10x crops the network adds nothing over the 10x numbers. A wider view with the surroundings did worse, not better.</div>
  <div><b>Isolation: no clear effect</b>How much material lies around a target hardly predicts a keep (hBN 0.59 at best), and for graphite kept flakes have slightly more around them.</div>
</section>

<h2>What the network does</h2>
<div class="col"><p>The current models measure a fixed list of things we chose: terrace size, debris area, edge length, roughness, wrinkle density, defects. A simple model then learns how much each one counts in your keep/discard decisions.</p>
<p>A neural network replaces the fixed list. It is a deep stack of image filters that was trained on millions of everyday photos. Early filters respond to edges, spots and colour changes; later ones combine them into textures and shapes. Shown a flake picture, it returns about 1,500 to 4,000 numbers describing everything in the picture. The same kind of simple model then learns which of those descriptions go with your keeps. Nobody told it what a wrinkle or residue is; it only sees whatever its filters respond to.</p></div>
<div><div class="flowlabel">Current model</div><div class="flow">
  <div><b>Picture</b>100x frame of the target</div><div><b>Our measures</b>~20 numbers: terrace a, debris, steps, roughness, wrinkles</div>
  <div><b>Simple model</b>weights learned from your decisions</div><div><b>Score</b>keep / review / screened out</div></div>
<div class="flowlabel">Network test</div><div class="flow">
  <div><b>Picture</b>the same frame, whole</div><div class="net"><b>Pretrained network</b>~1,500 to 4,000 numbers, nothing chosen by us</div>
  <div><b>Simple model</b>weights learned from your decisions</div><div><b>Score</b>combined with the current one</div></div></div>
<div class="col"><p>This test used the networks as they come (ResNet-50 and DINOv2-small, both trained on general photos), with no training on our pictures. That is the cheap version: it runs on the lab PC in 6 to 12 minutes per material. Training the network itself on our pictures (fine-tuning) is the stronger version and needs a PC with a graphics card.</p></div>

<h2>How it was scored</h2>
<div class="col"><p>Each model is trained on all runs but one and scored on the run it never saw, in turn for every run, exactly as the current pick models are checked. Nothing about the held-out run, not even the model settings, is chosen with it.</p>
<p><b>AUC</b> is the chance that a randomly picked kept flake scores above a randomly picked discarded one: 0.5 is a coin flip, 1.0 is perfect. The <b>combined</b> score is the average of the network's and the current model's held-out scores. Gains were checked by resampling flakes within runs (2,000 times); the range given is where 95 % of those resamples fall.</p></div>

<div class="tablewrap"><table>
<thead><tr><th>Task</th><th>Current model</th><th>Network alone</th><th>Combined</th><th>Verdict</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<p class="scale">Bars run from AUC 0.5 (coin flip) to 0.9. For the network, the best of the few versions tried is shown, so its numbers are if anything flattering. 10x outcome: kept vs discarded or dropped after the 100x shot.</p></div>

<h2>Graphite at 100x: where the network changes the call</h2>
<div class="col"><p class="legend">Each marker shows where the flake ranks among all graphite flakes under each model: 0 = most likely discard, 100 = most likely keep. Ranks are from the held-out runs.</p></div>
<h3>Kept flakes the combination rescues</h3>
{grid(card(r, CAP["rescued"]) for r in g["rescued"])}
<h3>Discards the combination catches</h3>
{grid(card(r, CAP["caught"]) for r in g["caught"])}
<h3>Still wrong with both</h3>
{grid([card(r, CAP["missed"]) for r in g["missed"]] + [card(r, CAP["fooled"]) for r in g["fooled"]])}
<div class="col"><p>The pattern: the network is good at recognising thin, faint graphite strips, which you keep and which the hand measures score modestly because they are faint and narrow. It also weighs clutter around thicker flakes. It cannot fix an empty or out-of-place frame.</p></div>

<h2>hBN at 100x: why it does not help yet</h2>
<h3>Kept flakes the network ranks low</h3>
{grid(card(r, CAP["net_misses"]) for r in h["net_misses"])}
<h3>Discards the network ranks low and the hand model does not</h3>
{grid(card(r, CAP["net_sees"]) for r in h["net_sees"])}
<div class="col"><p>For hBN the network sees the whole frame and cannot tell which flake is meant, or which part of it is terrace a. It does notice things the terrace-based measures skip (thick pieces attached at the edge, colour steps), but it is equally impressed by substrate dirt next to a clean flake. The hand-built defect measures already look at exactly the right region, so they win on their own and the network's extra signal and extra noise cancel out.</p></div>

<h2>10x targeting: the network on 10x crops</h2>
<div class="col"><p>For every past 100x target, a 157 µm crop (full 10x resolution) and a 419 µm crop (half resolution, with the neighbourhood) were cut from the viewer mosaics and compared with whether the flake was kept after its 100x shot. The bar to beat is the 10x numbers the analysis already writes (26 for hBN, 37 for graphite: score, terrace size, shape, haze ring, crowding and so on). These cards show the 419 µm crops.</p></div>
<h3>hBN</h3>
<div class="grid ten">{"".join(card(r, CAP10[r["kind"]], ten=True) for k in ("net_high_kept", "net_high_wasted", "net_low_kept") for r in t[("hbn", k)])}</div>
<h3>Graphite</h3>
<div class="grid ten">{"".join(card(r, CAP10[r["kind"]], ten=True) for k in ("net_high_kept", "net_high_wasted", "net_low_kept") for r in t[("graphite", k)])}</div>
<div class="col"><p>At 10x a kept and a wasted target look alike: a clean, compact flake of the right colour. What decides the outcome (cracks, single-layer steps, thin residue, bubbles) is below what 10x resolves. The network's main extra cue was how busy the scene is, and that turned out to point the wrong way about as often as the right way.</p>
<p><b>Isolation, measured directly.</b> The share of non-substrate pixels in two rings around each target (from the flake's edge out to 60 µm beyond it, and from there to 200 µm) was compared with the outcome. hBN: kept targets have the same amount of material close by (AUC 0.49) and slightly less farther out (0.59 within runs). Graphite: kept targets have slightly <i>more</i> material around them (0.45 to 0.50), probably because thin graphene sits beside thicker pieces from the same peel. This measure counts flakes and visible residue alike; faint tape residue below the threshold is not counted, so a residue-specific measure could still tell a different story.</p></div>

<h2>Advantages and shortcomings</h2>
<div class="two"><div><h3>Advantages</h3><ul>
<li>Sees things nobody wrote a measure for: thin faint strips, clutter, attached pieces, colour steps.</li>
<li>Cheap to try: no training of the network, minutes on the lab PC, the same scoring as the current models.</li>
<li>Combines well with the current models: the gain for graphite comes from the two disagreeing in useful ways.</li>
<li>Improves automatically as decisions accumulate, with no new code for each new kind of defect.</li>
</ul></div><div><h3>Shortcomings</h3><ul>
<li>Sees the whole frame and does not know which flake or terrace is meant (the main weakness for hBN).</li>
<li>Learns whatever correlates with your decisions, including things that should not matter: substrate dirt, how busy the scene is, picture compression.</li>
<li>Hard to explain a single score: there is no "this flake has 40 µm of cracks" behind it.</li>
<li>Bad input fools it as much as the hand model (empty or missed frames).</li>
<li>Few keeps to learn from (124 hBN, 166 graphite), so differences of a few AUC points are within noise.</li>
</ul></div></div>

<h2>What is needed to make it better</h2>
<ol class="col steps">
<li><b>Show it the flake, not the frame.</b> Mask or crop each picture to terrace a and its edge using the outlines we already have. Expected to help hBN most. Runs on this PC.</li>
<li><b>Fine-tune on our pictures.</b> Train the network itself on the decisions, with flips and rotations as extra examples. Needs a PC with an NVIDIA graphics card (8 GB or more); minutes per training run instead of hours.</li>
<li><b>More decisions, and why.</b> Every keep/discard adds a training example. A reason tag on discards (residue, cracks or steps, too thick, too small, folds, bubbles) would let a model learn each defect separately and let it say why a flake was screened out.</li>
<li><b>Consistent pictures.</b> Older runs survive only as compressed WebP copies; keeping the original 100x frames of decided flakes makes training and comparisons cleaner.</li>
<li><b>For 10x targeting, information rather than modelling.</b> The 10x picture does not contain the defects that decide a keep. Gains there have to come from better 10x data (focus, resolution) or from cheap 100x checks, not from a smarter model.</li>
</ol>

<h2>Where this leaves things</h2>
<div class="col"><p>Graphite screening could use the combined score now: the designer build would run the network once per 100x picture (about a second each on a normal PC; nothing changes at the microscope). For hBN and for 10x targeting, keep the current models. The next experiment is step 1 (network on terrace a's region), which would show whether hBN can gain too, before investing in a GPU machine.</p></div>
</main>
"""
(OUT / "nn_summary.html").write_text(page, encoding="utf-8")
print(len(page) // 1024, "KB")
