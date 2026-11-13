#!/usr/bin/env python3
"""Label the thickness of 100x-imaged flakes by eye: the data for a later 100x / 10x thickness calibration.

    python tools/label_thickness.py <piece folder> [--port 8765] [--no-browser]

<piece folder>: a piece as copied to the share (its viewer index.html, 100x/, data/), e.g.
"Z:\\<your folder>\\<run>\\<piece>".  A page opens in the browser
(served on this computer only, http://127.0.0.1:<port>): one flake at a time, its 100x picture and its 10x crop, the
10x thickness and the other thicknesses its 10x colour could also be (the thin-film colour model's other good fits over
0-200 nm), each with a swatch of the colour the model predicts at 100x beside the colour measured on the flake's clean
area in the 100x picture.  Click the one that looks right (keys 1-9), "none of these", "unsure", or type a thickness;
arrow keys move between flakes.  The labels go to <piece folder>/thickness_labels.json (one entry per flake: the
choice, the 10x and 100x colours, the candidates shown), or, if that folder cannot be written, to ~/FlakeScans.

Only reads the analysis' files; nothing else is changed.  The 100x swatches use the thin-film model at NA 0.9 without a
100x calibration (that is what these labels are for): compare hues, not exact shades.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAND_RANGE_NM = (0.0, 200.0)        # candidate thicknesses: local minima of the 10x colour fit in this range ...
CAND_MAX = 6                        # ... at most this many, best fit first ...
CAND_SEP_NM = 8.0                   # ... at least this far apart ...
CAND_RES = (2.5, 0.15)              # ... and fitting within max(2.5 x the best residual, best + 0.15)
NA_100X = 0.9


def viewer_data(piece):
    """(D, H100) of the piece's viewer page."""
    html = (Path(piece) / "index.html").read_text(encoding="utf-8")
    D = json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))
    m = re.search(r"const H100=(.*?);\n", html, re.S)
    H = json.loads(m.group(1)) if m and m.group(1).strip() not in ("null", "") else {}
    return D, H


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class Model:
    """The 10x colour fit (the analysis' profile and oxide) and the 100x prediction (the same, at NA 0.9)."""

    def __init__(self, material, oxide_nm, profile_name):
        from flakeid.optics import ContrastModel
        from flakeid.thickness import ThicknessEstimator, camera_from_profile, load_profile
        try:
            prof = load_profile(profile_name)
        except Exception:
            prof = load_profile(None)
        self.material, self.oxide = material, float(oxide_nm)
        self.est = ThicknessEstimator(material, self.oxide, prof, step_nm=0.25, search_range_nm=CAND_RANGE_NM)
        self.m100 = ContrastModel(material, self.oxide, na=NA_100X, camera=camera_from_profile(prof))
        self.profile = Path(prof.get("_path", "")).stem

    def candidates(self, c10, extra=()):
        """[(thickness nm, residual)]: the 10x colour c10's good fits (local minima of the distance to the model curve),
        best first, with the analysis' own values (extra) always in."""
        d = np.linalg.norm(self.est.curve - np.asarray(c10, float)[None, :], axis=1)
        mins = [i for i in range(1, len(d) - 1) if d[i] <= d[i - 1] and d[i] <= d[i + 1]] or [int(np.argmin(d))]
        best = float(min(d[i] for i in mins))
        lim = max(CAND_RES[0] * best, best + CAND_RES[1])
        out = []
        for i in sorted(mins, key=lambda i: d[i]):
            t = float(self.est.d[i])
            if d[i] <= lim and all(abs(t - u) >= CAND_SEP_NM for u, _ in out):
                out.append((round(t, 1), round(float(d[i]), 3)))
            if len(out) >= CAND_MAX:
                break
        for t in extra:
            if t == t and CAND_RANGE_NM[0] <= t <= CAND_RANGE_NM[1] and all(abs(t - u) >= 2.0 for u, _ in out):
                k = int(np.argmin(np.abs(self.est.d - t)))
                out.append((round(float(t), 1), round(float(d[k]), 3)))
        return out

    def swatch100(self, t):
        """The model's 100x colour at thickness t in the pictures' look (8-bit RGB)."""
        from flakeid.look import look
        c = self.m100.contrast([max(float(t), 0.0)])[0]
        return [int(v) for v in look(np.array([[1.0 + c]], np.float32), (1.0, 1.0, 1.0))[0, 0]]


def clean_colour(piece, entry, material):
    """The 100x picture's clean area colour against its substrate: (contrast (3,), look swatch RGB) or (None, None).
    The largest clean area (the 100x score's outline) if recorded, else the terrace; the substrate: the picture's
    commonest colour when it is the run's (hundred._same_substrate), else the run's (DEMO_SUBSTRATE in JPEG units)."""
    import cv2
    from flakeid.hundred import DEMO_SUBSTRATE, WHITE, WHITE_BY_MATERIAL, _mode_rgb, _same_substrate
    from flakeid.look import look
    src = Path(piece) / entry["f"]
    rj = src.with_name(src.stem + "_regions.json")
    if not src.is_file() or not rj.is_file():
        return None, None
    a = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if a is None:
        return None, None
    rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32)
    R = json.loads(rj.read_text())
    polys = (R.get("score100") or {}).get("poly") or R.get("terrace") or []
    m = np.zeros(rgb.shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
    if m.sum() < 50:
        return None, None
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    col = np.median(rgb[m > 0], axis=0) if m.any() else np.median(rgb[m >= 0], axis=0)
    sub = np.asarray(DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"]), float) * 255.0 / WHITE_BY_MATERIAL.get(material, WHITE)
    fm, frac = _mode_rgb(rgb[::4, ::4], step=3)
    if frac > 0.2 and _same_substrate(fm, sub):
        sub = np.asarray(fm, float)
    c = col / np.maximum(sub, 1e-6) - 1.0
    sw = [int(v) for v in look(np.array([[1.0 + c]], np.float32), (1.0, 1.0, 1.0))[0, 0]]
    return [round(float(v), 4) for v in c], sw


def build_items(piece, log=print):
    """One item per flake with a (non-domain) 100x picture: its pictures, 10x thickness, candidates, colours."""
    piece = Path(piece)
    D, H = viewer_data(piece)
    name = piece.name
    data = piece / "data"
    ter = {r["terrace_id"]: r for r in read_csv(next(data.glob("*_tiles_live_terraces.csv")))}
    fl = {r["flake_id"]: r for r in read_csv(next(data.glob("*_tiles_live_flakes.csv")))}
    material = D.get("material") or "hbn"
    oxide = _f(D.get("oxide"), 285.0)
    prof = ""
    fj = next(data.glob("*_tiles_live_flakes.json"), None)
    if fj is not None:
        try:
            prof = Path(str(json.loads(fj.read_text())["params"].get("profile") or "").replace("\\", "/")).stem
        except (OSError, ValueError, KeyError, TypeError):
            prof = ""
    # (the analysis' profile: recorded in the flakes' JSON when it was not the default; else the one pipeline.find_flakes
    # takes for the scope's linear RAW16 data)
    model = Model(material, oxide, prof or ("bx51_xc10_mm_raw16_graphite" if material == "graphite" else "bx51_xc10_mm_raw16"))
    log(f"{name}: {material} on {oxide:g} nm oxide (the analysis' value), colour model {model.profile}")
    crops = next(data.glob("*_targets_crops"), None)
    items = []
    for fid, es in sorted(H.items(), key=lambda kv: int(kv[0])):
        e = next((x for x in es if not x.get("domain")), None)
        f = fl.get(str(fid))
        if e is None or f is None:
            continue
        t = ter.get(f.get("best_terrace_id") or "")
        if t is None:
            continue
        c10 = [_f(t["contrast_r"]), _f(t["contrast_g"]), _f(t["contrast_b"])]
        alts = [_f(x) for x in (t.get("alt_thickness_nm") or "").split(";") if x.strip()]
        t10 = _f(t.get("thickness_nm"))
        cands = model.candidates(c10, [t10] + alts) if all(v == v for v in c10) else []
        c100, sw100 = clean_colour(piece, e, material)
        crop = None
        if crops is not None:
            lab = e.get("lab") or ""
            m = re.search(r"_F(\d+)", lab)
            hits = sorted(crops.glob(f"*_F{int(m.group(1)):03d}*.jpg")) if m else []
            crop = str(hits[0].relative_to(piece)).replace("\\", "/") if hits else None
        look_f = (e.get("look") or {}).get("f") or e["f"]
        items.append(dict(flake=int(fid), terrace=t["terrace_id"], label=e.get("lab", ""), img=look_f, crop=crop,
                          t10=t10, alt=alts, quality=t.get("quality", ""), c10=[round(v, 4) for v in c10], c100=c100,
                          meas100=sw100, s100=e.get("s100"),
                          cands=[dict(t=c, res=r, sw=model.swatch100(c), is10=abs(c - t10) < 2.0) for c, r in cands]))
    return dict(piece=name, material=material, oxide=oxide, profile=model.profile, items=items)


def labels_file(piece):
    """<piece>/thickness_labels.json, or ~/FlakeScans/<piece>_thickness_labels.json when the piece folder cannot be
    written."""
    p = Path(piece) / "thickness_labels.json"
    try:
        with open(p, "a", encoding="utf-8"):
            pass
        return p, None
    except OSError as e:
        alt = Path.home() / "FlakeScans" / f"{Path(piece).name}_thickness_labels.json"
        alt.parent.mkdir(parents=True, exist_ok=True)
        return alt, f"{p} cannot be written ({e}): the labels go to {alt}"


def load_labels(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Thickness labels: __PIECE__</title>
<style>
:root{--bg:#111318;--panel:#1b1e26;--ink:#e8e8ee;--muted:#9a9ab0;--acc:#4ea3ff;--ok:#37b24d;--rule:#2e3240}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 "Segoe UI",system-ui,sans-serif}
header{display:flex;gap:16px;align-items:baseline;padding:10px 16px;border-bottom:1px solid var(--rule)}
h1{font-size:17px;margin:0}#prog{color:var(--muted)}#where{color:var(--muted);margin-left:auto}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:14px;padding:12px 16px}
figure{margin:0}figure img{width:100%;height:auto;border-radius:6px;border:1px solid var(--rule);background:#000}
figcaption{color:var(--muted);font-size:12px;margin-top:4px}
#side{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px}
#crop{max-width:100%;max-height:260px;border-radius:6px;border:1px solid var(--rule)}
.row{display:flex;gap:10px;align-items:center}.sw{width:46px;height:30px;border-radius:5px;border:1px solid #555;flex:none}
.cand{display:flex;gap:10px;align-items:center;width:100%;padding:7px 9px;margin:5px 0;background:#232735;color:var(--ink);
 border:1px solid var(--rule);border-radius:7px;cursor:pointer;font:inherit;text-align:left}
.cand:hover{border-color:var(--acc)}.cand.sel{border-color:var(--ok);background:#1f3326}
.cand kbd{font:11px ui-monospace,Consolas,monospace;border:1px solid #666;border-radius:3px;padding:0 5px;color:var(--muted)}
.cand b{min-width:72px}.cand small{color:var(--muted)}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.btns button,.nav button{font:inherit;padding:6px 12px;border-radius:6px;
 border:1px solid var(--rule);background:#232735;color:var(--ink);cursor:pointer}.btns button.sel{border-color:var(--ok);background:#1f3326}
input{font:inherit;width:90px;padding:5px;border-radius:5px;border:1px solid var(--rule);background:#0d0f14;color:var(--ink)}
.nav{display:flex;gap:8px;margin-top:12px}.muted{color:var(--muted)}#saved{color:var(--ok);min-height:1.4em}
</style></head><body>
<header><h1>Thickness labels · __PIECE__</h1><span id="prog"></span><span id="where"></span></header>
<main><div><figure><img id="img100" alt="100x picture"><figcaption id="cap100"></figcaption></figure></div>
<div id="side"><div id="head"></div><img id="crop" alt="10x crop"><div class="muted" style="font-size:12px">10x picture of the flake</div>
<div class="row" style="margin-top:10px"><div class="sw" id="meas"></div><div>measured on the 100x picture (clean area)<br>
<span class="muted" id="c100"></span></div></div>
<div style="margin-top:10px">Which thickness looks right? <span class="muted">(swatches: the model's colour at 100x, uncalibrated: compare hues)</span></div>
<div id="cands"></div>
<div class="btns"><button id="none">none of these</button><button id="unsure">unsure</button>
<span class="row"><input id="typed" placeholder="nm" inputmode="decimal"><button id="settyped">this thickness</button></span></div>
<div id="saved"></div>
<div class="nav"><button id="prev">&larr; previous</button><button id="next">next &rarr;</button><button id="nextun">next unlabelled</button></div>
<div class="muted" style="margin-top:8px;font-size:12px">keys: 1-9 candidates, N none, U unsure, &larr; / &rarr; previous / next</div>
</div></main>
<script>
const DATA=__DATA__;let LAB=__LABELS__;let k=0;
const $=id=>document.getElementById(id);
const rgb=a=>a?`rgb(${a[0]},${a[1]},${a[2]})`:'transparent';
function done(){return DATA.items.filter(it=>LAB[it.flake]).length}
function show(){const it=DATA.items[k];if(!it)return;const L=LAB[it.flake];
 $('prog').textContent=`picture ${k+1} of ${DATA.items.length} · ${done()} of ${DATA.items.length} labelled`;
 $('where').textContent=`${DATA.material} on ${DATA.oxide} nm SiO2 · model ${DATA.profile}`;
 $('img100').src='file?p='+encodeURIComponent(it.img);$('cap100').textContent=`100x · ${it.label}`+(it.s100?` · 100x score ${Math.round(it.s100[0])} µm²`:'');
 $('crop').src=it.crop?'crop?p='+encodeURIComponent(it.crop):'';$('crop').style.display=it.crop?'':'none';
 $('head').innerHTML=`<b style="font-size:16px">Flake ${it.flake}</b> · 10x thickness <b>${it.t10} nm</b>`+(it.alt.length?` (alt ${it.alt.join(', ')})`:'')+
  (it.quality&&it.quality!=='good'?`<br><span class="muted">${it.quality.replace(/;/g,', ')}</span>`:'');
 $('meas').style.background=rgb(it.meas100);$('c100').textContent=it.c100?`contrast ${it.c100.map(v=>v.toFixed(2)).join(' / ')} (10x ${it.c10.map(v=>v.toFixed(2)).join(' / ')})`:'(no clean area found)';
 $('cands').innerHTML=it.cands.map((c,j)=>`<button class="cand${L&&L.choice==='t'&&Math.abs(L.t-c.t)<0.05?' sel':''}" data-j="${j}">
  <kbd>${j+1}</kbd><b>${c.t} nm</b><span class="sw" style="background:${rgb(c.sw)}"></span><small>${c.is10?'10x analysis · ':''}fit ${c.res}</small></button>`).join('');
 document.querySelectorAll('.cand').forEach(b=>b.onclick=()=>pick('t',it.cands[+b.dataset.j].t));
 $('none').classList.toggle('sel',!!L&&L.choice==='none');$('unsure').classList.toggle('sel',!!L&&L.choice==='unsure');
 $('typed').value=L&&L.choice==='typed'?L.t:'';
 $('saved').textContent=L?`saved: ${L.choice==='t'||L.choice==='typed'?L.t+' nm':L.choice}`:'';}
async function pick(choice,t){const it=DATA.items[k];
 const body={flake:it.flake,terrace:it.terrace,label:it.label,choice,t:t==null?null:+t,t10:it.t10,alt:it.alt,c10:it.c10,c100:it.c100,
  candidates:it.cands.map(c=>c.t)};
 const r=await fetch('label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok){$('saved').textContent='NOT saved: '+await r.text();return}
 LAB=await r.json();show();setTimeout(()=>{if(k<DATA.items.length-1){k++;show()}},250)}
$('none').onclick=()=>pick('none');$('unsure').onclick=()=>pick('unsure');
$('settyped').onclick=()=>{const v=parseFloat($('typed').value);if(v>=0&&v<1000)pick('typed',v);else $('saved').textContent='type a thickness in nm'};
$('typed').addEventListener('keydown',e=>{if(e.key==='Enter')$('settyped').click();e.stopPropagation()});
$('prev').onclick=()=>{if(k>0){k--;show()}};$('next').onclick=()=>{if(k<DATA.items.length-1){k++;show()}};
$('nextun').onclick=()=>{for(let j=1;j<=DATA.items.length;j++){const q=(k+j)%DATA.items.length;if(!LAB[DATA.items[q].flake]){k=q;show();return}}};
document.addEventListener('keydown',e=>{const it=DATA.items[k];if(!it)return;
 if(e.key==='ArrowRight')$('next').click();else if(e.key==='ArrowLeft')$('prev').click();
 else if(/^[1-9]$/.test(e.key)&&it.cands[+e.key-1])pick('t',it.cands[+e.key-1].t);
 else if(e.key==='n'||e.key==='N')pick('none');else if(e.key==='u'||e.key==='U')pick('unsure')});
// (?flake=N opens that flake; else the first one not labelled yet)
const want=+new URLSearchParams(location.search).get('flake'),wi=DATA.items.findIndex(it=>it.flake===want);
const first=DATA.items.findIndex(it=>!LAB[it.flake]);k=wi>=0?wi:(first<0?0:first);show();
</script></body></html>"""


def make_handler(piece, info, lab_path, lock):
    piece = Path(piece).resolve()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            b = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _path(self, q):
            p = (piece / (q.get("p") or [""])[0]).resolve()
            return p if str(p).startswith(str(piece)) and p.is_file() else None   # (files of the piece folder only)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                labs = load_labels(lab_path).get("labels", {})
                page = (PAGE.replace("__PIECE__", info["piece"]).replace("__DATA__", json.dumps(info))
                        .replace("__LABELS__", json.dumps(labs)))
                self._send(200, page, "text/html; charset=utf-8")
            elif u.path == "/file":
                p = self._path(q)
                if p is None:
                    return self._send(404, "not found")
                self._send(200, p.read_bytes(), "image/jpeg")
            elif u.path == "/crop":
                # (the 10x crop, grey-balanced like the viewer's linear layer, shown with the pictures' look)
                p = self._path(q)
                if p is None:
                    return self._send(404, "not found")
                import cv2
                from flakeid.look import look_tile
                a = cv2.imread(str(p), cv2.IMREAD_COLOR)
                ok, enc = cv2.imencode(".jpg", look_tile(a), [cv2.IMWRITE_JPEG_QUALITY, 92]) if a is not None else (False, None)
                self._send(200, enc.tobytes() if ok else p.read_bytes(), "image/jpeg")
            elif u.path == "/labels":
                self._send(200, json.dumps(load_labels(lab_path)), "application/json")
            else:
                self._send(404, "not found")

        def do_POST(self):
            if urlparse(self.path).path != "/label":
                return self._send(404, "not found")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                d = json.loads(self.rfile.read(n).decode("utf-8"))
                assert d.get("choice") in ("t", "typed", "none", "unsure")
                fid = str(int(d["flake"]))
            except Exception as e:
                return self._send(400, f"bad label: {e}")
            with lock:
                L = load_labels(lab_path)
                L.update(piece=info["piece"], material=info["material"], oxide_nm=info["oxide"], profile=info["profile"])
                L.setdefault("labels", {})[fid] = dict(d, time=time.strftime("%Y-%m-%d %H:%M:%S"))
                tmp = Path(str(lab_path) + ".tmp")
                tmp.write_text(json.dumps(L, indent=1), encoding="utf-8")
                tmp.replace(lab_path)
            self._send(200, json.dumps(L["labels"]), "application/json")
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("piece", help="the piece folder (its viewer index.html, 100x/, data/)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser (the address is printed)")
    a = ap.parse_args(argv)
    piece = Path(a.piece)
    if not (piece / "index.html").is_file() or not (piece / "data").is_dir():
        ap.error(f"{piece}: not a piece folder (index.html and data/ expected)")
    info = build_items(piece)
    if not info["items"]:
        print(f"{piece.name}: no flake with a 100x picture")
        return 1
    lab_path, note = labels_file(piece)
    if note:
        print(note)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(piece, info, lab_path, threading.Lock()))
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"{len(info['items'])} flakes with a 100x picture; labels to {lab_path}\nopen {url} (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
