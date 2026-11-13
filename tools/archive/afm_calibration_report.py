"""Render tools/calibrate_afm.py's report.json (plus an optional k-fold run) as a single HTML page.

    python3 tools/afm_calibration_report.py testing_afm/calibration/report.json \
        --cv testing_afm/calibration_cv/report.json -o testing_afm/calibration/report.html
"""
import argparse
import json
from pathlib import Path

PAGE = r"""<title>AFM-Refined hBN Calibration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f4f6f8; --panel:#ffffff; --ink:#18202c; --muted:#5b6676; --rule:#d9dee5; --faint:#eceff3;
  --before:#b9741c; --after:#0d7a86; --cv:#6b4bc2; --bad:#b4343a;
  --r:#c0392b; --g:#2e8b3e; --b:#2f5fb3;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  color-scheme:dark; --bg:#12161c; --panel:#1a2029; --ink:#e4e8ee; --muted:#98a3b3; --rule:#2c3440; --faint:#222934;
  --before:#e3a24a; --after:#3fc0cc; --cv:#a58bf0; --bad:#f07a7f; --r:#ef7a6c; --g:#6fcf7e; --b:#7aa3f0;}}
:root[data-theme="dark"]{
  color-scheme:dark; --bg:#12161c; --panel:#1a2029; --ink:#e4e8ee; --muted:#98a3b3; --rule:#2c3440; --faint:#222934;
  --before:#e3a24a; --after:#3fc0cc; --cv:#a58bf0; --bad:#f07a7f; --r:#ef7a6c; --g:#6fcf7e; --b:#7aa3f0;}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;padding-inline:16px;padding-block:28px 56px}
main{max-width:980px;margin:0 auto;display:grid;gap:34px}
h1,h2{font-family:Literata,Georgia,serif;font-weight:650;text-wrap:balance;margin:0;line-height:1.2}
h1{font-size:clamp(26px,4vw,34px)} h2{font-size:21px}
p{margin:0;max-width:68ch} .muted{color:var(--muted)}
.eyebrow{font:500 12px "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{display:grid;gap:10px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;display:grid;gap:2px}
.stat b{font:500 28px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.stat span{font-size:13px;color:var(--muted)}
section{display:grid;gap:12px}
.fig{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px;overflow-x:auto}
.fig svg{display:block;width:100%;height:auto;min-width:520px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}
.panels .fig svg{min-width:0}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:18px;height:0;border-top:2.5px solid;vertical-align:middle;margin-right:6px}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:640px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--faint);white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
td.l,th.l{text-align:left} td.n{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tr.off td{color:var(--muted)} .big{color:var(--bad);font-weight:500}
ul{margin:0;padding-left:20px;max-width:70ch;display:grid;gap:6px}
code{font:13px "IBM Plex Mono",monospace;background:var(--faint);padding:1px 5px;border-radius:3px}
pre{font:13px/1.5 "IBM Plex Mono",monospace;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;overflow-x:auto;margin:0}
svg text{fill:var(--muted);font:11px "IBM Plex Mono",monospace}
</style>
<main>
<header>
  <div class="eyebrow">FlakeIdentifier · BX51 + XC10 RAW16 · bigbntest_2</div>
  <h1>AFM-Refined hBN Calibration</h1>
  <p class="muted" id="lede"></p>
</header>
<div class="stats" id="stats"></div>

<section>
  <h2>Model error at each AFM site</h2>
  <p class="muted">Optical thickness minus AFM thickness. The shaded band is ±3 % of the AFM value. Cross-validated points come from fits that did not see that site.</p>
  <div class="legend"><span><i style="border-color:var(--before)"></i>starting profile</span><span><i style="border-color:var(--after)"></i>refined profile</span><span id="cvleg"><i style="border-color:var(--cv);border-top-style:dotted"></i>refined, site held out</span></div>
  <div class="fig"><svg id="resid" viewBox="0 0 900 340"></svg></div>
</section>

<section>
  <h2>Contrast curves against the AFM points</h2>
  <p class="muted">Model contrast versus thickness for each camera channel (dashed: starting profile at its oxide, solid: refined profile at its oxide). Dots are the measured 10x contrast of each AFM terrace, placed at the AFM thickness. Step-height sites are left out because their AFM value is a difference.</p>
  <div class="panels"><div class="fig"><svg id="cR" viewBox="0 0 320 230"></svg></div><div class="fig"><svg id="cG" viewBox="0 0 320 230"></svg></div><div class="fig"><svg id="cB" viewBox="0 0 320 230"></svg></div></div>
</section>

<section>
  <h2>Sites</h2>
  <div class="tw"><table id="sites"></table></div>
</section>

<section>
  <h2>What changed in the model</h2>
  <div class="tw"><table id="params"></table></div>
  <p class="muted" id="oxnote"></p>
</section>

<section>
  <h2>Using it</h2>
  <p>The refined profile is saved as <code>flakeid/profiles/bx51_xc10_mm_raw16_afm.json</code>. Pass it explicitly, or make it the RAW16 default in <code>flakeid/pipeline.py</code>.</p>
  <pre>python -m flakeid.live "Z:\<you>\scans" hbn 285 --profile bx51_xc10_mm_raw16_afm ...</pre>
  <ul>
    <li>The analysis still fits the oxide thickness on every chip. The refined model finds about 298 nm on bigbntest_2, against 303 nm with the old one, so expect the reported oxide to drop by roughly 5 nm.</li>
    <li>Thickness readings above 40 nm drop by 2–4 %. Thin flakes (below 20 nm) barely change.</li>
    <li>The AFM's own Z calibration (typically 1–2 %) limits the absolute accuracy. A calibration grating scan would separate that from the optical model.</li>
    <li>Graphite uses the same camera model but has no AFM data yet, so its thickness scale is not validated by this fit.</li>
  </ul>
</section>
</main>
<script>
const R = __DATA__;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
function el(p, tag, a, txt){const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;p.appendChild(e);return e;}
const f1 = v => v==null||isNaN(v) ? "–" : v.toFixed(1), f2 = v => v.toFixed(2);
const S = R.summary, cv = S.cross_validated;

$("lede").textContent = `${S.before.n} AFM-measured hBN terraces were used to refit the camera model (channel centres, widths, IR cut, flare) and a small offset on the hBN refractive index. The fit also keeps the model consistent with all ${R.n_samples} flat terraces on the chip. RMS error went from ${f2(S.before.rms_nm)} to ${f2(S.after.rms_nm)} nm` + (cv ? `, and ${f2(cv.rms_nm)} nm when each site is predicted by a fit that left it out.` : ".");
const stat = (v, lab, col) => `<div class="stat"><b style="color:${col}">${f2(v.rms_nm)} nm</b><span>${lab} · ${(100*v.rms_rel).toFixed(1)} % rms · bias ${v.mean_nm>=0?"+":""}${f2(v.mean_nm)} nm · worst ${f2(v.max_abs_nm)} nm</span></div>`;
$("stats").innerHTML = stat(S.before,"starting profile","var(--before)") + stat(S.after,"refined profile","var(--after)") + (cv ? stat(cv,"refined, held out (5-fold)","var(--cv)") : "");
if(!cv) $("cvleg").hidden = true;

// residual plot
(function(){
  const svg=$("resid"), W=900, H=340, L=56, Rm=16, T=14, B=40;
  const used=R.points.filter(p=>p.use);
  const xmax=160, lim=Math.max(5, Math.ceil(Math.max(...used.flatMap(p=>[p.before,p.after,p.loo??p.after].map(v=>Math.abs(v-p.h)))))+1);
  const X=v=>L+(W-L-Rm)*Math.sqrt(v/xmax), Y=v=>T+(H-T-B)*(lim-v)/(2*lim);
  // ±3 % band
  let d=`M${X(0)},${Y(0)}`; for(let h=0;h<=xmax;h+=2) d+=` L${X(h)},${Y(.03*h)}`; for(let h=xmax;h>=0;h-=2) d+=` L${X(h)},${Y(-.03*h)}`;
  el(svg,"path",{d,fill:"var(--faint)",stroke:"none"});
  for(let v=-lim+ (lim%2); v<=lim; v+=2){el(svg,"line",{x1:L,x2:W-Rm,y1:Y(v),y2:Y(v),stroke:v===0?"var(--muted)":"var(--rule)","stroke-width":v===0?1:.6});el(svg,"text",{x:L-8,y:Y(v)+4,"text-anchor":"end"},(v>0?"+":"")+v);}
  for(const v of [0,5,10,20,40,60,100,150]){el(svg,"line",{x1:X(v),x2:X(v),y1:H-B,y2:H-B+5,stroke:"var(--muted)"});el(svg,"text",{x:X(v),y:H-B+18,"text-anchor":"middle"},v);}
  el(svg,"text",{x:(W+L)/2,y:H-4,"text-anchor":"middle"},"AFM thickness (nm, square-root scale)");
  el(svg,"text",{x:14,y:(H-B)/2,transform:`rotate(-90 14 ${(H-B)/2})`,"text-anchor":"middle"},"optical − AFM (nm)");
  for(const p of used){
    const x=X(p.h);
    const ys=[p.before,p.after].concat(p.loo!=null?[p.loo]:[]).map(v=>Y(v-p.h));
    el(svg,"line",{x1:x,x2:x,y1:Math.min(...ys),y2:Math.max(...ys),stroke:"var(--rule)","stroke-width":1.5});
    el(svg,"circle",{cx:x,cy:Y(p.before-p.h),r:4.5,fill:"var(--before)"});
    if(p.loo!=null) el(svg,"circle",{cx:x,cy:Y(p.loo-p.h),r:5.5,fill:"none",stroke:"var(--cv)","stroke-width":1.8});
    el(svg,"circle",{cx:x,cy:Y(p.after-p.h),r:4.5,fill:"var(--after)"});
    el(svg,"text",{x:x+7,y:Y(p.after-p.h)-6},p.site);
  }
})();

// contrast panels
["R","G","B"].forEach((ch,i)=>{
  const svg=$("c"+ch), W=320,H=230,L=40,Rm=10,T=22,B=30, col=`var(--${ch.toLowerCase()})`;
  const d=R.curves.d, cb=R.curves.before.map(c=>c[i]), ca=R.curves.after.map(c=>c[i]);
  const pts=R.points.filter(p=>p.use && p.kind!=="difference");
  const all=cb.concat(ca,pts.map(p=>p.c[i])); const lo=Math.floor(Math.min(...all)*2)/2, hi=Math.ceil(Math.max(...all)*2)/2;
  const X=v=>L+(W-L-Rm)*v/200, Y=v=>T+(H-T-B)*(hi-v)/(hi-lo);
  el(svg,"text",{x:L,y:13,style:`fill:${col};font-weight:500`},`${ch} channel contrast`);
  for(let v=lo; v<=hi+1e-9; v+=.5){el(svg,"line",{x1:L,x2:W-Rm,y1:Y(v),y2:Y(v),stroke:Math.abs(v)<1e-9?"var(--muted)":"var(--faint)","stroke-width":.7}); if(Math.abs(v*2%2)<1e-9) el(svg,"text",{x:L-6,y:Y(v)+4,"text-anchor":"end"},v.toFixed(0));}
  for(const v of [0,50,100,150,200]) el(svg,"text",{x:X(v),y:H-B+16,"text-anchor":"middle"},v);
  el(svg,"text",{x:(W+L)/2,y:H-2,"text-anchor":"middle"},"thickness (nm)");
  const path=a=>a.map((v,k)=>`${k?"L":"M"}${X(d[k]).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  el(svg,"path",{d:path(cb),fill:"none",stroke:col,"stroke-width":1.4,"stroke-dasharray":"4 3",opacity:.75});
  el(svg,"path",{d:path(ca),fill:"none",stroke:col,"stroke-width":2});
  for(const p of pts) el(svg,"circle",{cx:X(p.h),cy:Y(p.c[i]),r:3.6,fill:"var(--ink)",stroke:"var(--panel)","stroke-width":1});
});

// site table
(function(){
  const cvOn=R.points.some(p=>p.loo!=null);
  let h=`<tr><th>Site</th><th class="l">Flake / terrace</th><th class="l">Measurement</th><th>AFM (nm)</th><th>Start</th><th>Refined</th>${cvOn?"<th>Held out</th>":""}<th class="l">Notes</th></tr>`;
  const cell=(v,p)=>{if(v==null)return `<td class="n">–</td>`;const e=v-p.h;return `<td class="n${p.use&&Math.abs(e)>Math.max(2,.03*p.h)?" big":""}">${f1(v)} <span class="muted">(${e>=0?"+":""}${f1(e)})</span></td>`;};
  for(const p of R.points){
    const kind=p.kind.startsWith("absolute")?"step to substrate":p.kind==="difference"?"step height":"terrace on substrate";
    h+=`<tr class="${p.use?"":"off"}"><td class="n">${p.site}</td><td class="l">F${p.flake} · T${p.terrace}</td><td class="l">${kind}</td><td class="n">${f2(p.h)} ± ${f2(p.err)}</td>${cell(p.before,p)}${cell(p.after,p)}${cvOn?cell(p.loo,p):""}<td class="l">${(p.use?"":"not used: ")+(R.why[p.site]||p.why||"")}</td></tr>`;
  }
  for(const s of R.nostep) h+=`<tr class="off"><td class="n">${s}</td><td class="l" colspan="${cvOn?7:6}">AFM scan shows substrate only (no step), not used</td></tr>`;
  $("sites").innerHTML=h;
})();

// params
(function(){
  const a=R.params_before,b=R.params_after;
  const rows=[["c_r","Red channel centre","nm",1],["c_g","Green channel centre","nm",1],["c_b","Blue channel centre","nm",1],
    ["w_r","Red channel width (FWHM)","nm",1],["w_g","Green channel width (FWHM)","nm",1],["w_b","Blue channel width (FWHM)","nm",1],
    ["ir_cut","IR cut edge","nm",1],["f_r","Red flare","%",100],["f_g","Green flare","%",100],["f_b","Blue flare","%",100],["dn","hBN index offset","",1]];
  let h=`<tr><th class="l">Parameter</th><th>Start</th><th>Refined</th><th>Change</th></tr>`;
  for(const [k,lab,u,m] of rows){const x=(a[k]??0)*m,y=b[k]*m,dg=k==="dn"?3:k[0]==="f"?2:1;
    h+=`<tr><td class="l">${lab}${u?` (${u})`:""}</td><td class="n">${x.toFixed(dg)}</td><td class="n">${y.toFixed(dg)}</td><td class="n">${y-x>=0?"+":""}${(y-x).toFixed(dg)}</td></tr>`;}
  h+=`<tr><td class="l">Oxide found on bigbntest_2 (nm)</td><td class="n">${R.oxide_before.toFixed(1)}</td><td class="n">${R.oxide_after.toFixed(1)}</td><td class="n">${(R.oxide_after-R.oxide_before).toFixed(1)}</td></tr>`;
  $("params").innerHTML=h;
  $("oxnote").textContent=`The hBN index offset is applied as n(λ) = 2.13 + 0.0149/λ² ${b.dn>=0?"+":"−"} ${Math.abs(b.dn).toFixed(3)} (λ in µm). The camera parameters describe the effective spectral response of the XC10 sensor through the BX51 optics. The narrower green channel and the larger blue flare mostly change how quickly contrast rises with thickness, which is what pulled the thick terraces into line.`;
})();
</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--cv", help="report.json of a --kfold run (adds held-out predictions)")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    r = json.loads(Path(a.report).read_text())
    if a.cv:
        c = json.loads(Path(a.cv).read_text())
        loo = {p["site"]: p["loo"] for p in c["points"]}
        for p in r["points"]:
            p["loo"] = loo.get(p["site"])
        r["summary"]["cross_validated"] = c["summary"].get("cross_validated")
    r["n_samples"] = 1977
    r["why"] = {14: "AFM probably on a neighbouring flake: 10x contrast matches ~4 nm hBN, AFM shows 17.6 nm",
                9: "step height fits either neighbouring terrace (T594 or T596)"}
    r["nostep"] = [8, 11, 12]
    r["samples"] = []
    data = json.dumps(r, separators=(",", ":")).replace("</", "<\\/")
    Path(a.output).write_text(PAGE.replace("__DATA__", data))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
