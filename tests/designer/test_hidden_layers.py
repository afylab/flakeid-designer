"""(2026-11-27e) A stack's layers whose scan run is not loaded on this computer.

1. A run added on a lab PC through \\\\NAS-E6-27-E4\\... (or \\\\128.111.16.66\\...) while the person's folder is Z:\\...
   had no relative path, and a Mac could not open the absolute Windows one: the run never loaded there, and every stack
   using its flakes lost those layers (DQD15: two hBN layers).  Paths on the share are now compared below its
   Microscope Images folder, however each computer reaches it.
2. The page kept such layers aside and put them at the end of the list (the bottom of the stack) on the next save: the
   stack came back in another order.  They now keep their place, show as rows in the layer list, and come back as soon
   as their run is loaded.
python3 -m pytest tests/designer -q"""
import json
import ntpath
import posixpath
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.people as P  # noqa: E402
from test_server import server  # noqa: E402,F401

MAC = "/Volumes/Documents/Microscope Images/Jordan"
PC = r"Z:\Microscope Images\Jordan"
UNC = r"\\NAS-E6-27-E4\Documents\Microscope Images\Jordan\scanresults\112726_run_1"
IP = r"\\128.111.16.66\Documents\Microscope Images\Paul\exfoliations\BN\BN7\BN7_run_1"


def test_unc_run_added_against_a_drive_letter_is_relative():
    assert P.rel_of(UNC, PC, ntpath) == "scanresults/112726_run_1"
    assert P.rel_of(IP, PC, ntpath) == "../Paul/exfoliations/BN/BN7/BN7_run_1"
    d = P.set_run(P.new_doc("Jordan"), UNC, True, PC, 1, ntpath)
    assert list(d["runs"]) == ["scanresults/112726_run_1"]
    assert P.run_paths(d, MAC, posixpath) == [MAC + "/scanresults/112726_run_1"]
    assert P.rel_of(r"Y:\other\run", r"Z:\Jordan\scanresults", ntpath) is None          # (not on the share: as before)


def test_entries_saved_before_resolve_on_every_computer():
    """The person file as the lab PC wrote it (rel: null, the Windows path) and the ticked key a Mac then wrote."""
    d = P.new_doc("Jordan")
    d["runs"] = {"//NAS-E6-27-E4/Documents/Microscope Images/Jordan/scanresults/112726_run_1":
                 dict(rel=None, path=UNC, on=True, t=2),
                 "//128.111.16.66/Documents/Microscope Images/Paul/exfoliations/BN/BN7/BN7_run_1":
                 dict(rel=None, path=IP, on=True, t=1),
                 "scanresults/gr_112126_run_2": dict(rel="scanresults/gr_112126_run_2", path=r"Z:\x", on=True, t=0)}
    d["ticked"] = dict(keys=["../../../../NAS-E6-27-E4/Documents/Microscope Images/Jordan/scanresults/112726_run_1",
                             "scanresults/gr_112126_run_2"], t=3)
    assert P.run_paths(d, MAC, posixpath) == [MAC + "/scanresults/gr_112126_run_2",
                                              "/Volumes/Documents/Microscope Images/Paul/exfoliations/BN/BN7/BN7_run_1",
                                              MAC + "/scanresults/112726_run_1"]
    assert P.ticked_paths(d, MAC, posixpath) == [MAC + "/scanresults/112726_run_1", MAC + "/scanresults/gr_112126_run_2"]
    assert P.run_paths(d, PC, ntpath)[-1] == PC + r"\scanresults\112726_run_1"
    assert P.ticked_paths(d, PC, ntpath)[0] == PC + r"\scanresults\112726_run_1"
    # the Mac saving its list again changes nothing (the same runs, matched by this computer's path)
    before = json.dumps(d, sort_keys=True)
    P.set_list(d, P.run_paths(d, MAC, posixpath), MAC, 9, posixpath)
    assert json.dumps(d, sort_keys=True) == before


@pytest.fixture()
def page(server):  # noqa: F811
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as e:                       # (no browser installed)
            pytest.skip(f"no headless Chromium: {e}")
        pg = b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(server["url"] + "/")
        pg.wait_for_function("typeof withHidden==='function'&&typeof loadStack==='function'")
        yield pg, errs
        b.close()


def test_hidden_layers_keep_their_place(page):
    pg, errs = page
    got = pg.evaluate("""()=>{const L=u=>({uid:u}),ids=a=>a.map(x=>x.uid).join("");
      PLAN_ORDER=[..."abcdef"];const r={};
      r.same=ids(withHidden([L("a"),L("c"),L("e")],[L("b"),L("d"),L("f")]));
      r.moved=ids(withHidden([L("e"),L("a"),L("c")],[L("b"),L("d"),L("f")]));
      r.gone=ids(withHidden([L("c")],[L("b"),L("d")]));            // (a and e removed meanwhile)
      PLAN_ORDER=[];r.unknown=ids(withHidden([L("a")],[L("z")]));
      return r}""")
    assert got == dict(same="abcdef", moved="efabcd", gone="bcd", unknown="az")
    assert not errs, errs


def test_stack_saved_without_its_runs_keeps_the_order(page):
    pg, errs = page
    got = pg.evaluate("""()=>{const lay=["top","mid","bot"].map((u,i)=>({uid:u,tid:`norun_${i}:c-f000${i}.a`,x:i,y:0,rot:0,
        mirror:false,color:"#3fa3e0",opacity:0.8,showImg:true,showOutline:true,inOverlap:true}));
      const s={format:"vdw-stack/1",name:"T",status:"planning",plan:{layers:lay},current:{},steps:[],final:{},updated:1};
      loadStack("t1",s,true);const body=stackBody();renderStack();
      return {n:layers.length,hidden:HIDDEN_LAYERS.length,order:body.plan.layers.map(L=>L.uid),
              rows:document.querySelectorAll("#stack .layer.missing").length,
              text:document.querySelector("#stack .layer.missing .meta").textContent}}""")
    assert got["n"] == 0 and got["hidden"] == 3
    assert got["order"] == ["top", "mid", "bot"]
    assert got["rows"] == 3 and "tick norun_2 in Library" in got["text"]
    assert not errs, errs


def test_layer_comes_back_when_its_run_loads(page):
    pg, errs = page
    got = pg.evaluate("""()=>{const lay=["top","mid","bot"].map((u,i)=>({uid:u,tid:`norun_${i}:c-f000${i}.a`,x:i,y:0,rot:0,
        mirror:false,color:"#3fa3e0",opacity:0.8,showImg:true,showOutline:true,inOverlap:true}));
      loadStack("t2",{format:"vdw-stack/1",name:"T",status:"planning",plan:{layers:lay},current:{},steps:[],final:{},updated:1},true);
      TR["norun_1:c-f0001.a"]={t:{},f:{terraces:[]}};reviveHidden();
      const r={layers:layers.map(L=>L.uid),hidden:HIDDEN_LAYERS.map(L=>L.uid),order:stackBody().plan.layers.map(L=>L.uid)};
      delete TR["norun_1:c-f0001.a"];return r}""")
    assert got == dict(layers=["mid"], hidden=["top", "bot"], order=["top", "mid", "bot"])
    assert not errs, errs
