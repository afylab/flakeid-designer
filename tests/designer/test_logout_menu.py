"""(2026-11-27e) User ▾ › Log out: shown when a person is chosen, posts /api/people/logout.
python3 -m pytest tests/designer -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_hidden_layers import page  # noqa: E402,F401
from test_server import server  # noqa: E402,F401


def test_log_out_in_the_user_menu(page):  # noqa: F811
    pg, errs = page
    got = pg.evaluate("""async()=>{const r={};LOCAL.person=null;await openUserMenu();r.none=!!document.getElementById("umOut");
      closeUserMenu();LOCAL.person="Jordan";await openUserMenu();const b=document.getElementById("umOut");r.shown=!!b&&b.textContent;
      let sent=null;const f=window.fetch;window.fetch=(u,o)=>{sent=u;return Promise.resolve({json:()=>({error:"stop"})})};
      await logoutUser();window.fetch=f;r.sent=sent;return r}""")
    assert got["none"] is False and "Log out" in got["shown"] and "Jordan" in got["shown"]
    assert got["sent"] == "/api/people/logout"
    assert not errs, errs


def test_log_out_asks_about_unsaved_changes(page):  # noqa: F811
    pg, errs = page
    got = pg.evaluate("""async()=>{const r={};const lay=[{uid:"a",tid:"norun_0:c-f0000.a",x:0,y:0,rot:0,mirror:false,color:"#3fa3e0",
        opacity:0.8,showImg:true,showOutline:true,inOverlap:true}];
      const s={format:"vdw-stack/1",name:"T",status:"planning",plan:{layers:lay},current:{},steps:[],final:{images:[],poses:{},shapes:{},notes:""},updated:1};
      loadStack("d~t",s,true);STACKS["d~t"]=s;r.fresh=stackUnsaved();
      enterView("stack",true);r.stackView=stackUnsaved();enterView("finalize",true);r.finView=stackUnsaved();enterView("plan",true);
      $("dname").value="T2";r.renamed=stackUnsaved();
      let sent=[];const f=window.fetch;window.fetch=(u,o)=>{sent.push(u);return Promise.resolve({json:()=>({error:"stop"})})};
      LOCAL.person="Jordan";await openUserMenu();$("umOut").click();
      r.asked=!!$("umSave")&&!!$("umDrop")&&!!$("umStay");r.sentAfterAsk=sent.filter(u=>u==="/api/people/logout").length;
      $("umStay").click();r.cancelled=!$("umSave")&&stackUnsaved();
      $("umOut").click();$("umDrop").click();await new Promise(z=>setTimeout(z,50));
      r.dropped=!stackUnsaved()&&$("dname").value==="T";r.sent=sent.filter(u=>u==="/api/people/logout");
      $("dname").value="T3";markSaved();r.afterSave=stackUnsaved();
      window.fetch=f;return r}""")
    assert got["fresh"] is False and got["stackView"] is False and got["finView"] is False
    assert got["renamed"] is True and got["asked"] and got["sentAfterAsk"] == 0 and got["cancelled"]
    assert got["dropped"] and got["sent"] == ["/api/people/logout"] and got["afterSave"] is False
    assert not errs, errs
