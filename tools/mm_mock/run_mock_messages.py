"""The Messages popup (asked right after the start screen, before the chips are found): Slack (a member found as you
type), email (anyone; sent through an SMTP stand-in here) or no messages; the answer is this job's and is remembered.

    python tools/mm_mock/run_mock_messages.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.scope import notify  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import apply_messages, remembered_messages  # noqa: E402

MEMBERS = [dict(id="U01JORDAN", name="Jordan Pack", handle="jpack"), dict(id="U02ANNA", name="Anna Young", handle="ayoung")]


def walk(wd):
    yield wd
    for c in wd.winfo_children():
        yield from walk(c)


def popup(ui, s, act):
    """Open the popup from a worker thread (as the scan does), let act(window widgets) answer it; returns the answer."""
    from tkinter import ttk  # noqa: F401
    sc = type("FakeScanner", (), {"s": s})()
    res, seen = {}, {}

    def find():
        wins = [w for w in ui.root.winfo_children() if w.winfo_class() == "Toplevel" and w.title() == "Messages"]
        if not wins:
            ui.root.after(100, find)
            return
        ui.root.after(400, lambda: act(list(walk(wins[0])), seen))

    def work():
        res["r"] = ui.ask_messages(sc)
        ui._post(ui.root.quit)
    ui.root.after(100, find)
    threading.Thread(target=work, daemon=True).start()
    ui.root.after(15000, ui.root.quit)
    ui.root.mainloop()
    return res.get("r"), seen


def smtp_standin(port):
    """A plain SMTP server keeping what it gets (smtpd: Python 3.8 / 3.9)."""
    import asyncore
    import smtpd
    got = []

    class S(smtpd.SMTPServer):
        def process_message(self, peer, mailfrom, rcpttos, data, **kw):
            got.append(dict(frm=mailfrom, to=rcpttos, data=data.decode("utf-8", "replace") if isinstance(data, bytes) else data))
    S(("127.0.0.1", port), None)
    threading.Thread(target=lambda: asyncore.loop(timeout=0.2), daemon=True).start()
    return got


def main():
    out = ROOT / "tools" / "mm_mock" / "out" / "messages"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        from flakeid.scope.ui import TkUI
        ui = TkUI("messages popup test")
    except Exception as e:
        print(f"no Tk ({type(e).__name__}: {e}): not tested")
        return 0
    from tkinter import ttk
    import tkinter as tk
    os.environ["FLAKEID_SLACK_TOKEN"] = "xoxb-test"
    notify.members = lambda s, **k: list(MEMBERS)
    ok = True
    s = load(None, local_root=str(out), notify_smtp_host="127.0.0.1", notify_smtp_port=8826, notify_email_from="scope@lab.example")

    # 1. Slack: type part of a name, pick it from the list -> the member's ID
    def slack(ws, seen):
        e = [x for x in ws if isinstance(x, ttk.Entry)][0]
        e.focus_set()
        e.delete(0, "end")
        e.insert(0, "jor")
        e.event_generate("<KeyRelease>", keysym="r")
        ui.root.update()
        lst = next(x for x in ws if isinstance(x, tk.Listbox))
        seen["listed"] = list(lst.get(0, "end"))
        lst.selection_set(0)
        lst.event_generate("<Return>")
        ui.root.update()
        next(x for x in ws if isinstance(x, ttk.Button) and x.cget("text") == "OK").invoke()
    r, seen = popup(ui, s, slack)
    good = r == dict(via="slack", to="U01JORDAN", name="Jordan Pack", handle="jpack") and seen.get("listed") and seen["listed"][0].startswith("Jordan Pack")
    print(f"1. Slack: listed {seen.get('listed')}, answer {r}: {'ok' if good else 'WRONG'}")
    ok &= bool(good)
    apply_messages(s, r)
    ok &= s.notify_to == "U01JORDAN" and s.notify_email_to == ""

    # 2. email: someone not in the lab's Slack; the popup opens on Slack (last time's), typing the address switches
    def email(ws, seen):
        e = [x for x in ws if isinstance(x, ttk.Entry)][1]
        seen["start"] = next(x for x in ws if isinstance(x, ttk.Radiobutton) and x.cget("text").startswith("Slack")).instate(["selected"])
        e.focus_force()
        ui.root.update()
        e.insert(0, "visitor@ucsb.edu")
        e.event_generate("<KeyRelease>", keysym="u")
        ui.root.update()
        okb = next(x for x in ws if isinstance(x, ttk.Button) and x.cget("text") == "OK")
        okb.invoke()                                       # (no name for the screen yet: refused)
        ui.root.update()
        seen["noname"] = [x.cget("text") for x in ws if isinstance(x, ttk.Label) and "Type your name" in str(x.cget("text"))]
        n = [x for x in ws if isinstance(x, ttk.Entry)][2]
        n.insert(0, "Vera Visitor")
        okb.invoke()
    r, seen = popup(ui, s, email)
    good = r == dict(via="email", email="visitor@ucsb.edu", name="visitor@ucsb.edu", shown="Vera Visitor") and seen.get("start") \
        and seen.get("noname")
    print(f"2. email: opened on Slack (last time's) {seen.get('start')}, answer {r}: {'ok' if good else 'WRONG'}")
    ok &= bool(good)
    apply_messages(s, r)
    got = smtp_standin(8826)
    s.notify_on = tuple(s.notify_on) + ("test",)
    notify.send(s, "operator", "20261020_1: you are needed", "Put in the 100x objective.", wait=True, log=print)
    time.sleep(0.5)
    sent = got and got[0]["to"] == ["visitor@ucsb.edu"] and "you are needed" in got[0]["data"]
    print(f"   sent by email: {[(g['to'], g['data'].split('Subject: ')[1].splitlines()[0]) for g in got]}; Slack DM off: {s.notify_to == ''}")
    ok &= bool(sent) and s.notify_to == ""
    mem = remembered_messages(s)
    print(f"   remembered {mem}")
    ok &= mem["via"] == "email" and mem["email"] == "visitor@ucsb.edu" and mem["to"] == "U01JORDAN"   # (Slack's kept)

    # 3. email without an SMTP server on this PC: refused with a note; then No messages
    s3 = load(None, local_root=str(out))

    def nosmtp(ws, seen):
        okb = next(x for x in ws if isinstance(x, ttk.Button) and x.cget("text") == "OK")
        okb.invoke()                                     # (email selected from last time, no server)
        ui.root.update()
        seen["err"] = [x.cget("text") for x in ws if isinstance(x, ttk.Label) and "not set up" in str(x.cget("text"))]
        next(x for x in ws if isinstance(x, ttk.Radiobutton) and x.cget("text") == "No messages").invoke()
        okb.invoke()
    r, seen = popup(ui, s3, nosmtp)
    good = r == dict(via="none", shown="Vera Visitor") and len(seen.get("err") or []) >= 2
    print(f"3. no SMTP server: notes {seen.get('err')}; answer {r}: {'ok' if good else 'WRONG'}")
    ok &= bool(good)
    apply_messages(s3, r)
    ok &= s3.notify_to == "" and s3.notify_email_to == "" and not notify.configured(s3)

    # 4. the Scan in progress screen: who and how to reach them, status follows progress(), hide comes back, off at the end
    from flakeid.scope.ui import contact_of
    c = contact_of(dict(via="slack", to="U01JORDAN", name="Jordan Pack", handle="jpack"))
    ok &= c == dict(name="Jordan Pack", how="Slack: @jpack")
    ok &= contact_of(dict(via="email", email="v@ucsb.edu", shown="Vera")) == dict(name="Vera", how="Email: v@ucsb.edu")
    ok &= contact_of(dict(via="none", shown="Vera")) == dict(name="Vera", how="")
    ui.BANNER_HIDE_MIN = 0.01                              # (0.6 s)
    sc = type("FakeScanner", (), {"s": s})()
    ui.busy_banner(sc, c)
    ui.progress(sc, "piece 2 of 4: 10x tiles 120 / 336")
    st = {}

    def check():
        b = ui._banner
        st["texts"] = [x.cget("text") for x in walk(b["win"]) if x.winfo_class() == "Label"]
        st["shot"] = str(out / "banner.png")
        try:
            from PIL import ImageGrab
            b["win"].update()
            x, y, w_, h_ = b["win"].winfo_rootx(), b["win"].winfo_rooty(), b["win"].winfo_width(), b["win"].winfo_height()
            ImageGrab.grab((x, y, x + w_, y + h_)).save(st["shot"])
        except Exception as e:
            st["shot"] = f"no screenshot ({e})"
        b["hide"]()
        ui.root.after(100, lambda: st.update(hidden=not b["win"].winfo_viewable()))
        ui.root.after(1500, lambda: (st.update(back=b["win"].winfo_viewable()), ui.banner_off()))
        ui.root.after(1800, lambda: (st.update(gone=ui._banner is None), ui.root.quit()))
    ui.root.after(500, check)
    ui.root.mainloop()
    good = ("SCAN IN PROGRESS" in st.get("texts", []) and "Jordan Pack" in st["texts"] and "Slack: @jpack" in st["texts"]
            and "Now: piece 2 of 4: 10x tiles 120 / 336" in st["texts"] and st.get("hidden") and st.get("back") and st.get("gone"))
    print(f"4. banner: {st.get('texts')}; hidden {st.get('hidden')}, back {st.get('back')}, off {st.get('gone')}; {st.get('shot')}: "
          f"{'ok' if good else 'WRONG'}")
    ok &= bool(good)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
