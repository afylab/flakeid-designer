"""A run's 100x export finishing in the background while the next run starts (export_in_background), with the Slack
bot (a stand-in Slack server): the start screen is back at once, "scanning finished" goes to the run's thread, the
export's progress line says "previous run", re-imaging one of its chips is refused meanwhile, and "export finished"
(with the copy on the share) goes to the first run's thread even though the second run has started its own.

    python tools/mm_mock/run_mock_bgexport.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "mm_mock"))
import slack_standin  # noqa: E402
from flakeid.scope import notify  # noqa: E402
from flakeid.scope.mock import MockCore  # noqa: E402
from flakeid.scope.scan import EXPORTING, Scanner, Stop  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import AutoUI  # noqa: E402

FAKE_EXPORT = r"""
import sys, time
from pathlib import Path
res, items = Path(sys.argv[1]), sys.argv[2:]
for k, it in enumerate(items):
    chip, stem, kind = it.split("|")
    v = res / chip / f"{chip}_tiles_viewer" / "100x" / ("domains" if kind == "IR" else "")
    v.mkdir(parents=True, exist_ok=True)
    (v / (stem + (".png" if kind == "IR" else ".jpg"))).write_bytes(b"x")
    print(f"  100x: writing images: {int(100 * (k + 1) / len(items)):3d}% ({k + 1}/{len(items)}), 1s elapsed, 1s left", flush=True)
    time.sleep(0.8)
"""


def main():
    out = ROOT / "tools" / "mm_mock" / "out" / "bgexport"
    if out.exists():
        shutil.rmtree(out)
    rd, share = out / "local", out / "share"
    rd.mkdir(parents=True); share.mkdir()
    srv, calls, st = slack_standin.start(8816, users={"op@lab.example": "U01OPERATR"}, names={"U01OPERATR": "Test Operator"})
    notify.SLACK_API = "http://127.0.0.1:8816/api/"
    os.environ["FLAKEID_SLACK_TOKEN"] = "xoxb-test"
    progress, ok = [], True

    class U(AutoUI):
        def analysis_progress(self, sc, frac, text):
            progress.append((frac, text))

    def scanner(label):
        s = load(None, mode="multi", local_root=str(rd), out_dir=str(rd), copy_results_to=str(share), notify_to="Test Operator",
                 notify_min_job_min=0.0)
        sc = Scanner(MockCore(1.09), s, U(MockCore(1.09), analysis=False))
        sc.run_dir, sc.run_label, sc.results_dest = str(rd), label, str(share / label)
        return sc

    # ---- run 1: two chips, 4 + 3 pictures at 100x, 2 IR pictures of the first
    sc1 = scanner("t_run_1")
    sc1.pieces = [dict(name="t_1", c=(0, 0, 0), state="done", material="graphite", oxide=285.0),
                  dict(name="t_2", c=(0, 0, 0), state="done", material="graphite", oxide=285.0)]
    items = []
    for nm, n, nd in (("t_1", 4, 2), ("t_2", 3, 0)):
        (rd / f"{nm}_100x").mkdir()
        (rd / f"{nm}_100x" / f"{nm}_100x_index.csv").write_text(
            "order,label,file\n" + "".join(f"{k},{k:02d}_F{k:03d}_5nm,{nm}_F{k:03d}_5nm_100x_{k:03d}.tif\n" for k in range(1, n + 1)))
        items += [f"{nm}|{nm}_F{k:03d}_5nm_100x_{k:03d}|100x" for k in range(1, n + 1)]
        if nd:
            (rd / f"{nm}_domains").mkdir()
            (rd / f"{nm}_domains" / f"{nm}_domains_index.csv").write_text(
                "n,label,file\n" + "".join(f"{k},l,{nm}_F{k:03d}_5nm_IR_{k:03d}.tif\n" for k in range(1, nd + 1)))
            items += [f"{nm}|{nm}_F{k:03d}_5nm_IR_{k:03d}|IR" for k in range(1, nd + 1)]
    res = out / "results" / "t_run_1"
    res.mkdir(parents=True)
    sc1.analysis_out = str(res)
    logp = res / "analysis_log.txt"
    sc1.slack_run = notify.start_run(sc1.s, "Scan t_run_1 started", "2 chips", log=print)
    sc1.analysis = subprocess.Popen([sys.executable, "-c", FAKE_EXPORT, str(res)] + items, stdout=open(logp, "a"), stderr=subprocess.STDOUT)
    threading.Thread(target=sc1._follow_analysis, args=(logp,), daemon=True).start()
    time.sleep(1.0)

    t0 = time.time()
    sc1.finish_analysis()
    back = time.time() - t0
    prio = os.getpriority(os.PRIO_PROCESS, sc1.analysis.pid) if sc1.analysis.poll() is None else None
    print(f"run 1: start screen back after {back:.2f} s; finishing in the background: {sc1.upload_thread.is_alive()}; "
          f"export priority (nice) {prio}; chips exporting {sorted(EXPORTING)}")
    ok &= back < 2.0 and sc1.upload_thread.is_alive() and prio == 10 and EXPORTING == {"t_1", "t_2"}

    # ---- run 2 starts meanwhile: its own thread; re-imaging a chip of run 1 is refused
    sc2 = scanner("t_run_2")
    sc2.slack_run = notify.start_run(sc2.s, "Scan t_run_2 started", "1 chip", log=print)
    sc2.alert("Put in the 100x objective (nosepiece position 5).")
    sc2.s.retarget_piece = "t_1"
    try:
        sc2.run_retarget()
        refused = False
    except Stop as e:
        refused = "still being exported" in str(e)
    print(f"run 2: re-imaging t_1 meanwhile refused: {refused}")
    ok &= refused

    sc1.upload_thread.join(60)
    notify.flush(30)
    copied = sorted(str(p.relative_to(share)) for p in (share / "t_run_1").rglob("*.jpg"))
    print(f"run 1: copied to the share: {len(copied)} JPEGs, copy_done {(share / 't_run_1' / 'copy_done.json').is_file()}; chips exporting now {sorted(EXPORTING)}")
    ok &= len(copied) == 7 and (share / "t_run_1" / "copy_done.json").is_file() and not EXPORTING
    prev = [t for f, t in progress if t.startswith("previous run t_run_1")]
    print(f"progress line: {len(progress)} updates, {len(prev)} 'previous run' ones, e.g. {prev[len(prev) // 2] if prev else None!r}; "
          f"last {progress[-1] if progress else None}")
    ok &= bool(prev) and progress[-1][0] == "done"

    posts = [d for m, d in calls if m == "chat.postMessage"]
    tops = {d["text"].split("*")[1]: None for d in posts if not d.get("thread_ts")}
    ts_of = {}
    for i, d in enumerate(posts):
        if not d.get("thread_ts"):
            ts_of[d["text"].split("*")[1]] = f"{1001 + i}.0001"
    for d in posts:
        thread = next((k for k, v in ts_of.items() if v == d.get("thread_ts")), "-")
        print(f"   {'THREAD' if not d.get('thread_ts') else 'reply ':6s} in {thread:22s} {d['text'].splitlines()[0][:60]}{'  (broadcast)' if d.get('reply_broadcast') else ''}")
    texts = [d["text"].splitlines()[0] for d in posts]       # (no threads: replies do not notify; each a new message)
    ok &= (not any(d.get("thread_ts") for d in posts) and not any("started" in t for t in texts)
           and texts.index(next(t for t in texts if "t_run_1: scanning finished" in t)) < texts.index(next(t for t in texts if "t_run_1: export finished" in t))
           and any("you are needed" in t for t in texts))
    srv.shutdown()
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
