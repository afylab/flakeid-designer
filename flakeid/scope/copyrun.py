"""Copy a finished run's results to the share again (a copy that was interrupted, e.g. the program closed during it):
the same layout as the scan's own copy (scan.share_layout: one folder per chip, its viewer on top), files already there
with the same size and time skipped, copy_done.json written last (the designer then builds the run).

    C:\\FlakeID\\python38\\python.exe -m flakeid.scope.copyrun graphite_20261019_run_1 --settings C:\\FlakeID\\scan_settings.json

The run is found by its run file (<local_root>/<run>.json): its chips, and where it was copied to ("results_copied_to";
else <copy_results_to>/<run>, or --to).  The analysis results: <results_dir or local_root/results>/<run>.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .scan import copy_parallel, share_layout, write_copy_done, write_viewer_link
from .settings import load


def copy_run(label, s, to=None, log=print):
    run_dir = Path(s.run_dir or s.local_root)
    man = Path(s.local_root) / f"{label}.json"
    if not man.is_file():
        man = run_dir / f"{label}.json"
    if not man.is_file():
        raise SystemExit(f"no run file {label}.json in {s.local_root} (the run's name, e.g. graphite_20261019_run_1)")
    M = json.loads(man.read_text())
    names = [p["name"] for p in M.get("pieces") or []]
    dst = Path(to) if to else Path(M.get("results_copied_to") or (Path(s.copy_results_to) / label if s.copy_results_to else ""))
    if not str(dst) or str(dst) == ".":
        raise SystemExit("where to copy to is not known: give --to <share folder>\\" + label)
    res = Path(s.results_dir or Path(s.local_root) / "results") / label
    log(f"{label}: {len(names)} chips ({', '.join(names)}) from {run_dir} and {res} -> {dst}")
    pairs, viewers = share_layout(man.parent, res if res.is_dir() else None, label, names, dst)
    todo = []
    for a, b in pairs:
        try:
            sa, sb = Path(a).stat(), Path(b).stat()
            if sa.st_size == sb.st_size and abs(sa.st_mtime - sb.st_mtime) < 2:
                continue                                  # (copied already)
        except OSError:
            pass
        todo.append((a, b))
    log(f"  {len(pairs)} files, {len(pairs) - len(todo)} already there, {len(todo)} to copy ...")
    t0 = time.time()
    nb = copy_parallel(todo) if todo else 0
    for nm, vfile in viewers.items():
        write_viewer_link(Path(dst) / nm / f"{nm} viewer.html", vfile)
    write_copy_done(Path(dst), names)
    log(f"  copied {len(todo)} files ({nb / 1e6:.0f} MB, {time.time() - t0:.0f} s); copy_done.json written: the designer builds the run")
    return len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flakeid.scope.copyrun", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("run", help="the run's name, e.g. graphite_20261019_run_1")
    ap.add_argument("--settings", help="the scan settings file (local_root, results_dir, copy_results_to)")
    ap.add_argument("--to", help="the run's folder on the share (default: where the scan copied it)")
    a = ap.parse_args(argv)
    copy_run(a.run, load(a.settings), a.to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
