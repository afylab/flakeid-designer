"""(2026-11-18q) The AFM lists are not listed before the first scan of the runs has found their own data folders: a page
that asked meanwhile got no list kept in a run's folder (its card had no ✓ AFM, its flakes no Optical+AFM)."""
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flakeid_designer.__main__ as D  # noqa: E402
from test_server import make_run, req  # noqa: E402


def test_afm_plans_wait_for_the_first_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "settings.json")
    lib = tmp_path / "lib"
    make_run(lib, "20261103_run_1")
    store = D.Store([str(lib)], str(lib / "_designer_data"), str(lib))
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert not store.first_scan.is_set()
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", req(f"http://127.0.0.1:{srv.server_address[1]}/api/afm/plans?summary=1")))
        t.start()
        time.sleep(0.5)
        assert "r" not in out                      # (waiting for the scan)
        store.scan()
        t.join(10)
        st, _, body = out["r"]
        assert st == 200 and isinstance(json.loads(body), list)
    finally:
        srv.shutdown()
        srv.server_close()
