"""Saving an AFM point list to a folder (/api/afm/save, 2026-11-03j): the .pgm and .csv copies are named with the time
saved, to the second (<stem>_afm_sites_YYYYMMDD_HHMMSS.pgm), and a second save never writes over the first."""
import json
import re
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_save_named_with_time(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    import flakeid_designer.__main__ as D
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "root"
    root.mkdir()
    store = D.Store([root], root / "_designer_data")
    d = store._dir("afm")
    pid = "chip1_20261103_120000"
    (d / pid).mkdir(parents=True)
    (d / pid / "chip1_afm_sites.pgm").write_bytes(b"Version: 1.0\r\n")
    (d / pid / "chip1_afm_sites.csv").write_text("n,flake\r\n")
    (d / pid / "plan.json").write_text(json.dumps(dict(id=pid, scan="chip1", sites=[], pgm="chip1_afm_sites.pgm", csv="chip1_afm_sites.csv")))
    out = tmp_path / "afm_pc"
    out.mkdir()
    monkeypatch.setattr(D, "pick_folder", lambda *a, **k: str(out))
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        def post(body):
            r = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/afm/save", data=json.dumps(body).encode(),
                                       method="POST", headers={"Content-Type": "application/json", "X-Designer": "1"})
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        a, b = post(dict(id=pid)), post(dict(id=pid))
        pat = r"chip1_afm_sites_\d{8}_\d{6}(_\d+)?\.pgm"
        assert re.fullmatch(pat, Path(a["saved"]).name) and re.fullmatch(pat, Path(b["saved"]).name)
        assert a["saved"] != b["saved"]
        assert Path(a["saved"]).read_bytes() == b"Version: 1.0\r\n" and Path(a["saved"]).with_suffix(".csv").is_file()
        assert sorted(p.suffix for p in out.iterdir()) == [".csv", ".csv", ".pgm", ".pgm"]
        assert (d / pid / "chip1_afm_sites.pgm").is_file()                     # (the kept copy keeps its name)
    finally:
        srv.shutdown()
        srv.server_close()


def test_generate_saves_without_dialog(tmp_path, monkeypatch):
    """(2026-11-03k) Generate AFM file = /api/afm/save {auto}: no dialog when a folder is known (the one chosen by hand
    last, remembered in the config; else the run's folder); with none known, the dialog; by hand the choice is kept."""
    pytest.importorskip("numpy")
    import flakeid_designer.__main__ as D
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "cfg.json")
    root = tmp_path / "root"
    root.mkdir()
    store = D.Store([root], root / "_designer_data")
    d = store._dir("afm")
    pid = "chip1_20261103_120000"
    (d / pid).mkdir(parents=True)
    (d / pid / "chip1_afm_sites.pgm").write_bytes(b"Version: 1.0\r\n")
    (d / pid / "chip1_afm_sites.csv").write_text("n,flake\r\n")
    (d / pid / "plan.json").write_text(json.dumps(dict(id=pid, scan="chip1", sites=[], pgm="chip1_afm_sites.pgm", csv="chip1_afm_sites.csv")))
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    calls = []
    pick = {"to": str(a_dir)}
    monkeypatch.setattr(D, "pick_folder", lambda prompt, start=None: calls.append(start) or pick["to"])
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        def post(body):
            r = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/afm/save", data=json.dumps(body).encode(),
                                       method="POST", headers={"Content-Type": "application/json", "X-Designer": "1"})
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        r = post(dict(id=pid, auto=True))                     # (no folder known, no run: the dialog)
        assert len(calls) == 1 and Path(r["saved"]).parent == a_dir and r["plan"]["saved_to"] == r["saved"]
        r = post(dict(id=pid, auto=True))                     # (the folder chosen is remembered: no dialog)
        assert len(calls) == 1 and Path(r["saved"]).parent == a_dir
        pick["to"] = str(b_dir)
        r = post(dict(id=pid))                                # (by hand: the dialog opens in the remembered folder)
        assert calls[-1] == str(a_dir) and Path(r["saved"]).parent == b_dir
        r = post(dict(id=pid, auto=True))
        assert len(calls) == 2 and Path(r["saved"]).parent == b_dir
        assert json.loads((d / pid / "plan.json").read_text())["saved_to"] == r["saved"]
        assert json.loads((tmp_path / "cfg.json").read_text())["afm_save_dir"] == str(b_dir)
    finally:
        srv.shutdown()
        srv.server_close()
