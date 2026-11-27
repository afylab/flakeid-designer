"""Corrected flake outlines (2026-11-27, 100x window › Edit outline): kept in the "terraces" collection as
{kind: "outline", flake, outline} (no "poly", so older designers do not take them for drawn terraces), one per flake;
deleting one gives the scan's outline back.  python3 -m pytest tests/designer -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_server import req, server  # noqa: E402,F401


def test_outline_round_trip(server):
    u = server["url"] + "/api/db/terraces"
    doc = dict(kind="outline", flake="20261016_run_1:A:F001", outline=[0, 0, 10, 0, 10, 10, 0, 10], created=1)
    st, _, _ = req(u + "/20261016_run_1_A_F001.outline", "PUT", doc)
    assert st == 200
    st, _, b = req(u)
    items = {x["id"]: x["data"] for x in json.loads(b)}
    got = items["20261016_run_1_A_F001.outline"]
    assert got["kind"] == "outline" and got["outline"] == doc["outline"] and "poly" not in got and got["by"]
    f = server["root"] / "_designer_data" / "terraces" / "20261016_run_1_A_F001.outline.json"
    assert json.loads(f.read_text(encoding="utf-8"))["flake"] == doc["flake"]
    assert req(u + "/20261016_run_1_A_F001.outline", "DELETE")[0] == 200
    st, _, b = req(u)
    assert "20261016_run_1_A_F001.outline" not in {x["id"] for x in json.loads(b)}
