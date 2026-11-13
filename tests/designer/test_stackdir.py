"""Stacks in folders of their own (flakeid_designer/stackdir.py): open / new, save in place with references relative to
the folder, uploads into 01_stacking/ and final/, step folders (scanner's and imported), lock, recent list."""
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flakeid_designer.__main__ as D  # noqa: E402
from test_server import make_run, req  # noqa: E402


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "config.json")        # (never the user's own config)
    root = tmp_path / "root"
    make_run(root)
    store = D.Store([root], root / "_designer_data")
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield dict(url=f"http://127.0.0.1:{srv.server_address[1]}", store=store, tmp=tmp_path)
    srv.shutdown()
    srv.server_close()


def post_bytes(url, data, ctype, headers=None):
    r = urllib.request.Request(url, data=data, method="POST", headers=dict({"Content-Type": ctype}, **(headers or {})))
    with urllib.request.urlopen(r, timeout=10) as resp:
        return json.loads(resp.read())


def scanner_step(folder, dname, device):
    d = folder / dname
    d.mkdir()
    (d / f"{device}_contacts_100x.jpg").write_bytes(b"\xff\xd8jpg100\xff\xd9")
    (d / f"{device}_contacts_10x.jpg").write_bytes(b"\xff\xd8jpg10\xff\xd9")
    (d / f"{device}_contacts.html").write_text("<html>step</html>")
    (d / "step.json").write_text(json.dumps(dict(
        format="vdw-fabstep/1", name="contacts", description="Cr/Au", at=1760900000000, by="scope", device=device, oxide_nm=285,
        source="scanner device photos", page=f"{device}_contacts.html",
        photos=[dict(objective="100x", label="main", file=f"{device}_contacts_100x.jpg", um_per_px=0.069, w=10, h=8, terraces=[]),
                dict(objective="10x", label="main", file=f"{device}_contacts_10x.jpg", um_per_px=0.69, w=10, h=8)])))


def test_folder_stack_roundtrip(server):
    u, tmp = server["url"], server["tmp"]
    folder = tmp / "Devices" / "DQD" / "DQD8"
    folder.mkdir(parents=True)
    # Open on a folder without a stack: its id, the file it will be (named after the folder)
    st, _, b = req(u + "/api/stackdir/open", "POST", dict(path=str(folder)))
    r = json.loads(b)
    assert st == 200 and r["id"].startswith("d~") and not r["exists"] and r["file"] == "DQD8.stack"
    sid = r["id"]
    # a pick-up picture and an attachment go into 01_stacking/, a final original into final/
    pic = post_bytes(f"{u}/api/stackimage?stack={sid}", b"\xff\xd8pic\xff\xd9", "image/jpeg")
    att = post_bytes(f"{u}/api/stackfile?stack={sid}&name=afm%20scan.ibw", b"afm", "application/octet-stream", {"X-Designer": "1"})
    fin = post_bytes(f"{u}/api/stackfile?stack={sid}&name=final.tif&sub=final", b"tif", "application/octet-stream", {"X-Designer": "1"})
    assert pic["url"].startswith(f"/stackfile/{sid}/01_stacking/") and att["url"].startswith(f"/stackfile/{sid}/01_stacking/")
    assert fin["url"] == f"/stackfile/{sid}/final/final.tif" and (folder / "final" / "final.tif").read_bytes() == b"tif"
    assert req(u + pic["url"])[2] == b"\xff\xd8pic\xff\xd9"
    assert req(u + f"/stackfile/{sid}/../config.json")[0] == 404
    # save: references in the file relative to the folder
    body = dict(name="DQD8", session="p1", steps=[dict(photos=[dict(url=pic["url"])], attachments=[att])], final=dict(images=[dict(url=fin["url"])]))
    st, _, b = req(f"{u}/api/db/stacks/{sid}", "PUT", body)
    assert st == 200
    raw = (folder / "DQD8.stack").read_text()
    assert f"/stackfile/{sid}" not in raw and '"stack:01_stacking/' in raw and '"stack:final/final.tif"' in raw
    # the scanner writes a step folder: it is in the stack (photos, page) at the next read
    scanner_step(folder, "02_contacts", "DQD8")
    lst = {x["id"]: x["data"] for x in json.loads(req(u + "/api/db/stacks")[2])}
    s = lst[sid]
    assert s["steps"][0]["photos"][0]["url"] == pic["url"] and s["folder"] == str(folder)
    fs = s["fab"]["steps"]
    assert [x["dir"] for x in fs] == ["02_contacts"] and fs[0]["photos"][0]["url"] == f"/stackfile/{sid}/02_contacts/DQD8_contacts_100x.jpg"
    assert fs[0]["page"] and req(u + fs[0]["photos"][1]["url"])[2] == b"\xff\xd8jpg10\xff\xd9"
    # a description edited in the designer goes back to step.json, the step is not copied into the .stack
    fs[0]["description"] = "Cr/Au 5/50 nm edge contacts"
    st, _, b = req(f"{u}/api/db/stacks/{sid}", "PUT", dict(s, session="p1", _base_sig=s["_sig"]))
    assert st == 200
    assert json.loads((folder / "02_contacts" / "step.json").read_text())["description"] == "Cr/Au 5/50 nm edge contacts"
    assert "02_contacts" not in (folder / "DQD8.stack").read_text()
    # another page with the old version: refused
    assert req(f"{u}/api/db/stacks/{sid}", "PUT", dict(s, session="p2", _base_sig=s["_sig"]))[0] == 409
    # recent list; the folder moved / opened from elsewhere: same content, references follow
    rec = json.loads(req(u + "/api/stackdir/recent")[2])
    assert rec[0]["id"] == sid and rec[0]["file"] == "DQD8.stack"
    moved = tmp / "elsewhere" / "DQD8"
    moved.parent.mkdir()
    folder.rename(moved)
    r2 = json.loads(req(u + "/api/stackdir/open", "POST", dict(path=str(moved / "DQD8.stack")))[2])
    assert r2["exists"] and r2["id"] != sid
    d2 = r2["data"]
    assert d2["steps"][0]["photos"][0]["url"].startswith(f"/stackfile/{r2['id']}/01_stacking/")
    assert req(u + d2["steps"][0]["photos"][0]["url"])[0] == 200
    assert d2["fab"]["steps"][0]["description"] == "Cr/Au 5/50 nm edge contacts"


def test_folder_lock_import_and_removed(server):
    u, tmp = server["url"], server["tmp"]
    folder = tmp / "DQD9"
    folder.mkdir()
    sid = json.loads(req(u + "/api/stackdir/open", "POST", dict(path=str(folder)))[2])["id"]
    assert req(f"{u}/api/db/stacks/{sid}", "PUT", dict(name="DQD9", session="p1"))[0] == 200
    # lock in the folder
    ok = json.loads(req(u + "/api/stacklock", "POST", dict(id=sid, session="p1"))[2])
    other = json.loads(req(u + "/api/stacklock", "POST", dict(id=sid, session="p2"))[2])
    assert ok["ok"] and not other["ok"] and (folder / ".designer_lock.json").is_file()
    # an older Device photos set imported: a new step folder NN_<step> with step.json and renamed photos
    src = tmp / "device_photos" / "DQD9_20261019"
    src.mkdir(parents=True)
    (src / "a100.jpg").write_bytes(b"1")
    (src / "a5.jpg").write_bytes(b"5")
    (src / "map.jpg").write_bytes(b"m")
    (src / "t1.jpg").write_bytes(b"t")
    (src / "DQD9_photos.json").write_text(json.dumps(dict(device="DQD9", oxide_nm=90, photos=[
        dict(objective="100x", label="main", jpg="a100.jpg", um_per_px=0.07, w=4, h=3, terrace_map="map.jpg",
             terraces=[dict(n=1, area_um2=40, contrast=0.05, hc="t1.jpg")]),
        dict(objective="5x", label="main", jpg="a5.jpg", um_per_px=1.4, w=4, h=3)])))
    (folder / "02_etch").mkdir()                          # (a step folder without step.json still takes a number)
    st, _, b = req(u + "/api/devicephotos/import", "POST", dict(stack=sid, path=str(src), title="top gate", description="Pd 20 nm"),
                   headers={"X-Designer": "1"})
    step = json.loads(b)["step"]
    assert st == 200 and step["dir"] == "03_top_gate" and step["description"] == "Pd 20 nm"
    sj = json.loads((folder / "03_top_gate" / "step.json").read_text())
    assert [p["file"] for p in sj["photos"]] == ["DQD9_top_gate_100x.jpg", "DQD9_top_gate_5x.jpg"]
    assert sj["photos"][0]["terraces"][0]["hc"] == "DQD9_top_gate_100x_T1.jpg" and sj["source"] == "designer import"
    assert req(u + step["photos"][0]["terraces"][0]["hc"])[2] == b"t"
    # a step taken off the stack stays off (its folder stays)
    s = {x["id"]: x["data"] for x in json.loads(req(u + "/api/db/stacks")[2])}[sid]
    assert [x["dir"] for x in s["fab"]["steps"]] == ["03_top_gate"]
    s["fab"] = dict(steps=[], removed=["03_top_gate"])
    assert req(f"{u}/api/db/stacks/{sid}", "PUT", dict(s, session="p1", _base_sig=s["_sig"]))[0] == 200
    s = {x["id"]: x["data"] for x in json.loads(req(u + "/api/db/stacks")[2])}[sid]
    assert s["fab"]["steps"] == [] and (folder / "03_top_gate" / "step.json").is_file()
    # older stacks still work beside folder stacks
    assert req(u + "/api/db/stacks/s_old", "PUT", dict(name="old", session="p1"))[0] == 200
    ids = [x["id"] for x in json.loads(req(u + "/api/db/stacks")[2])]
    assert "s_old" in ids and sid in ids
