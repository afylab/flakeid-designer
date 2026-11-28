"""(2026-11-27e) A folder stack has one id on every computer: made from its path below the share's Microscope Images
folder.  Before, the id was a hash of the path as each computer writes it (a Mac's /Volumes/..., a PC's Z:\\... or
\\\\NAS-...), and the shared pick-up decisions ("used", design_id) carried that computer's id: undoing a pick-up on
another computer left the flake "used".  The old ids are still recognised.  python3 -m pytest tests/designer -q"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.stackdir as SD  # noqa: E402
from test_hidden_layers import page  # noqa: E402,F401
from test_server import server  # noqa: E402,F401

MAC = "/Volumes/Documents/Microscope Images/Jordan/Devices/DQD/DQD15"
PC = r"Z:\Microscope Images\Jordan\Devices\DQD\DQD15"
UNC = r"\\NAS-E6-27-E4\Documents\Microscope Images\Jordan\Devices\DQD\DQD15"


def old_id(key):
    return "d~" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def test_one_id_on_every_computer():
    ids = {SD.folder_id(p) for p in (MAC, PC, UNC, MAC.replace("DQD15", "dqd15"), MAC + "/")}
    assert len(ids) == 1
    assert SD.folder_id(MAC) != SD.folder_id(MAC.replace("DQD15", "DQD14"))
    assert SD.folder_id("/elsewhere/S1") == SD.legacy_folder_id("/elsewhere/S1")      # (not on the share: as before)


def test_old_ids_recognised():
    old = set(SD.legacy_ids(MAC))
    assert old_id(MAC) in old                                                   # (the Mac's)
    assert old_id(PC.lower()) in old and old_id(UNC.lower()) in old             # (a PC's: normcase writes lower case)
    assert SD.folder_id(MAC) not in old
    assert SD.same_folder(PC, old_id(MAC)) and not SD.same_folder(MAC, old_id("/x"))


def test_folder_stack_opened_and_saved_with_an_old_id(tmp_path):
    folder = tmp_path / "Documents" / "Microscope Images" / "J" / "Dev" / "S1"
    folder.mkdir(parents=True)
    cfg = {}
    F = SD.Folders(lambda: json.loads(json.dumps(cfg)), lambda c: cfg.update(c), lambda: "J")
    sid, _, fname, exists = F.register(folder)
    assert sid == SD.folder_id(folder) and not exists
    F.put(sid, dict(name="S1", plan=dict(layers=[])), "mac")
    F.remember(sid, "S1")
    old = SD.legacy_folder_id(folder)                     # (what a browser kept from before)
    F2 = SD.Folders(lambda: json.loads(json.dumps(cfg)), lambda c: cfg.update(c), lambda: "J")
    assert F2.where(old)[0] == folder and F2.where(sid)[0] == folder
    page_doc = F2.read(sid)
    assert old in page_doc["_ids"]
    body = dict(page_doc, plan=dict(layers=[dict(uid="a", photos=[dict(url=f"/stackfile/{old}/01_stacking/p.jpg")])]))
    F2.put(old, body, "pc")
    saved = json.loads((folder / fname).read_text(encoding="utf-8"))
    assert saved["plan"]["layers"][0]["photos"][0]["url"] == "stack:01_stacking/p.jpg"
    assert "_ids" not in saved
    F2.remember(old, "S1")
    assert len(cfg["recent_stacks"]) == 1                 # (the old id and the new one: one recent stack)


def test_pickup_undone_with_an_old_id(page):  # noqa: F811
    pg, errs = page
    got = pg.evaluate("""()=>{curStackId="d~new";STACK={_ids:["d~old"]};
      return [isThisStack("d~new"),isThisStack("d~old"),isThisStack("d~other"),isThisStack(null)]}""")
    assert got == [True, True, False, False]
    assert not errs, errs
