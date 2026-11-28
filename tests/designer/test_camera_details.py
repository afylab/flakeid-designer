"""The stacker's live camera Details window (2026-11-27): each step of opening the camera, the error with its traceback,
IC Imaging Control 4's retries, and an error while running no longer ends the view silently."""
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flakeid_designer"))
CAM = pytest.importorskip("camera")
if not CAM.available():
    pytest.skip("no OpenCV", allow_module_level=True)
HTML = (ROOT / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def wait(f, t=10.0):
    t0 = time.time()
    while time.time() - t0 < t:
        if f():
            return True
        time.sleep(0.05)
    return False


def test_sim_steps():
    L = CAM.Live()
    L.start("sim")
    L.last_req = time.time()
    assert wait(lambda: L.jpeg is not None)
    d = L.details()
    L.stop()
    assert "Camera: sim (Simulated camera)" in d and "Steps:" in d and "starting the simulated camera" in d and "first picture" in d
    assert "Computer:" in d and "OpenCV" in d and L.error is None


def test_error_while_running(monkeypatch):
    # (before: an exception in the grab loop's own code ended the thread: no picture and no error on the page)
    monkeypatch.setattr(CAM.SimCamera, "grab", lambda self: "not a picture")
    L = CAM.Live()
    L.start("sim")
    L.last_req = time.time()
    assert wait(lambda: L.error is not None)
    d = L.details()
    L.stop()
    assert L.error is None
    assert "the camera stopped: AttributeError" in d and "Error (traceback):" in d and "Traceback" in d


class IC4Error(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = types.SimpleNamespace(name=code)


def fake_ic4(opens_ok_after):
    n = {"open": 0}
    dev = types.SimpleNamespace(model_name="DFK 33UP1300", serial="123")

    class Grabber:
        def device_open(self, d):
            n["open"] += 1
            if n["open"] <= opens_ok_after:
                raise IC4Error("Device is already opened by another process", "DEVICE_IN_USE")
    m = types.SimpleNamespace(Library=types.SimpleNamespace(init=lambda **k: None), LogTarget=types.SimpleNamespace(),
                              DeviceEnum=types.SimpleNamespace(devices=lambda: [dev]), Grabber=Grabber)
    return m, n


def test_ic4_open_refused(monkeypatch):
    m, n = fake_ic4(opens_ok_after=99)
    monkeypatch.setitem(sys.modules, "imagingcontrol4", m)
    L = CAM.Live()
    L.start("ic4")
    assert wait(lambda: L.error is not None, 15)
    d = L.details()
    assert n["open"] == 2                                     # tried once more after 2 s
    assert "could not open it: IC4Error: Device is already opened by another process (error code DEVICE_IN_USE); trying again in 2 s" in d
    assert "ERROR IC4Error: Device is already opened by another process (error code DEVICE_IN_USE)" in d and "Traceback" in d
    assert "found 1: DFK 33UP1300 (123)" in d


def test_waits_for_previous(monkeypatch):
    monkeypatch.setattr(CAM.SimCamera, "close", lambda self: time.sleep(1.0))
    L = CAM.Live()
    L.start("sim")
    L.last_req = time.time()
    assert wait(lambda: L.jpeg is not None)
    L.start("sim")
    L.last_req = time.time()
    assert wait(lambda: L.jpeg is not None)
    d = L.details()
    L.stop()
    assert "waiting for the previous camera to close" in d and "has not closed" not in d


def test_page():
    assert '<button id="liveDet"' in HTML and '<dialog id="camDlg"' in HTML and 'fetch("/api/camera/details"' in HTML
    for b in ("camCopy", "camRetry", "camRefresh", "camClose"):
        assert f'id="{b}"' in HTML
    assert 'if(!await liveStart()){$("liveDet").onclick();return}' in HTML


def test_no_stale_frame(monkeypatch):
    # a frame grabbed while the camera is being stopped is not kept (it showed as the next camera's picture)
    L = CAM.Live()
    L.start("sim")
    L.last_req = time.time()
    assert wait(lambda: L.jpeg is not None)
    slow = CAM.SimCamera.grab
    monkeypatch.setattr(CAM.SimCamera, "grab", lambda self: (time.sleep(0.4), slow(self))[1])
    time.sleep(0.1)
    L.stop()
    time.sleep(0.8)
    assert L.jpeg is None
