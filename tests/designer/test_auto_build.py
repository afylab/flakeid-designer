"""(2026-11-27h) Builds start by themselves: older runs without designer files too, and a run the share is slow for
once no other computer has built it in Builder.SLOW_WAIT_S."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flakeid_designer.__main__ as D  # noqa: E402


class B(D.Builder):
    def __init__(self, slow=None):
        super().__init__()
        self.built, self.slow_msg = [], slow

    def too_slow(self, d):
        return self.slow_msg

    def _build(self, d, full=False):
        self.built.append(Path(d).name)
        self._set(d, state="done", msg="built")


def settle(b):
    t0 = time.time()
    while time.time() - t0 < 5 and (b.thread is not None and b.thread.is_alive()):
        time.sleep(0.02)


def test_older_runs_built(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "build_here", lambda: True)
    b = B()
    b.consider(None, [tmp_path / "new_run"], [tmp_path / "old_run"])
    settle(b)
    assert sorted(b.built) == ["new_run", "old_run"]


def test_not_on_a_computer_set_not_to_build(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "build_here", lambda: False)
    b = B()
    b.consider(None, [tmp_path / "new_run"], [tmp_path / "old_run"])
    settle(b)
    assert b.built == [] and {v["state"] for v in b.state.values()} == {"off", "unbuilt"}


def test_slow_share_waits_then_builds(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "build_here", lambda: True)
    b = B(slow="the share is slow from this computer")
    d = tmp_path / "run_x"
    b.consider(None, [d], [])
    settle(b)
    st = b.state[str(d)]
    assert b.built == [] and st["state"] == "slow" and "since" in st
    b.consider(None, [d], [])                         # (a later walk, still within the wait: nothing)
    settle(b)
    assert b.built == [] and b.state[str(d)]["state"] == "slow"
    b.state[str(d)]["since"] -= D.Builder.SLOW_WAIT_S + 1
    b.consider(None, [d], [])                         # (no other computer built it meanwhile: built here)
    settle(b)
    assert b.built == ["run_x"]


def test_slow_message_and_banner():
    src = (ROOT / "flakeid_designer" / "__main__.py").read_text(encoding="utf-8")
    assert 'f"{self.SLOW_WAIT_S // 60} min to a computer on the lab network (minutes there), then built here")' in src
    assert D.Builder.SLOW_WAIT_S == 900
    html = (ROOT / "flakeid_designer" / "index.html").read_text(encoding="utf-8")
    assert ">Not here</button>" not in html and '<button data-slowskip="${escapeHtml(b.dir)}" title="Hide this message' in html
