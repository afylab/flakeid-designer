"""Minimal progress display: a bar with ETA on a terminal, a line every 10 % otherwise."""
from __future__ import annotations

import sys
import time


class Progress:
    def __init__(self, label: str, total: int, enabled: bool = True, stream=None):
        self.label, self.total, self.enabled = label, max(int(total), 1), enabled
        self.stream = stream or sys.stderr
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.t0 = time.time()
        self.n = 0
        self._last_pct = -10
        self._last_t = 0.0

    def update(self, n: int | None = None):
        self.n = self.n + 1 if n is None else n
        if not self.enabled:
            return
        pct = 100.0 * self.n / self.total
        el = time.time() - self.t0
        eta = el / self.n * (self.total - self.n) if self.n else 0.0
        if self.tty:
            if time.time() - self._last_t < 0.2 and self.n < self.total:
                return
            self._last_t = time.time()
            w = 30
            fill = int(round(w * self.n / self.total))
            self.stream.write(f"\r  {self.label:<22s} [{'#' * fill}{'-' * (w - fill)}] {pct:5.1f}%  {self.n}/{self.total}  "
                              f"{_fmt(el)} elapsed, {_fmt(eta)} left ")
            if self.n >= self.total:
                self.stream.write("\n")
            self.stream.flush()
        elif pct >= self._last_pct + 10 or self.n >= self.total:
            self._last_pct = pct - pct % 10
            self.stream.write(f"  {self.label}: {pct:3.0f}% ({self.n}/{self.total}), {_fmt(el)} elapsed, {_fmt(eta)} left\n")
            self.stream.flush()

    def done(self):
        if self.n < self.total:
            self.update(self.total)


def _fmt(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}" if s >= 60 else f"{s}s"
