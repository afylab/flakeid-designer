"""The coded nosepiece (Olympus U-D7RES) through the U-CBS box on a serial port: which objective is in.

Protocol (found with tools/win7/ucbs_probe.py on the microscope PC, 2026-10-03): 19200 baud, 8 data bits, even parity,
2 stop bits, commands ended by CR LF.  "1LOG IN" -> "1LOG +"; "1OB?" -> "1OB n" (position 1..7) or "1OB X" (between
positions); "1UNIT?" -> "1UNIT SCU,U-D7RES".  After logging in the box also announces every change by itself
("1OBM n", "1OBM X"); those lines are skipped when a query is answered.
"""
from __future__ import annotations

import threading
import time


class Nosepiece:
    def __init__(self, port, baud=19200):
        import serial
        self.ser = serial.Serial(port, baud, bytesize=8, parity="E", stopbits=2, timeout=0.3, write_timeout=1.0)
        self.lock = threading.Lock()
        self.port = port
        r = self._cmd("1LOG IN", "1LOG")
        if not r.startswith("1LOG +"):
            self.ser.close()
            raise RuntimeError(f"no answer from the U-CBS on {port} (got {r!r}); is Olympus Stream still running?")
        self.unit = self._cmd("1UNIT?", "1UNIT")[6:].strip()

    def _cmd(self, q, expect, timeout=1.0):
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write((q + "\r\n").encode("ascii"))
            t_end, buf = time.time() + timeout, b""
            while time.time() < t_end:
                buf += self.ser.read(self.ser.in_waiting or 1)
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    s = line.decode("ascii", "replace").strip()
                    if s.startswith(expect) and not s.startswith("1OBM"):
                        return s
            return ""

    def position(self):
        """Objective position 1..7, or None between positions / no answer."""
        r = self._cmd("1OB?", "1OB ")
        try:
            return int(r.split()[1])
        except (IndexError, ValueError):
            return None

    def close(self):
        if not self.ser.is_open:
            return
        try:
            self._cmd("1LOG OUT", "1LOG", timeout=0.5)
        except Exception:
            pass
        finally:
            self.ser.close()


class MockNosepiece:
    """For the simulated microscope: the position follows MockCore's objective."""

    def __init__(self, core, positions):
        self.core, self.positions, self.unit, self.port = core, positions, "SCU,U-D7RES (simulated)", "mock"

    def position(self):
        o = self.core.objective
        name = "domain" if o == getattr(self.core, "DOMAIN", None) else f"{o:g}x"      # (5x, 10x, 50x, 100x)
        return self.positions.get(name) or None

    def close(self):
        pass
