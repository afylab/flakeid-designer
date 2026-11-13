"""Find out how the Olympus U-CBS (coded nosepiece box) talks over RS-232: read-only probe.

    C:\\FlakeID\\run_flakeid.bat ucbs                    list the serial ports
    C:\\FlakeID\\run_flakeid.bat ucbs --port COM1        probe that port (close Olympus Stream first)

The U-CBS reads the coded nosepiece's position and sends it to Stream / an Olympus camera; its protocol is not in the
public manual.  Other Olympus boxes of the same generation (BX-UCB, IX2-UCB) use short text commands ("1LOG IN",
"1OB?" = objective position) at 19200 baud, even parity, 2 stop bits.  This tool, for each likely serial setting:
listens for data the box sends by itself, then sends the standard Olympus queries and records every reply.  Then it
watches the port for WATCH_S seconds while you turn the nosepiece through its positions, polling with the query that
answered (if any) and printing everything that arrives.  The U-CBS only reads encoders, so nothing can move; still,
give the U-CBS's port, never the stage controller's.  Everything goes to ucbs_probe.json: send it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

SETTINGS = [(19200, "E", 2), (19200, "N", 1), (9600, "N", 1), (9600, "E", 2), (4800, "N", 1), (38400, "N", 1), (115200, "N", 1)]
QUERIES = ["1LOG IN", "1OB?", "OB?", "1UNIT?", "1VER?", "VER?", "U?", "?"]
EOLS = ["\r\n", "\r"]
WATCH_S = 40.0


def show(b: bytes) -> str:
    return f"{b!r} [{b.hex(' ')}]" if b else "(nothing)"


def read_all(ser, t):
    t_end, out = time.time() + t, b""
    while time.time() < t_end:
        n = ser.in_waiting
        if n:
            out += ser.read(n)
            t_end = max(t_end, time.time() + 0.15)          # keep reading while bytes keep coming
        else:
            time.sleep(0.02)
    return out


def main(argv=None):
    import serial
    from serial.tools import list_ports
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="the U-CBS's serial port, e.g. COM1 (without: list the ports)")
    ap.add_argument("--watch", type=float, default=WATCH_S, help=f"seconds to watch while you turn the nosepiece (default {WATCH_S:g})")
    ap.add_argument("--out", default="ucbs_probe.json")
    a = ap.parse_args(argv)
    ports = [(p.device, p.description, p.hwid) for p in list_ports.comports()]
    print("serial ports on this PC:")
    for d, desc, hw in ports:
        print(f"  {d}: {desc}  ({hw})")
    if not a.port:
        print("\nrun again with --port <the U-CBS's port> (Device Manager > Ports shows which is which; a USB adapter "
              "or the stage controller is not the U-CBS's built-in RS-232 port)")
        return 0
    rep = {"port": a.port, "ports": ports, "tries": [], "watch": []}
    working = None
    print(f"\nprobing {a.port} (close Olympus Stream first: only one program can open the port)")
    for baud, par, stop in SETTINGS:
        try:
            ser = serial.Serial(a.port, baud, bytesize=8, parity=par, stopbits=stop, timeout=0.2, write_timeout=1.0)
        except serial.SerialException as e:
            print(f"  cannot open {a.port}: {e}")
            rep["error"] = str(e)
            break
        tag = f"{baud} 8{par}{stop}"
        unsolicited = read_all(ser, 1.5)
        print(f"  {tag}: listening 1.5 s -> {show(unsolicited)}")
        t = {"settings": tag, "unsolicited": unsolicited.hex(), "replies": []}
        for eol in EOLS:
            for q in QUERIES:
                ser.reset_input_buffer()
                ser.write((q + eol).encode("ascii"))
                r = read_all(ser, 0.6)
                t["replies"].append({"query": q, "eol": eol.encode().hex(), "reply": r.hex(), "text": r.decode("ascii", "replace")})
                if r:
                    print(f"    {q!r:10} -> {show(r)}")
                    if working is None and q != "1LOG IN":
                        working = (baud, par, stop, q, eol)
        if not any(x["reply"] for x in t["replies"]):
            print("    (no replies)")
        rep["tries"].append(t)
        if any(x["reply"] for x in t["replies"]):
            try:
                ser.write(("1LOG OUT" + (working[4] if working else "\r\n")).encode())
            except Exception:
                pass
            ser.close()
            break                                            # these settings answer: no need to try others
        ser.close()
    # watch: turn the nosepiece
    if "error" not in rep:
        baud, par, stop, q, eol = working if working else (19200, "E", 2, None, "\r\n")
        print(f"\nnow turn the nosepiece slowly through all its positions (stopping a few seconds at each) for "
              f"{a.watch:g} s; settings {baud} 8{par}{stop}" + (f", polling with {q!r}" if q else ", listening only"))
        ser = serial.Serial(a.port, baud, bytesize=8, parity=par, stopbits=stop, timeout=0.2)
        if q and q.startswith("1"):
            ser.write(("1LOG IN" + eol).encode()); read_all(ser, 0.5)
        t0, last = time.time(), None
        while time.time() - t0 < a.watch:
            if q:
                ser.write((q + eol).encode())
            r = read_all(ser, 0.5)
            if r and r != last:
                print(f"  {time.time() - t0:5.1f} s: {show(r)}")
                last = r
            rep["watch"].append({"t": round(time.time() - t0, 2), "data": r.hex()})
        if q and q.startswith("1"):
            ser.write(("1LOG OUT" + eol).encode())
        ser.close()
    rep["working"] = list(working) if working else None
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nreport: {a.out}" + ("" if working else "  (no query answered; if the watch showed nothing either, the box "
                                                      "may need Stream's own start-up sequence: we can capture it with a "
                                                      "serial monitor)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
