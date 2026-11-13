"""Preview of the scan program's windows in a newer look, to choose a style before the real windows are changed.

    C:\\FlakeID\\run_flakeid.bat scope.theme_demo          (microscope PC)
    python -m flakeid.scope.theme_demo                    (any computer with sv-ttk installed)

Left: the current look (Tk's classic widgets, as the scan program uses now).  Right: the same windows built from ttk
widgets, in the theme chosen at the top: Windows native (on Windows 7 the "vista" theme), Sun Valley light or Sun
Valley dark (the sv-ttk package).  Nothing here talks to the microscope.
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

SAMPLE_LOG = """==== piece 1 of 2: pleasework_1, centre 25772.1, 29430.8 um, Z 10332.0, hbn on 285 nm oxide
  reference substrate R/G 0.535, B/G 0.937, G 961.5 counts; substrate fraction of the start frame 0.746
  chip about 25.5 x 18.0 mm, rotation 1.3 deg
  scan: 520 tiles (grid 26 x 20, overlap 15 %), 1025 frames used to find the edges
  focus point 1: 63574.4, 36329.4 -> Z 10340.3
  focus surface: quadratic through 9 points, rms 0.191 um (plane 0.720 um)
  scanning 520 tiles into C:\\FlakeScans\\pleasework_1_tiles (hbn on 285.0 nm) ...
    250 / 520 tiles, 217.6 s, about 235.0 s left
    [analysis] [18:32:32 +  261s] oxide fixed at 306.6 nm (pooled samples of 3 block(s), gain 0.97)
    [analysis] [18:43:04 +  893s] provisional target list after 31 blocks: 30 flakes
"""

PIECES = ("Move the stage to the approximate centre of piece 3 (10x) and focus roughly, then press \"Add piece 3 here\".\n\n"
          "Pieces so far:\n  piece 1: 25772.1, 29430.8 um, Z 10332.0, hbn on 285 nm\n"
          "  piece 2: 55097.9, 28345.9 um, Z 10349.2, graphite on 90 nm")


def dpi_aware():
    """Crisp text on high-resolution screens (Windows scales unaware programs up, blurring them)."""
    if sys.platform == "win32":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)         # Windows 8.1 and later
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()              # Windows 7
        except Exception:
            pass


def live_placeholder(parent, w=300, h=200, bg="#2b2350"):
    """Stand-in for the live camera view (the real one is a picture drawn by the program, not a themed widget)."""
    c = tk.Canvas(parent, width=w, height=h, bg=bg, highlightthickness=0)
    c.create_polygon(90, 120, 120, 70, 175, 62, 205, 105, 180, 150, 115, 158, fill="#4fc3b8", outline="")
    c.create_polygon(122, 112, 140, 85, 170, 83, 182, 108, 165, 132, 128, 133, fill="#f2c14e", outline="")
    c.create_line(w // 2, h // 2 - 12, w // 2, h // 2 + 12, fill="white")
    c.create_line(w // 2 - 12, h // 2, w // 2 + 12, h // 2, fill="white")
    c.create_text(8, h - 8, text="live view  12.4 fps  Z 10332.0 um  objective 10x", anchor="sw", fill="#d8d4f0",
                  font=("Consolas", 8))
    return c


# ---------------------------------------------------------------------------------------------- the current look
def classic(parent):
    f = tk.LabelFrame(parent, text=" Now: classic Tk widgets ", padx=8, pady=8)
    # run window
    top = tk.Frame(f)
    top.pack(fill="x")
    tk.Label(top, text="pleasework_1: tile 250 / 520", anchor="w", font=("Segoe UI", 11, "bold")).pack(side="left", fill="x", expand=True)
    tk.Button(top, text="Stop the scan").pack(side="right")
    t = tk.Text(f, width=60, height=9, font=("Consolas", 9))
    t.insert("end", SAMPLE_LOG)
    t.pack(fill="both", expand=True, pady=(4, 8))
    # add pieces
    row = tk.Frame(f)
    row.pack(fill="x")
    live_placeholder(row, 260, 170).pack(side="left")
    side = tk.Frame(row)
    side.pack(side="left", padx=8, anchor="n")
    tk.Label(side, text=PIECES, justify="left", anchor="nw", wraplength=230).pack(anchor="w")
    for txt in ("Add piece 3 here", "Done (2 pieces)", "Cancel"):
        tk.Button(side, text=txt, width=22).pack(anchor="w", pady=2)
    # material and oxide
    d = tk.Frame(f, bd=1, relief="groove", padx=8, pady=8)
    d.pack(fill="x", pady=(8, 0))
    tk.Label(d, text="Piece 3 at 41210.4, 30022.1 um").grid(row=0, column=0, columnspan=2, sticky="w")
    tk.Label(d, text="Material").grid(row=1, column=0, sticky="w")
    tk.OptionMenu(d, tk.StringVar(value="hbn"), "hbn", "graphite").grid(row=1, column=1, sticky="w")
    tk.Label(d, text="Oxide (nm)").grid(row=2, column=0, sticky="w")
    e = tk.Entry(d, width=8)
    e.insert(0, "285")
    e.grid(row=2, column=1, sticky="w")
    tk.Checkbutton(d, text="Image rhombohedral domains").grid(row=3, column=0, columnspan=2, sticky="w")
    b = tk.Frame(d)
    b.grid(row=4, column=0, columnspan=2, sticky="e", pady=(6, 0))
    tk.Button(b, text="OK", width=8).pack(side="left", padx=3)
    tk.Button(b, text="Cancel", width=8).pack(side="left")
    return f


# ---------------------------------------------------------------------------------------------- the ttk version
def themed(parent, texts):
    f = ttk.LabelFrame(parent, text=" Proposed: ttk widgets ", padding=10)
    top = ttk.Frame(f)
    top.pack(fill="x")
    ttk.Label(top, text="pleasework_1: tile 250 / 520", style="Status.TLabel").pack(side="left", fill="x", expand=True)
    ttk.Button(top, text="Stop the scan").pack(side="right")
    pb = ttk.Progressbar(f, maximum=520, value=250)
    pb.pack(fill="x", pady=(6, 4))
    t = tk.Text(f, width=60, height=9, font=("Consolas", 9), relief="flat", borderwidth=0, padx=6, pady=4)
    t.insert("end", SAMPLE_LOG)
    t.pack(fill="both", expand=True, pady=(4, 8))
    texts.append(t)
    row = ttk.Frame(f)
    row.pack(fill="x")
    live_placeholder(row, 260, 170).pack(side="left")
    side = ttk.Frame(row)
    side.pack(side="left", padx=10, anchor="n")
    ttk.Label(side, text=PIECES, justify="left", anchor="nw", wraplength=230).pack(anchor="w", pady=(0, 6))
    ttk.Button(side, text="Add piece 3 here", style="Accent.TButton", width=22).pack(anchor="w", pady=2)
    ttk.Button(side, text="Done (2 pieces)", width=22).pack(anchor="w", pady=2)
    ttk.Button(side, text="Cancel", width=22).pack(anchor="w", pady=2)
    d = ttk.LabelFrame(f, text=" Piece 3: material and oxide ", padding=10)
    d.pack(fill="x", pady=(10, 0))
    ttk.Label(d, text="Piece 3 at 41210.4, 30022.1 um").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
    ttk.Label(d, text="Material").grid(row=1, column=0, sticky="w", padx=(0, 10))
    cb = ttk.Combobox(d, values=("hbn", "graphite"), width=12, state="readonly")
    cb.set("hbn")
    cb.grid(row=1, column=1, sticky="w", pady=2)
    ttk.Label(d, text="Oxide (nm)").grid(row=2, column=0, sticky="w", padx=(0, 10))
    e = ttk.Entry(d, width=8)
    e.insert(0, "285")
    e.grid(row=2, column=1, sticky="w", pady=2)
    v = tk.BooleanVar(value=True)
    ttk.Checkbutton(d, text="Image rhombohedral domains", variable=v, style="Switch.TCheckbutton").grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
    b = ttk.Frame(d)
    b.grid(row=4, column=0, columnspan=2, sticky="e", pady=(8, 0))
    ttk.Button(b, text="OK", style="Accent.TButton", width=8).pack(side="left", padx=4)
    ttk.Button(b, text="Cancel", width=8).pack(side="left")
    return f


def main():
    dpi_aware()
    from .ui import set_icon
    root = tk.Tk()
    root._icons = set_icon(root)
    root.title("FlakeIdentifier: window style preview")
    from .ui import use_sun_valley
    sv_ttk = True
    style = ttk.Style(root)
    native = "vista" if "vista" in style.theme_names() else ("aqua" if "aqua" in style.theme_names() else "clam")
    choice = tk.StringVar(value="dark")
    bar = ttk.Frame(root, padding=(10, 8))
    bar.pack(fill="x")
    ttk.Label(bar, text="Right-hand side:").pack(side="left", padx=(0, 8))
    rb = []
    for val, txt in (("native", f"Windows native ({native})"), ("light", "Sun Valley light"), ("dark", "Sun Valley dark")):
        r = ttk.Radiobutton(bar, text=txt, value=val, variable=choice)
        r.pack(side="left", padx=6)
        rb.append((val, r))
    if sv_ttk is None:
        for val, r in rb[1:]:
            r.state(["disabled"])
        ttk.Label(bar, text="(sv-ttk not installed)").pack(side="left", padx=8)
    body = tk.Frame(root)
    body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    left = classic(body)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    texts = []
    right = themed(body, texts)
    right.grid(row=0, column=1, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    def apply(*_):
        c = choice.get()
        if c == "native" or sv_ttk is None:
            style.theme_use(native)
            fg, bg = "black", "white"
        else:
            use_sun_valley(root, c)
            fg, bg = ("#1c1c1c", "#fafafa") if c == "light" else ("#e8e8e8", "#1c1c1c")
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"))
        for t in texts:
            t.configure(fg=fg, bg=bg, insertbackground=fg)
        body.configure(bg=style.lookup("TFrame", "background") or root.cget("bg"))
    choice.trace_add("write", apply)
    apply()
    root.mainloop()


if __name__ == "__main__":
    main()
