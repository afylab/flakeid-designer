"""Desktop shortcut for the scan program (with the FlakeIdentifier icon), pinned to the taskbar on Windows 7.

    C:\\FlakeID\\run_flakeid.bat shortcut [--mm "C:\\Program Files\\Micro-Manager-2.0"] [--cfg ...\\autoscan_withir.cfg]
                                          [--no-pin]

(setup_scope_pc.bat runs it.)  The shortcut starts C:\\FlakeID\\python38\\pythonw.exe -X utf8 -s -m flakeid.scope with
the Micro-Manager folder, the hardware configuration and C:\\FlakeID\\scan_settings.json: no console window; the program
sets up what run_flakeid.bat does itself (flakeid/scope/__main__.py: launched_directly).  Pinning uses the Explorer's
"Pin to Taskbar" command, which Windows 7 offers to programs; on Windows 10 and later it is not available to programs:
right-click the desktop shortcut and choose "Pin to taskbar" there.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEST = Path(r"C:\FlakeID")
NAME = "FlakeIdentifier scan"


def vbs_str(s):
    return '"' + str(s).replace('"', '""') + '"'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm", default=r"C:\Program Files\Micro-Manager-2.0")
    ap.add_argument("--cfg", default=r"C:\Program Files\Micro-Manager-2.0\autoscan_withir.cfg")
    ap.add_argument("--settings", default=str(DEST / "scan_settings.json"))
    ap.add_argument("--no-pin", action="store_true", help="do not pin it to the taskbar")
    a = ap.parse_args(argv)
    if sys.platform != "win32":
        print("Windows only")
        return 1
    pyw = DEST / "python38" / "pythonw.exe"
    code = DEST / "FlakeIdentifier"
    icon = code / "flakeid" / "scope" / "assets" / "logo.ico"
    for p in (pyw, code, icon):
        if not p.exists():
            print(f"missing: {p}")
            return 1
    if not Path(a.cfg).is_file():
        print(f"warning: no hardware configuration at {a.cfg} (give the right one with --cfg)")
    args = f'-X utf8 -s -m flakeid.scope --mm "{a.mm}" --cfg "{a.cfg}" --settings "{a.settings}"'
    vbs = f"""
Set sh = CreateObject("WScript.Shell")
desk = sh.SpecialFolders("Desktop")
lnkPath = desk & "\\{NAME}.lnk"
Set lnk = sh.CreateShortcut(lnkPath)
lnk.TargetPath = {vbs_str(pyw)}
lnk.Arguments = {vbs_str(args)}
lnk.WorkingDirectory = {vbs_str(code)}
lnk.IconLocation = {vbs_str(str(icon) + ",0")}
lnk.Description = "Scan chips for 2D flakes (FlakeIdentifier, Young Lab)"
lnk.Save
WScript.Echo "desktop shortcut: " & lnkPath
pin = {"False" if a.no_pin else "True"}
If pin Then
  Set app = CreateObject("Shell.Application")
  Set item = app.Namespace(desk).ParseName("{NAME}.lnk")
  done = False
  For Each v In item.Verbs
    n = LCase(Replace(v.Name, "&", ""))
    If InStr(n, "taskbar") > 0 And InStr(n, "unpin") = 0 And InStr(n, "pin") > 0 Then
      v.DoIt
      done = True
      Exit For
    End If
    If InStr(n, "taskbar") > 0 And InStr(n, "unpin") > 0 Then
      WScript.Echo "already pinned to the taskbar"
      done = True
      Exit For
    End If
  Next
  If done Then
    WScript.Echo "pinned to the taskbar"
  Else
    WScript.Echo "could not pin it automatically: right-click the desktop shortcut and choose Pin to taskbar"
  End If
End If
"""
    f = Path(tempfile.gettempdir()) / "flakeid_shortcut.vbs"
    f.write_text(vbs, encoding="mbcs" if os.name == "nt" else "utf-8")
    r = subprocess.run(["cscript", "//nologo", str(f)], capture_output=True, text=True)
    print((r.stdout or "") + (r.stderr or ""))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
