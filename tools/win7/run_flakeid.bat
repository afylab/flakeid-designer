@echo off
rem Run a FlakeIdentifier command with the isolated Python 3.8 in C:\FlakeID (no Anaconda on the path).
rem
rem   run_flakeid.bat live D:\Scans\multitest --flat C:\FlakeID\flats\flat_10x_3ms.npz -o D:\FlakeResults\multitest --workers 2
rem   run_flakeid.bat hundred D:\Scans\multitest\chip_1_100x --viewer D:\FlakeResults\multitest\chip_1\chip_1_tiles_viewer --flat ...
rem   run_flakeid.bat check            (the install check)
rem   run_flakeid.bat update [--check] (the newest version from GitHub: flakeid/scope/update.py)
rem   run_flakeid.bat shortcut         (desktop shortcut with the icon, pinned to the taskbar)
rem   run_flakeid.bat capture C:\FlakeScans\pleasework_1_targets.csv --flake 11   (raw 100x images of one flake)
rem   run_flakeid.bat screenshots      (pictures of the scan program's windows for docs/scan_guide.md; simulated, no hardware)
rem   run_flakeid.bat objtest [--port COM1]   (does the objective window close by itself when the objective is changed?)
rem   run_flakeid.bat ucbs [--port COM1]   (probe the Olympus U-CBS coded nosepiece box)
rem   run_flakeid.bat mmtest --mm "C:\Program Files\Micro-Manager-2.0" --cfg "C:\Program Files\Micro-Manager-2.0\autoscan_withir.cfg"
rem   run_flakeid.bat irtest [--live 60] [--exposure MS] [--camera NAME]   (the IR camera: properties, frames, report)
rem
rem The first word is the flakeid module (live, hundred, mm, ...); the rest is passed on unchanged.

setlocal
set DEST=C:\FlakeID
rem the original PATH (without its Anaconda entries, which mm_python_test.py drops) is still needed to find
rem hardware libraries such as the XIMEA API that Micro-Manager's device drivers load
set FLAKEID_ORIG_PATH=%PATH%
set PATH=%DEST%\python38;%DEST%\python38\Scripts;%SystemRoot%\system32;%SystemRoot%
set PYTHONNOUSERSITE=1
rem text files as UTF-8, as on the analysis Mac (Windows defaults to cp1252: the viewer page has characters it lacks)
set PYTHONUTF8=1
set PYTHONPATH=%DEST%\FlakeIdentifier
cd /d "%DEST%\FlakeIdentifier"
if /i "%~1"=="capture" (
  rem raw 100x images of one flake for developing the processing (the arguments after "capture")
  "%DEST%\python38\python.exe" tools\win7\capture_flake.py %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="shortcut" (
  rem desktop shortcut with the icon, pinned to the taskbar (the arguments after "shortcut")
  "%DEST%\python38\python.exe" tools\win7\make_shortcut.py %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="screenshots" (
  rem pictures of every scan-program window that needs input, on the simulated microscope (no hardware used)
  "%DEST%\python38\python.exe" tools\win7\dialog_screenshots.py %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="objtest" (
  rem does the objective window close by itself? (nosepiece reading test; the arguments after "objtest")
  "%DEST%\python38\python.exe" tools\win7\objective_test.py %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="update" (
  rem the newest version from GitHub (as the start screen's "Update..."; --check: only what is new)
  "%DEST%\python38\python.exe" -m flakeid.scope.update %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="check" (
  "%DEST%\python38\python.exe" tools\win7\check_env.py
) else if /i "%~1"=="ucbs" (
  rem (the arguments after "ucbs")
  "%DEST%\python38\python.exe" tools\win7\ucbs_probe.py %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="mmtest" (
  rem (the arguments after "mmtest")
  "%DEST%\python38\python.exe" tools\win7\mm_python_test.py %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="irtest" (
  rem the IR (stacking-domain) camera: properties, frames, report.json (the arguments after "irtest")
  "%DEST%\python38\python.exe" tools\win7\ir_camera_test.py %2 %3 %4 %5 %6 %7 %8 %9
) else (
  "%DEST%\python38\python.exe" -m flakeid.%*
)
endlocal
