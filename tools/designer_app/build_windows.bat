@echo off
rem Builds the Windows 10/11 app (run on a Windows 10/11 PC with Python 3.9+ from python.org):
rem   tools\designer_app\build_windows.bat           office version
rem   tools\designer_app\build_windows.bat stacker   with the live camera (IC Imaging Control 4 for the transfer
rem                                                  stations' DFK 33UP1300, OpenCV for other cameras, simulated camera)
rem -> dist_app\<name>-<version>-windows.zip: the app folder and Install.bat (per user, no admin).
setlocal
cd /d "%~dp0\..\.."
if /i "%~1"=="stacker" (
  set VENV=.venv-stacker
  set "NAME=vdW Stack Designer (Stacker)"
  set "PKGS=pywebview pyinstaller opencv-python-headless numpy scipy scikit-image gdstk imagingcontrol4"
  set "EXTRA=--hidden-import flakeid_designer.camera --hidden-import flakeid_designer.marks_to_klayout --hidden-import cv2 --hidden-import numpy --hidden-import gdstk --hidden-import imagingcontrol4 --collect-all imagingcontrol4"
) else (
  set VENV=.venv-app
  set "NAME=vdW Stack Designer"
  set "PKGS=pywebview pyinstaller opencv-python-headless numpy scipy scikit-image gdstk"
  set "EXTRA=--hidden-import flakeid_designer.camera --hidden-import flakeid_designer.marks_to_klayout --hidden-import gdstk --exclude-module imagingcontrol4"
)
rem the builder of runs' designer files (tools\make_designer_bundle.py and the flakeid modules it uses, with its calibrations)
set "BUILDER=--paths "%CD%\tools" --hidden-import make_designer_bundle --hidden-import make_catalog_100x --hidden-import make_afm_set --hidden-import make_catalog --hidden-import make_catalog_live --hidden-import make_stack_bundle --hidden-import terraces100 --hidden-import flakeid.background --hidden-import flakeid.branch --hidden-import flakeid.flakes --hidden-import flakeid.graphite --hidden-import flakeid.hundred --hidden-import flakeid.io --hidden-import flakeid.mm --hidden-import flakeid.optics --hidden-import flakeid.outline --hidden-import flakeid.pipeline --hidden-import flakeid.scorer --hidden-import flakeid.segment --hidden-import flakeid.targets --hidden-import flakeid.thickness --hidden-import flakeid.screen100 --hidden-import flakeid.progress --hidden-import flakeid.afm --exclude-module sklearn --add-data "%CD%\flakeid\profiles;flakeid\profiles" --exclude-module matplotlib --exclude-module flakeid.scope"
rem Python: 3.12 (else 3.11 / 3.13).  Not the newest one: the app built with Python 3.14 (2026-10-20) froze its window
rem while building and its scikit-learn did not load; the window's libraries (pywebview, pythonnet) lag new Pythons.
rem PYTHON=<path to python.exe> before running this: that Python.  Else 3.12 / 3.11 / 3.13 from the py launcher, else
rem from the usual install folders (a launcher-less install, or a window opened before Python was installed).
set "PYEXE="
for /f "usebackq delims=" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0find_python.ps1"`) do set "PYEXE=%%p"
rem (no PowerShell answer: the py launcher, as before)
if not defined PYEXE if defined PYTHON set "PYEXE=%PYTHON%"
for %%v in (3.12 3.11 3.13) do if not defined PYEXE for /f "usebackq delims=" %%p in (`py -%%v -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%p"
if not defined PYEXE (
  echo.
  echo Python 3.12 or 3.13 was not found ^(the Pythons it saw are listed above^).
  echo Install Python 3.13 from https://www.python.org/downloads/windows/ ^(64-bit^), open a NEW command window and run this again,
  echo or give its path:  set PYTHON=C:\path\to\python.exe   then run this again.
  goto :fail
)
set "PYV="
for /f "usebackq delims=" %%v in (`call "%PYEXE%" -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))"`) do set "PYV=%%v"
if not defined PYV (echo "%PYEXE%" does not run & goto :fail)
echo Building with Python %PYV% ^(%PYEXE%^)
if "%PYV%"=="3.14" echo WARNING: Python 3.14 is what made the app freeze ^(2026-10-20^); 3.12 or 3.13 is better.
rem an environment made with another Python (e.g. 3.14 before 2026-10-20): made again
if exist %VENV%\Scripts\python.exe (
  %VENV%\Scripts\python -c "import sys; sys.exit(0 if '{}.{}'.format(*sys.version_info[:2]) == '%PYV%' else 1)" || (echo The build environment %VENV% uses another Python: making it again & rmdir /s /q %VENV%)
)
if not exist %VENV%\Scripts\python.exe (
  "%PYEXE%" -m venv %VENV% || goto :fail
  %VENV%\Scripts\python -m pip install --no-cache-dir -q --upgrade pip
)
rem (every build: packages added since the environment was made)
%VENV%\Scripts\python -m pip install --no-cache-dir -q %PKGS% || (echo Installing the packages failed: is the PC online? & goto :fail)
rem the versions this build uses (for the log: send this window if something goes wrong)
%VENV%\Scripts\python -c "import sys, importlib.metadata as m; print('Python', sys.version.split()[0]); print(', '.join(p + ' ' + m.version(p) for p in ('pywebview', 'pyinstaller', 'numpy', 'scipy', 'scikit-image', 'opencv-python-headless') if __import__('importlib.util').util.find_spec(p.replace('-', '_').replace('opencv_python_headless', 'cv2').replace('scikit_image', 'skimage').replace('pyinstaller', 'PyInstaller')) is not None))"
if exist build_app rmdir /s /q build_app
if exist "dist_app\%NAME%" rmdir /s /q "dist_app\%NAME%"
%VENV%\Scripts\python -m PyInstaller --noconfirm --windowed --name "%NAME%" ^
  --icon "%CD%\flakeid_designer\static\designer_win.ico" ^
  --add-data "%CD%\flakeid_designer\index.html;flakeid_designer" ^
  --add-data "%CD%\flakeid_designer\static;flakeid_designer\static" ^
  --add-data "%CD%\flakeid_designer\README.txt;flakeid_designer" ^
  --hidden-import flakeid_designer.__main__ --hidden-import flakeid_designer.stacks --hidden-import flakeid_designer.manual --hidden-import flakeid_designer.afmplan --hidden-import flakeid_designer.stackdir --hidden-import flakeid_designer.kept %EXTRA% %BUILDER% --paths "%CD%" ^
  --workpath build_app --distpath dist_app --specpath build_app ^
  "%CD%\flakeid_designer\app.py" || goto :fail
rem the libraries' test folders: never used, and their data has the longest paths ("path too long" when the zip is
rem unzipped in a deep folder, 2026-10-20)
for /d /r "dist_app\%NAME%" %%d in (tests) do if exist "%%d\" rmdir /s /q "%%d"
rem the packaged builder must load (2026-10-20: a newer missed compiled modules: KeyError '__reduce_cython__')
start "" /wait "dist_app\%NAME%\%NAME%.exe" --check-builder
if errorlevel 1 (echo. & echo The built app cannot build runs: its builder does not load. Send the folder build_app and this window. & goto :fail)
for /f "tokens=3 delims= " %%v in ('findstr /b "VERSION = " flakeid_designer\__main__.py') do set VER=%%~v
if not exist dist_app\pkg mkdir dist_app\pkg
copy /y tools\designer_app\Install.bat dist_app\pkg\Install.bat >nul
powershell -NoProfile -Command "(Get-Content 'dist_app\pkg\Install.bat') -replace '^set NAME=.*$', 'set NAME=%NAME%' | Set-Content 'dist_app\pkg\Install.bat'"
rem the zip: Python's zipfile (PowerShell's Compress-Archive breaks on files dated before 1980, e.g. in OpenCV)
%VENV%\Scripts\python tools\designer_app\make_zip.py "dist_app\%NAME%-%VER%-windows.zip" "dist_app\%NAME%" "dist_app\pkg\Install.bat" || goto :fail
rmdir /s /q dist_app\pkg
echo built: dist_app\%NAME%-%VER%-windows.zip
exit /b 0
:fail
echo the build failed
exit /b 1
