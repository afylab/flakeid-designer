@echo off
rem FlakeIdentifier on the microscope PC (Windows 7 SP1, 64-bit): offline install into C:\FlakeID.
rem
rem Before running, copy the kit folder (made on the Mac by tools/win7/make_kit.sh) to the PC.  It holds:
rem   python-3.8.10-amd64.exe     the last Python installer that supports Windows 7
rem   wheels\                     numpy, scipy, opencv, scikit-image, scikit-learn, tifffile (requirements-py38.txt)
rem   FlakeIdentifier\            the code (flakeid\, tools\, profiles)
rem Then run this file from that folder (double-click, or from cmd).  Nothing is downloaded, and the old Anaconda
rem install is left alone: this Python is not added to PATH and runs only through run_flakeid.bat.

setlocal
set KIT=%~dp0
set DEST=C:\FlakeID
set PY=%DEST%\python38\python.exe

echo === FlakeIdentifier setup into %DEST%
if not exist "%DEST%" mkdir "%DEST%"

if exist "%PY%" (
  echo Python already installed at %DEST%\python38
) else (
  echo Installing Python 3.8.10 ^(per user, not on PATH^) ...
  "%KIT%python-3.8.10-amd64.exe" /quiet InstallAllUsers=0 TargetDir="%DEST%\python38" PrependPath=0 Include_launcher=0 AssociateFiles=0 Include_test=0 Shortcuts=0
  if not exist "%PY%" (
    echo.
    echo FAILED: Python did not install.  On Windows 7 the installer needs Service Pack 1 and update KB2533623
    echo ^(and the Visual C++ 2015-2022 x64 redistributable for numpy / opencv^).  Install those and run this again.
    goto :end
  )
)

rem keep Anaconda's folders out of this Python
set PATH=%DEST%\python38;%DEST%\python38\Scripts;%SystemRoot%\system32;%SystemRoot%
set PYTHONNOUSERSITE=1
set PYTHONUTF8=1
set PYTHONPATH=

echo Installing packages from the kit (offline) ...
"%PY%" -m pip install --no-index --find-links "%KIT%wheels" -r "%KIT%FlakeIdentifier\tools\win7\requirements-py38.txt" -r "%KIT%FlakeIdentifier\tools\win7\requirements-mm-py38.txt"
if errorlevel 1 (
  echo FAILED: package install.  See the messages above.
  goto :end
)

echo Copying the code to %DEST%\FlakeIdentifier ...
xcopy /E /I /Y /Q "%KIT%FlakeIdentifier" "%DEST%\FlakeIdentifier" >nul
copy /Y "%KIT%FlakeIdentifier\tools\win7\run_flakeid.bat" "%DEST%\run_flakeid.bat" >nul
rem scan settings: only if there is none yet (yours are kept)
if not exist "%DEST%\scan_settings.json" copy /Y "%KIT%FlakeIdentifier\tools\win7\scan_settings.json" "%DEST%\scan_settings.json" >nul

echo.
echo Checking the install ...
"%PY%" "%DEST%\FlakeIdentifier\tools\win7\check_env.py"

echo.
echo Desktop shortcut and taskbar ...
"%PY%" "%DEST%\FlakeIdentifier\tools\win7\make_shortcut.py"

:end
echo.
pause
endlocal
