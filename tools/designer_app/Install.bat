@echo off
rem Installs vdW Stack Designer for this user (no admin): %LOCALAPPDATA%\Programs\vdW Stack Designer,
rem with a Start-menu entry and a desktop shortcut.  Run it from the unzipped folder.
setlocal
set NAME=vdW Stack Designer
set DEST=%LOCALAPPDATA%\Programs\%NAME%
if not exist "%~dp0%NAME%\%NAME%.exe" goto :nofiles
taskkill /im "%NAME%.exe" /f >nul 2>&1
if exist "%DEST%" rmdir /s /q "%DEST%"
xcopy /e /i /q /y "%~dp0%NAME%" "%DEST%" >nul || goto :nocopy
powershell -NoProfile -Command "$s=New-Object -ComObject WScript.Shell; foreach($d in @([Environment]::GetFolderPath('Desktop'),[Environment]::GetFolderPath('Programs'))){ $l=$s.CreateShortcut((Join-Path $d '%NAME%.lnk')); $l.TargetPath='%DEST%\%NAME%.exe'; $l.WorkingDirectory='%DEST%'; $l.IconLocation='%DEST%\%NAME%.exe,0'; $l.Save() }"
echo Installed: %DEST%
echo Start it from the desktop icon or the Start menu (pin it to the taskbar from there).
pause
exit /b 0
:nofiles
echo Unzip the download first, then run Install.bat from the unzipped folder.
pause
exit /b 1
:nocopy
echo Could not copy the app to "%DEST%".
pause
exit /b 1
