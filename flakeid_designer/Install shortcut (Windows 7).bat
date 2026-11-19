@echo off
rem vdW Stack Designer on a Windows 7 SP1 PC: Python 3.8 (the web browser downloads it, if this PC has none), the packages,
rem then the desktop and Start-menu shortcut.  For this user only, no admin needed; safe to run again.  The steps:
rem install_windows7.ps1.  Windows 10 / 11: "Install shortcut (Windows).bat" instead.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows7.ps1"
pause
