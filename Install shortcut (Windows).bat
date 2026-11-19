@echo off
rem vdW Stack Designer on this PC: Python (from python.org, if this PC has none), the packages, then the desktop and
rem Start-menu shortcut.  For this user only, no admin needed; safe to run again.  The steps:
rem flakeid_designer\install_windows.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flakeid_designer\install_windows.ps1"
pause
