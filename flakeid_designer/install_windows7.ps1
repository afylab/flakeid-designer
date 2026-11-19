# vdW Stack Designer on WINDOWS 7 SP1 (64-bit): everything it needs, for this user (no admin), then its desktop and
# Start-menu shortcut.  Started by "Install shortcut (Windows 7).bat" (next to this file); safe to run again.  Windows 10 /
# 11 use install_windows.ps1 and "Install shortcut (Windows).bat" (this file changes nothing there).  (2026-11-18o)
# Windows 7 has PowerShell 2.0: the folder from $MyInvocation, the hash with .NET, the Python download by the web browser.
#   1. Python: Python 3.8 already on this PC, else Python 3.8.10 (the last with a Windows 7 installer): the web browser
#      downloads it into Downloads, installed for this user only (checked against its known SHA-256 before it runs).
#   2. A Python environment of the designer's own (%USERPROFILE%\.flakeid_designer_venv): pip and the package versions the
#      Windows 7 microscope PC uses (tools\win7\requirements-py38.txt) and matplotlib; gdstk if there is one.
#   3. A designer in a folder this user cannot write (C:\Program Files) is copied to %USERPROFILE%\flakeid-designer-app
#      (update.py --writable-root), or Update... could not replace its files.
#   4. The shortcut: "FlakeIdentifier Designer" on the desktop and in the Start menu, started with that environment.
# Log: %USERPROFILE%\.flakeid_designer_install_log.txt.  For tests: FLAKEID_VENV (another environment folder),
# FLAKEID_INSTALL_NO_SHORTCUT=1 (no shortcut).
$ErrorActionPreference = 'Continue'                  # (not Stop: in PowerShell 5.1 a program's warnings on stderr, e.g. pip's,
                                                     # would end the script; every step's result is checked instead)
$ProgressPreference = 'SilentlyContinue'                      # (a progress bar makes downloads many times slower)
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072 } catch { }   # (TLS 1.2, for pip)

$PyVer = '3.8.10'
$PyUrl = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-amd64.exe"
$PySha = '7628244cb53408b50639d2c1287c659f4e29d3dfdb9084b11aed5870c0c6a48a'     # (python.org's MD5 62cf1a12... matches too)
$Ok = @('3.8')
$Here = Split-Path -Parent $MyInvocation.MyCommand.Definition   # (the flakeid_designer folder)
$Venv = if ($env:FLAKEID_VENV) { $env:FLAKEID_VENV } else { Join-Path $env:USERPROFILE '.flakeid_designer_venv' }
$Log = Join-Path $env:USERPROFILE '.flakeid_designer_install_log.txt'
try { Start-Transcript -Path $Log -Append | Out-Null } catch { }

function Say($m) { Write-Host $m }
function Fail($m) {
  Write-Host ''
  Write-Host "NOT INSTALLED: $m" -ForegroundColor Red
  Write-Host "(the log: $Log)"
  try { Stop-Transcript | Out-Null } catch { }
  exit 1
}
function Sha256($f) {
  $s = [Security.Cryptography.SHA256]::Create(); $fs = [IO.File]::OpenRead($f)
  try { return (-join ($s.ComputeHash($fs) | ForEach-Object { $_.ToString('x2') })) } finally { $fs.Close() }
}
function ByHand {
  # the Python installer saved by the web browser in Downloads, or put next to the designer
  $n = "python-$PyVer-amd64.exe"
  @((Join-Path $Here $n), (Join-Path (Split-Path $Here -Parent) $n), (Join-Path $env:USERPROFILE "Downloads\$n")) |
    Where-Object { Test-Path $_ } | Select-Object -First 1
}
function PyVersion($exe) {
  # 'major.minor' of a python.exe, or $null (missing, broken, or the Microsoft Store's stub)
  if (-not $exe -or -not (Test-Path $exe) -or $exe -match 'WindowsApps') { return $null }
  try {
    $v = & $exe -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $v) { return ($v | Select-Object -First 1).Trim() }
  } catch { }
  return $null
}
function FindPython {
  $c = New-Object System.Collections.Generic.List[string]
  foreach ($v in $Ok) {
    try {
      $p = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
      if ($LASTEXITCODE -eq 0 -and $p) { $c.Add(($p | Select-Object -First 1).Trim()) }
    } catch { }
  }
  foreach ($root in 'HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore', 'HKLM:\Software\WOW6432Node\Python\PythonCore') {
    Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
      $ip = Get-ItemProperty (Join-Path $_.PSPath 'InstallPath') -ErrorAction SilentlyContinue
      if ($ip) {
        if ($ip.ExecutablePath) { $c.Add($ip.ExecutablePath) } elseif ($ip.'(default)') { $c.Add((Join-Path $ip.'(default)' 'python.exe')) }
      }
    }
  }
  foreach ($pat in "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "$env:ProgramFiles\Python3*\python.exe", 'C:\Python3*\python.exe') {
    Get-ChildItem $pat -ErrorAction SilentlyContinue | ForEach-Object { $c.Add($_.FullName) }
  }
  Get-Command python -CommandType Application -ErrorAction SilentlyContinue | ForEach-Object { $c.Add($_.Definition) }
  $seen = @{}
  foreach ($x in $c) {
    $v = PyVersion $x
    if ($v) { $seen[(Resolve-Path $x).Path] = $v }
  }
  foreach ($want in $Ok) { foreach ($k in $seen.Keys) { if ($seen[$k] -eq $want) { return $k } } }
  if ($seen.Count) { Say ('Pythons on this PC, none of ' + ($Ok -join ' / ') + ': ' + (($seen.Keys | ForEach-Object { "$($seen[$_]) $_" }) -join '; ')) }
  return $null
}

Say 'vdW Stack Designer: installing what it needs (a few minutes the first time)'
Say "  the designer: $Here"
if (-not (Test-Path (Join-Path $Here '__main__.py'))) { Fail "this script must stay in the designer's folder (flakeid_designer), next to __main__.py" }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') { Fail 'a 64-bit Windows is needed' }
if ([Environment]::OSVersion.Version -ge (New-Object Version 6, 2)) { Fail 'this is not Windows 7: use "Install shortcut (Windows).bat"' }

# 1. Python
$py = FindPython
if ($py) {
  Say "1. Python: $py ($(PyVersion $py))"
} else {
  $inst = Join-Path $env:TEMP "python-$PyVer-amd64.exe"
  $byHand = ByHand
  if (-not $byHand) {
    Say "1. Python: none usable on this PC; opening the Python $PyVer download in the web browser: save it in Downloads"
    try { Start-Process $PyUrl } catch { }
    $null = Read-Host "   then press Enter here"
    $byHand = ByHand
    if (-not $byHand) { Fail "python-$PyVer-amd64.exe is not in Downloads: download $PyUrl with a web browser into Downloads and run this again" }
  }
  Say "1. Python: installing Python $PyVer from $byHand ..."
  Copy-Item $byHand $inst -Force
  $h = Sha256 $inst
  if ($h -ne $PySha) { Remove-Item $inst -ErrorAction SilentlyContinue; Fail "$byHand is not the expected Python installer (SHA-256 $h)" }
  Say "   installing Python $PyVer for this user (no admin needed) ..."
  $p = Start-Process -FilePath $inst -Wait -PassThru -ArgumentList @('/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1',
         'InstallLauncherAllUsers=0', 'Include_test=0', 'Include_doc=0', 'Shortcuts=0')
  Remove-Item $inst -ErrorAction SilentlyContinue
  if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { Fail "the Python installer stopped with code $($p.ExitCode) (Python $PyVer on Windows 7 needs Service Pack 1 and the updates KB2533623 and KB2999226: Windows Update)" }
  $py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python38\python.exe'
  if (-not (PyVersion $py)) { $py = FindPython }
  if (-not $py) { Fail 'Python was installed but cannot be found' }
  Say "   installed: $py"
}

# 2. the designer's own environment, pip and the packages
$vpy = Join-Path $Venv 'Scripts\python.exe'
if ((PyVersion $vpy) -and -not ($Ok -contains (PyVersion $vpy))) { Say "   the old environment's Python ($(PyVersion $vpy)) is not used any more: made again"; Remove-Item $Venv -Recurse -Force }
if (-not (PyVersion $vpy)) {
  if (Test-Path $Venv) { Remove-Item $Venv -Recurse -Force }       # (broken, e.g. its Python was removed)
  Say "2. the designer's Python environment: $Venv"
  & $py -m venv $Venv
  if ($LASTEXITCODE -ne 0 -or -not (PyVersion $vpy)) { Fail "the environment could not be made ($py -m venv)" }
} else {
  Say "2. the designer's Python environment: $Venv (there already)"
}
& $vpy -m ensurepip --upgrade 2>$null | Out-Null
Say '   pip ...'
& $vpy -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail 'pip could not be updated (is this PC online?)' }
$req = Join-Path (Split-Path $Here -Parent) 'tools\win7\requirements-py38.txt'
Say '   the packages (numpy, scipy, OpenCV, scikit-image, ...) ...'
if (Test-Path $req) { & $vpy -m pip install --disable-pip-version-check -r $req 'matplotlib<3.8' }
else { & $vpy -m pip install --disable-pip-version-check numpy==1.24.4 scipy==1.10.1 opencv-python-headless==4.8.1.78 scikit-image==0.21.0 scikit-learn==1.3.2 tifffile==2023.7.10 pillow==10.4.0 certifi 'matplotlib<3.8' }
if ($LASTEXITCODE -ne 0) { Fail 'the packages could not be installed (see above)' }
& $vpy -m pip install --disable-pip-version-check --only-binary=:all: gdstk 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Say '   (gdstk: none for Windows 7 / Python 3.8: KLayout marks files are not made on this PC)' }
& $vpy -c "import numpy, scipy, cv2, skimage, sklearn, PIL, tifffile, matplotlib, certifi; print('   packages: all import (numpy ' + numpy.__version__ + ', OpenCV ' + cv2.__version__ + ')')"
if ($LASTEXITCODE -ne 0) { Fail 'the packages were installed but do not all import (see above)' }

# 3. a folder Update... can write (2026-11-16c: a designer in C:\Program Files could not be updated)
$out = & $vpy (Join-Path $Here 'update.py') --writable-root --root (Split-Path $Here -Parent)
if ($LASTEXITCODE -ne 0 -or -not $out) { Fail "the designer's folder cannot be written and could not be copied to one that can (see above)" }
$top = ($out | Select-Object -Last 1).Trim()
if ($top -ne (Split-Path $Here -Parent)) { Say "3. the designer's folder: $top (the old one needs admin rights to change)" }
$Here = Join-Path $top 'flakeid_designer'

# 4. the shortcut, started with that environment
if ($env:FLAKEID_INSTALL_NO_SHORTCUT -eq '1') {
  Say '4. the shortcut: skipped (FLAKEID_INSTALL_NO_SHORTCUT)'
} else {
  Say '4. the shortcut ...'
  & $vpy $Here --install-shortcut
  if ($LASTEXITCODE -ne 0) { Fail 'the shortcut could not be made (see above)' }
}
Say ''
Say 'Done: start the designer from "FlakeIdentifier Designer" on the desktop or in the Start menu.'
try { Stop-Transcript | Out-Null } catch { }
exit 0
