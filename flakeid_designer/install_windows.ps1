# vdW Stack Designer: everything it needs on a fresh Windows 10 / 11 or Windows 7 SP1, for this user (no admin), then its
# desktop and Start-menu shortcut.  Started by "Install shortcut (Windows).bat" (next to this file); safe to run again (it then only
# checks, updates the packages and makes the shortcut again).
#   1. Python: one already on this PC (3.10 - 3.13, not the Microsoft Store stub), else Python 3.12.10 from python.org,
#      installed for this user only (checked against its known SHA-256 before it runs).
#   2. A Python environment of the designer's own (%USERPROFILE%\.flakeid_designer_venv): pip and the packages the
#      designer's builds and tools need (requirements.txt beside the designer, and certifi for Update...).
#   3. A designer in a folder this user cannot write (C:\Program Files) is copied to %USERPROFILE%\flakeid-designer-app
#      (update.py --writable-root), or Update... could not replace its files.
#   4. The shortcut: "FlakeIdentifier Designer" on the desktop and in the Start menu, started with that environment.
# Windows 7 (2026-11-18k): PowerShell 2.0 (no $PSScriptRoot, Invoke-WebRequest, Get-FileHash), Python 3.8.10 (the last
#   with a Windows 7 installer) and the package versions the Windows 7 microscope PC uses (tools\win7\requirements-py38.txt).
#   When this PC cannot download from python.org itself (old TLS), python-3.8.10-amd64.exe downloaded by hand into this
#   folder or Downloads is used (checked against its SHA-256 the same way).
# Log: %USERPROFILE%\.flakeid_designer_install_log.txt.  For tests: FLAKEID_VENV (another environment folder),
# FLAKEID_INSTALL_NO_SHORTCUT=1 (no shortcut).
$ErrorActionPreference = 'Continue'                  # (not Stop: in PowerShell 5.1 a program's warnings on stderr, e.g. pip's,
                                                     # would end the script; every step's result is checked instead)
$ProgressPreference = 'SilentlyContinue'                      # (a progress bar makes downloads many times slower)
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072 } catch { }   # (TLS 1.2)

$Win7 = [Environment]::OSVersion.Version -lt (New-Object Version 6, 2)   # (Windows 7 = 6.1; 8 = 6.2)
if ($Win7) {
  $PyVer = '3.8.10'
  $PySha = '7628244cb53408b50639d2c1287c659f4e29d3dfdb9084b11aed5870c0c6a48a'
  $Ok = @('3.8')
} else {
  $PyVer = '3.12.10'
  $PySha = '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
  $Ok = @('3.12', '3.13', '3.11', '3.10')                       # (in this order of preference)
}
$PyExe = "python-$PyVer-amd64.exe"
$PyUrl = "https://www.python.org/ftp/python/$PyVer/$PyExe"
$Here = $PSScriptRoot                                           # (the flakeid_designer folder)
if (-not $Here) { $Here = Split-Path -Parent $MyInvocation.MyCommand.Definition }   # (PowerShell 2.0: Windows 7)
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
  # (Get-FileHash needs PowerShell 4)
  $s = [Security.Cryptography.SHA256]::Create(); $fs = [IO.File]::OpenRead($f)
  try { return (-join ($s.ComputeHash($fs) | ForEach-Object { $_.ToString('x2') })) } finally { $fs.Close() }
}
function Download($url, $out) {
  # (Invoke-WebRequest needs PowerShell 3; Windows 7's .NET may not speak TLS 1.2: then BITS, then certutil)
  # (each try starts with no file: a broken try's part file is not taken as the download)
  $e = $null
  Remove-Item $out -ErrorAction SilentlyContinue
  try { (New-Object Net.WebClient).DownloadFile($url, $out); if (Test-Path $out) { return $null } } catch { $e = $_.Exception.Message }
  Remove-Item $out -ErrorAction SilentlyContinue
  try { & bitsadmin /transfer flakeid_designer /download /priority foreground $url $out | Out-Null; if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) { return $null } } catch { }
  Remove-Item $out -ErrorAction SilentlyContinue
  try { & certutil -urlcache -split -f $url $out | Out-Null; if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) { return $null } } catch { }
  Remove-Item $out -ErrorAction SilentlyContinue
  return $e
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
if ($Win7) { Say "  Windows 7: Python $PyVer and the Windows 7 package versions" }

# 1. Python
$py = FindPython
if ($py) {
  Say "1. Python: $py ($(PyVersion $py))"
} else {
  $inst = Join-Path $env:TEMP $PyExe
  $byHand = @((Join-Path $Here $PyExe), (Join-Path (Split-Path $Here -Parent) $PyExe), (Join-Path $env:USERPROFILE "Downloads\$PyExe")) |
            Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($byHand) {
    Say "1. Python: none usable on this PC; installing Python $PyVer from $byHand ..."
    Copy-Item $byHand $inst -Force
  } else {
    Say "1. Python: none usable on this PC; downloading Python $PyVer from python.org ..."
    Remove-Item $inst -ErrorAction SilentlyContinue
    $e = Download $PyUrl $inst
    if (-not (Test-Path $inst)) {
      Fail ("the download failed ($e). Is this PC online? If it is, download $PyUrl with a web browser into this " +
            "folder or Downloads, and run this again.")
    }
  }
  $h = Sha256 $inst
  if ($h -ne $PySha) { Remove-Item $inst -ErrorAction SilentlyContinue; Fail "the Python installer is not the expected file (SHA-256 $h)" }
  Say "   installing Python $PyVer for this user (no admin needed) ..."
  $p = Start-Process -FilePath $inst -Wait -PassThru -ArgumentList @('/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1',
         'InstallLauncherAllUsers=0', 'Include_test=0', 'Include_doc=0', 'Shortcuts=0')
  Remove-Item $inst -ErrorAction SilentlyContinue
  if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) {
    if ($Win7) { Fail "the Python installer stopped with code $($p.ExitCode) (Python $PyVer on Windows 7 needs Service Pack 1 and the updates KB2533623 and KB2999226: Windows Update)" }
    Fail "the Python installer stopped with code $($p.ExitCode)"
  }
  $py = Join-Path $env:LOCALAPPDATA ('Programs\Python\Python' + ($PyVer.Split('.')[0..1] -join '') + '\python.exe')
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
Say '   the packages (numpy, scipy, OpenCV, scikit-image, ...) ...'
if ($Win7) {
  # (the versions the Windows 7 microscope PC runs: the newest with Python 3.8 wheels that work on Windows 7; matplotlib
  # and gdstk are only for some tools: gdstk not installed is said, not a failure)
  $req = Join-Path (Split-Path $Here -Parent) 'tools\win7\requirements-py38.txt'
  if (Test-Path $req) { & $vpy -m pip install --disable-pip-version-check -r $req 'matplotlib<3.8' }
  else { & $vpy -m pip install --disable-pip-version-check numpy==1.24.4 scipy==1.10.1 opencv-python-headless==4.8.1.78 scikit-image==0.21.0 scikit-learn==1.3.2 tifffile==2023.7.10 pillow==10.4.0 certifi 'matplotlib<3.8' }
  if ($LASTEXITCODE -ne 0) { Fail 'the packages could not be installed (see above)' }
  & $vpy -m pip install --disable-pip-version-check --only-binary=:all: gdstk 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) { Say '   (gdstk: none for Windows 7 / Python 3.8: KLayout marks files are not made on this PC)' }
  $imp = 'numpy, scipy, cv2, skimage, sklearn, PIL, tifffile, matplotlib, certifi'
} else {
  $req = Join-Path (Split-Path $Here -Parent) 'requirements.txt'
  if (Test-Path $req) { & $vpy -m pip install --disable-pip-version-check --upgrade -r $req certifi }
  else { & $vpy -m pip install --disable-pip-version-check --upgrade numpy scipy opencv-python scikit-image scikit-learn pillow tifffile matplotlib gdstk certifi }
  if ($LASTEXITCODE -ne 0) { Fail 'the packages could not be installed (see above)' }
  $imp = 'numpy, scipy, cv2, skimage, sklearn, PIL, tifffile, matplotlib, gdstk, certifi'
}
& $vpy -c "import $imp; print('   packages: all import (numpy ' + numpy.__version__ + ', OpenCV ' + cv2.__version__ + ')')"
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
