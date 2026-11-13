# vdW Stack Designer: everything it needs on a fresh Windows 10 / 11, for this user (no admin), then its desktop and
# Start-menu shortcut.  Started by "Install shortcut (Windows).bat" (next to this file); safe to run again (it then only
# checks, updates the packages and makes the shortcut again).
#   1. Python: one already on this PC (3.10 - 3.13, not the Microsoft Store stub), else Python 3.12.10 from python.org,
#      installed for this user only (checked against its known SHA-256 before it runs).
#   2. A Python environment of the designer's own (%USERPROFILE%\.flakeid_designer_venv): pip and the packages the
#      designer's builds and tools need (requirements.txt beside the designer, and certifi for Update...).
#   3. The shortcut: "FlakeIdentifier Designer" on the desktop and in the Start menu, started with that environment.
# Log: %USERPROFILE%\.flakeid_designer_install_log.txt.  For tests: FLAKEID_VENV (another environment folder),
# FLAKEID_INSTALL_NO_SHORTCUT=1 (no shortcut).
$ErrorActionPreference = 'Continue'                  # (not Stop: in PowerShell 5.1 a program's warnings on stderr, e.g. pip's,
                                                     # would end the script; every step's result is checked instead)
$ProgressPreference = 'SilentlyContinue'                      # (a progress bar makes downloads many times slower)
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$PyVer = '3.12.10'
$PyUrl = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-amd64.exe"
$PySha = '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
$Ok = @('3.12', '3.13', '3.11', '3.10')                         # (in this order of preference)
$Here = $PSScriptRoot                                           # (the flakeid_designer folder)
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
  Get-Command python -All -ErrorAction SilentlyContinue | ForEach-Object { $c.Add($_.Source) }
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
if ([Environment]::Is64BitOperatingSystem -eq $false) { Fail 'a 64-bit Windows is needed' }

# 1. Python
$py = FindPython
if ($py) {
  Say "1. Python: $py ($(PyVersion $py))"
} else {
  Say "1. Python: none usable on this PC; downloading Python $PyVer from python.org ..."
  $inst = Join-Path $env:TEMP "python-$PyVer-amd64.exe"
  try { Invoke-WebRequest -Uri $PyUrl -OutFile $inst -UseBasicParsing -ErrorAction Stop } catch { Fail "the download failed ($($_.Exception.Message)); is this PC online?" }
  $h = (Get-FileHash $inst -Algorithm SHA256).Hash.ToLower()
  if ($h -ne $PySha) { Remove-Item $inst -ErrorAction SilentlyContinue; Fail "the downloaded Python installer is not the expected file (SHA-256 $h)" }
  Say "   installing Python $PyVer for this user (no admin needed) ..."
  $p = Start-Process -FilePath $inst -Wait -PassThru -ArgumentList @('/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1',
         'InstallLauncherAllUsers=0', 'Include_test=0', 'Include_doc=0', 'Shortcuts=0')
  Remove-Item $inst -ErrorAction SilentlyContinue
  if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { Fail "the Python installer stopped with code $($p.ExitCode)" }
  $py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
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
$req = Join-Path (Split-Path $Here -Parent) 'requirements.txt'
Say '   the packages (numpy, scipy, OpenCV, scikit-image, ...) ...'
if (Test-Path $req) { & $vpy -m pip install --disable-pip-version-check --upgrade -r $req certifi }
else { & $vpy -m pip install --disable-pip-version-check --upgrade numpy scipy opencv-python scikit-image scikit-learn pillow tifffile matplotlib gdstk certifi }
if ($LASTEXITCODE -ne 0) { Fail 'the packages could not be installed (see above)' }
& $vpy -c "import numpy, scipy, cv2, skimage, sklearn, PIL, tifffile, matplotlib, gdstk, certifi; print('   packages: all import (numpy ' + numpy.__version__ + ', OpenCV ' + cv2.__version__ + ')')"
if ($LASTEXITCODE -ne 0) { Fail 'the packages were installed but do not all import (see above)' }

# 3. the shortcut, started with that environment
if ($env:FLAKEID_INSTALL_NO_SHORTCUT -eq '1') {
  Say '3. the shortcut: skipped (FLAKEID_INSTALL_NO_SHORTCUT)'
} else {
  Say '3. the shortcut ...'
  & $vpy $Here --install-shortcut
  if ($LASTEXITCODE -ne 0) { Fail 'the shortcut could not be made (see above)' }
}
Say ''
Say 'Done: start the designer from "FlakeIdentifier Designer" on the desktop or in the Start menu.'
try { Stop-Transcript | Out-Null } catch { }
exit 0
