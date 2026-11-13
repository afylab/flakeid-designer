# Finds the Python to build the Windows app with: prints the path of python.exe (3.12, else 3.11, else 3.13), or prints
# nothing and lists every Python it saw (on stderr).  Looks at: $env:PYTHON, the py launcher, the registry (PEP 514:
# python.org installers), the usual folders (python.org, the Python install manager, Program Files, C:\), the PATH.
$ErrorActionPreference = 'SilentlyContinue'
$cands = New-Object System.Collections.Generic.List[string]
if ($env:PYTHON) { $cands.Add($env:PYTHON) }
foreach ($v in '3.12', '3.11', '3.13') {
  $p = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
  if ($LASTEXITCODE -eq 0 -and $p) { $cands.Add(($p | Select-Object -First 1).Trim()) }
}
foreach ($root in 'HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore', 'HKLM:\Software\WOW6432Node\Python\PythonCore') {
  Get-ChildItem $root | ForEach-Object {
    $e = (Get-ItemProperty (Join-Path $_.PSPath 'InstallPath')).ExecutablePath
    if (-not $e) { $d = (Get-ItemProperty (Join-Path $_.PSPath 'InstallPath')).'(default)'; if ($d) { $e = Join-Path $d 'python.exe' } }
    if ($e) { $cands.Add($e) }
  }
}
foreach ($pat in "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "$env:LOCALAPPDATA\Python\pythoncore-3*\python.exe",
                 "$env:ProgramFiles\Python3*\python.exe", "C:\Python3*\python.exe") {
  Get-ChildItem $pat | ForEach-Object { $cands.Add($_.FullName) }
}
Get-Command python -All | ForEach-Object { if ($_.Source -notmatch 'WindowsApps') { $cands.Add($_.Source) } }
$seen = @{}
foreach ($c in $cands) {
  if (-not $c -or -not (Test-Path $c)) { continue }
  $full = (Resolve-Path $c).Path
  if ($seen.ContainsKey($full)) { continue }
  $v = & $full -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))" 2>$null
  if ($LASTEXITCODE -eq 0 -and $v) { $seen[$full] = ($v | Select-Object -First 1).Trim() }
}
foreach ($want in '3.12', '3.11', '3.13') {
  foreach ($k in $seen.Keys) { if ($seen[$k] -eq $want) { Write-Output $k; exit 0 } }
}
[Console]::Error.WriteLine("Pythons found on this PC (none of 3.12 / 3.11 / 3.13):")
if ($seen.Count -eq 0) { [Console]::Error.WriteLine("  none") }
foreach ($k in $seen.Keys) { [Console]::Error.WriteLine("  $($seen[$k])  $k") }
exit 1
