# msvc-env.ps1 -- put the MSVC command-line environment for one architecture
# into the rest of the job: what ilammy/msvc-dev-cmd did, as a composite step
# so no JavaScript runtime (and no deprecated Node version) is involved.
#
#   pwsh msvc-env.ps1 -Arch x64|x86|amd64_x86|...
#
# Runs vcvarsall.bat for the newest Visual Studio with the C++ tools and
# writes every variable it set or changed to $GITHUB_ENV. Called a second time
# in the same job (x86 after x64), it first undoes the PATH entries the first
# call added and clears INCLUDE/LIB/LIBPATH, so the two do not stack.
param([Parameter(Mandatory = $true)][string]$Arch)
$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$vs = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vs) { throw 'no Visual Studio installation with the C++ tools' }
$bat = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $bat)) { throw "vcvarsall.bat not found at $bat" }

# What the job had before this step, to diff against.
$before = @{}
Get-ChildItem env: | ForEach-Object { $before[$_.Name.ToUpperInvariant()] = $_.Value }

# Start vcvarsall from a clean slate for the variables it builds up.
$cleanPath = $env:PATH
if ($env:MSVC_ENV_ADDED_PATH) {
    $drop = $env:MSVC_ENV_ADDED_PATH -split ';'
    $cleanPath = (($env:PATH -split ';') | Where-Object { $_ -and ($drop -notcontains $_) }) -join ';'
}
$env:PATH = $cleanPath
# vcvarsall looks itself up with vswhere and complains on stderr when it is
# not on PATH; give it the installer directory.
$installer = Split-Path $vswhere
if (($cleanPath -split ';') -notcontains $installer) { $env:PATH = "$installer;$cleanPath" }
foreach ($v in 'INCLUDE', 'LIB', 'LIBPATH') { Remove-Item "env:$v" -ErrorAction SilentlyContinue }

# A .cmd file sidesteps every quoting rule between pwsh, cmd and the path.
$tmp = Join-Path ([IO.Path]::GetTempPath()) "msvc-env-$Arch.cmd"
Set-Content -Path $tmp -Encoding ascii -Value @(
    '@echo off'
    "call `"$bat`" $Arch >nul || exit /b 1"
    'set'
)
$lines = & cmd.exe /d /c $tmp
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat $Arch failed ($LASTEXITCODE)" }
Remove-Item $tmp

$out = [System.Collections.Generic.List[string]]::new()
$newPath = $null
foreach ($line in $lines) {
    $i = $line.IndexOf('=')
    if ($i -le 0) { continue }
    $name = $line.Substring(0, $i)
    $value = $line.Substring($i + 1)
    $key = $name.ToUpperInvariant()
    if ($key -eq 'PATH') { $newPath = $value; continue }
    if ($key -eq 'MSVC_ENV_ADDED_PATH') { continue }
    if ($before.ContainsKey($key) -and $before[$key] -eq $value) { continue }
    $out.Add("$name=$value")
}
if (-not $newPath) { throw 'vcvarsall.bat left no PATH' }

# Remember what was added, for the next call to take back out.
$old = $cleanPath -split ';'
$added = ($newPath -split ';') | Where-Object { $_ -and ($old -notcontains $_) }
$out.Add("PATH=$newPath")
$out.Add("MSVC_ENV_ADDED_PATH=$($added -join ';')")

if ($env:GITHUB_ENV) {
    $out | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
} else {
    $out    # outside Actions: print what would be exported
}
Write-Host "MSVC environment for $Arch from $vs ($($out.Count) variables)"
