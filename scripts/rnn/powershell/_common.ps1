# Common paths and helpers for RNN scripts.
# Usage: . "$PSScriptRoot\_common.ps1"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
$BuildDir = "$RepoRoot\build\examples"

function Find-Exe {
    param([string]$Name)
    $rel = "$BuildDir\$Name\Release\$Name.exe"
    $dbg = "$BuildDir\$Name\Debug\$Name.exe"
    if (Test-Path $rel) { return $rel }
    if (Test-Path $dbg) { return $dbg }
    Write-Error "$Name.exe not found. Build the project first (cmake --build build --config Release)."
    exit 1
}

function Run-Example {
    param(
        [string]   $Label,
        [string]   $Exe,
        [string[]] $CmdArgs
    )
    $banner = "=" * 60
    Write-Host ""
    Write-Host $banner                      -ForegroundColor Cyan
    Write-Host "  $Label"                   -ForegroundColor Yellow
    Write-Host "  $Exe $($CmdArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host $banner                      -ForegroundColor Cyan
    Write-Host ""
    $start = Get-Date
    & $Exe @CmdArgs
    $elapsed = (Get-Date) - $start
    Write-Host ""
    Write-Host ("Finished in " + $elapsed.ToString('hh\:mm\:ss')) -ForegroundColor Green
}

# Run an example and return its captured output as a string array.
function Invoke-Example {
    param([string]$Exe, [string[]]$CmdArgs)
    & $Exe @CmdArgs 2>&1
}
