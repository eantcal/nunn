<#
.SYNOPSIS
    Generate a Visual Studio solution for nunn using CMake.

.DESCRIPTION
    Runs CMake with the Visual Studio generator so that the solution and all
    project files are created inside the build directory.  The generated
    solution can then be opened directly in Visual Studio.

    The solution is always written to .\build\, keeping it separate from any
    command-line or CI builds that use a different generator or config.

.PARAMETER Config
    Default build configuration embedded in the solution (default: Release).
    Valid values: Release, Debug, RelWithDebInfo, MinSizeRel.

.PARAMETER Open
    If specified, open the generated solution in Visual Studio after generation.

.EXAMPLE
    .\generate-vs.ps1
    .\generate-vs.ps1 -Config Debug -Open
#>

param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release',

    [switch]$Open
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root  = $PSScriptRoot
$Build = Join-Path $Root 'build'

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

# Remove stale cache so CMake re-detects the VS instance correctly.
$Cache = Join-Path $Build 'CMakeCache.txt'
if (Test-Path $Cache) {
    Write-Step "Removing stale CMakeCache.txt"
    Remove-Item -Force $Cache
}

Write-Step "Generating Visual Studio solution (config: $Config)"

cmake -S $Root -B $Build `
      -DCMAKE_BUILD_TYPE=$Config `
      -DNUNN_BUILD_TESTS=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# Find the generated solution file (.sln or .slnx)
$Sln = Get-ChildItem -Path $Build -Filter '*.sln'  | Select-Object -First 1
if (-not $Sln) {
    $Sln = Get-ChildItem -Path $Build -Filter '*.slnx' | Select-Object -First 1
}

if ($Sln) {
    Write-Host "`nSolution: $($Sln.FullName)" -ForegroundColor Green
} else {
    Write-Host "`nConfiguration done. No .sln file found in $Build." -ForegroundColor Yellow
}

if ($Open -and $Sln) {
    Write-Step "Opening solution in Visual Studio"
    Start-Process $Sln.FullName
}
