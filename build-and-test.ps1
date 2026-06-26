<#
.SYNOPSIS
    Build nunn and run the non-regression test suite.

.PARAMETER Config
    Build configuration: Release (default) or Debug.

.PARAMETER Clean
    If specified, delete the build directory before configuring.

.PARAMETER Jobs
    Parallel job count passed to cmake --build (default: number of logical CPUs).

.EXAMPLE
    .\build-and-test.ps1
    .\build-and-test.ps1 -Config Debug -Clean
#>

param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release',

    [switch]$Clean,

    [int]$Jobs = [Environment]::ProcessorCount
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root  = $PSScriptRoot
$Build = Join-Path $Root 'build'

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Invoke-Cmd([string]$desc, [scriptblock]$cmd) {
    Write-Step $desc
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── Clean ────────────────────────────────────────────────────────────────────
if ($Clean -and (Test-Path $Build)) {
    Write-Step "Removing build directory"
    Remove-Item -Recurse -Force $Build
}

# Always remove CMakeCache.txt so CMake re-detects the VS instance path.
# This is safe: cached compiler/linker paths can point to an edition that
# no longer exists (e.g. Community → Professional), causing configure to fail.
$Cache = Join-Path $Build 'CMakeCache.txt'
if (Test-Path $Cache) {
    Write-Step "Removing stale CMakeCache.txt"
    Remove-Item -Force $Cache
}

# ── Configure ────────────────────────────────────────────────────────────────
Invoke-Cmd "Configuring ($Config)" {
    cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=$Config -DNUNN_BUILD_TESTS=ON
}

# ── Build ─────────────────────────────────────────────────────────────────────
Invoke-Cmd "Building (jobs=$Jobs)" {
    cmake --build $Build --config $Config --parallel $Jobs
}

# ── Test ──────────────────────────────────────────────────────────────────────
Invoke-Cmd "Running tests" {
    ctest --test-dir $Build -C $Config --output-on-failure
}

Write-Host "`nAll steps completed successfully." -ForegroundColor Green
