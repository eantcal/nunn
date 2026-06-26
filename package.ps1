<#
.SYNOPSIS
    Build nunn and create an installer package with CPack.

.DESCRIPTION
    Configures and builds the project in Release mode, then runs CPack to
    produce a Windows installer (NSIS exe by default), a ZIP archive, or an
    MSI (WiX).

    Requires CMake 3.14+ on PATH.  For NSIS packages, NSIS must be installed.
    For WiX (MSI) packages, WiX Toolset v3 or v4 must be installed.

.PARAMETER Config
    CMake build configuration (default: Release).

.PARAMETER Generator
    CPack generator to use.  Supported values:
      NSIS   - Windows installer wizard (.exe)  [requires NSIS]
      WIX    - MSI package                      [requires WiX Toolset]
      ZIP    - Zip archive (no installer UI)
      ALL    - NSIS + ZIP (default)

.PARAMETER NoBuild
    Skip the CMake configure + build step and run CPack against an existing
    build directory.  Useful when you have already built the project.

.EXAMPLE
    .\package.ps1
    .\package.ps1 -Generator ZIP
    .\package.ps1 -Generator WIX
    .\package.ps1 -NoBuild -Generator ZIP
#>

param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release',

    [ValidateSet('NSIS', 'WIX', 'ZIP', 'ALL')]
    [string]$Generator = 'ALL',

    [switch]$NoBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root  = $PSScriptRoot
$Build = Join-Path $Root 'build'

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Assert-Success([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$step failed (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── 1. Build ──────────────────────────────────────────────────────────────────
if (-not $NoBuild) {
    # Remove stale cache to avoid VS instance path mismatches.
    $Cache = Join-Path $Build 'CMakeCache.txt'
    if (Test-Path $Cache) {
        Write-Step "Removing stale CMakeCache.txt"
        Remove-Item -Force $Cache
    }

    Write-Step "Configuring (config: $Config)"
    cmake -S $Root -B $Build `
          -DCMAKE_BUILD_TYPE=$Config `
          -DNUNN_BUILD_TESTS=OFF
    Assert-Success "CMake configure"

    Write-Step "Building"
    cmake --build $Build --config $Config
    Assert-Success "CMake build"
}

# ── 2. Package with CPack ─────────────────────────────────────────────────────
$CPackConfig = Join-Path $Build 'CPackConfig.cmake'
if (-not (Test-Path $CPackConfig)) {
    Write-Host "CPackConfig.cmake not found in $Build. Run without -NoBuild first." -ForegroundColor Red
    exit 1
}

$cpackGenerators = switch ($Generator) {
    'ALL'  { @('NSIS', 'ZIP') }
    default { @($Generator) }
}

foreach ($gen in $cpackGenerators) {
    Write-Step "Packaging with CPack generator: $gen"
    cpack --config $CPackConfig -C $Config -G $gen -B $Root/dist
    Assert-Success "CPack $gen"
}

# ── 3. Report ─────────────────────────────────────────────────────────────────
Write-Host "`nPackages written to: $Root\dist" -ForegroundColor Green
Get-ChildItem -Path (Join-Path $Root 'dist') -File |
    Where-Object { $_.Name -notlike '_CPack_Packages*' } |
    Select-Object Name, @{N='Size';E={"$([math]::Round($_.Length/1MB,1)) MB"}} |
    Format-Table -AutoSize
