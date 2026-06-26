<#
.SYNOPSIS
    Convert all legacy .net files in the repository to JSON format.

.DESCRIPTION
    Builds the net2json converter (if not already built), then converts every
    .net file found under the repo root.  Each output .json file is written
    alongside its .net source with the same stem.

.PARAMETER Config
    Build configuration used when compiling net2json (default: Release).

.PARAMETER NoBuild
    Skip the build step and assume net2json is already compiled.

.EXAMPLE
    .\convert-nets.ps1
    .\convert-nets.ps1 -Config Debug
    .\convert-nets.ps1 -NoBuild
#>

param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release',

    [switch]$NoBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root  = $PSScriptRoot
$Build = Join-Path $Root 'build'

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

# ── Build net2json ────────────────────────────────────────────────────────────
if (-not $NoBuild) {
    Write-Step "Configuring"
    $Cache = Join-Path $Build 'CMakeCache.txt'
    if (Test-Path $Cache) { Remove-Item -Force $Cache }
    cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=$Config -DNUNN_BUILD_TESTS=OFF
    if ($LASTEXITCODE -ne 0) { Write-Host "Configure failed" -ForegroundColor Red; exit $LASTEXITCODE }

    Write-Step "Building net2json"
    cmake --build $Build --config $Config --target net2json --parallel ([Environment]::ProcessorCount)
    if ($LASTEXITCODE -ne 0) { Write-Host "Build failed" -ForegroundColor Red; exit $LASTEXITCODE }
}

# Locate the compiled binary (MSVC puts it in a config subfolder)
$Candidates = @(
    (Join-Path $Build "tools\net2json\$Config\net2json.exe"),
    (Join-Path $Build "tools\net2json\net2json.exe"),
    (Join-Path $Build "tools\net2json\net2json")
)
$Converter = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Converter) {
    Write-Host "Cannot find net2json binary under $Build" -ForegroundColor Red
    exit 1
}

# ── Find and convert .net files ───────────────────────────────────────────────
Write-Step "Converting .net files"

$NetFiles = Get-ChildItem -Path $Root -Recurse -Filter '*.net' |
            Where-Object { $_.FullName -notmatch '\\build\\' }

if ($NetFiles.Count -eq 0) {
    Write-Host "No .net files found." -ForegroundColor Yellow
    exit 0
}

$Errors = 0
foreach ($f in $NetFiles) {
    $out = [System.IO.Path]::ChangeExtension($f.FullName, '.json')
    & $Converter $f.FullName $out
    if ($LASTEXITCODE -ne 0) { $Errors++ }
}

if ($Errors -gt 0) {
    Write-Host "`n$Errors file(s) failed to convert." -ForegroundColor Red
    exit 1
}

Write-Host "`nDone - $($NetFiles.Count) file(s) converted." -ForegroundColor Green
