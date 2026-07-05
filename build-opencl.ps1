<#
.SYNOPSIS
    Configure and build nunn with ArrayFire/OpenCL enabled.

.DESCRIPTION
    Uses the standard ArrayFire Windows installation directory by default:
    C:\Program Files\ArrayFire\v3

    The script points CMake at ArrayFire's package config directory and enables
    NUNN_ENABLE_OPENCL. If ArrayFire is not found at that path, CMake will report
    the detection problem during configure.

.PARAMETER ArrayFireRoot
    ArrayFire installation directory. Defaults to C:\Program Files\ArrayFire\v3.

.PARAMETER BuildDir
    CMake build directory. Defaults to .\build.

.PARAMETER Config
    Build configuration. Defaults to Release.

.PARAMETER Target
    Optional target to build, for example nunn, mnist_test, or ocr_test.

.PARAMETER CleanCache
    Remove CMakeCache.txt before configuring, forcing CMake to re-detect ArrayFire.

.EXAMPLE
    .\build-opencl.ps1
    .\build-opencl.ps1 -Target mnist_test
    .\build-opencl.ps1 -ArrayFireRoot "D:\ArrayFire\v3" -CleanCache
#>

param(
    [string]$ArrayFireRoot = 'C:\Program Files\ArrayFire\v3',

    [string]$BuildDir = (Join-Path $PSScriptRoot 'build'),

    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release',

    [string]$Target = '',

    [switch]$CleanCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot

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

if (-not (Test-Path -LiteralPath $ArrayFireRoot)) {
    throw "ArrayFireRoot not found: $ArrayFireRoot"
}

$ArrayFireDir = Join-Path $ArrayFireRoot 'cmake'
if (-not (Test-Path -LiteralPath $ArrayFireDir)) {
    throw "ArrayFire CMake package directory not found: $ArrayFireDir"
}

if (-not (Test-Path -LiteralPath $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

$Cache = Join-Path $BuildDir 'CMakeCache.txt'
if ($CleanCache -and (Test-Path -LiteralPath $Cache)) {
    Write-Step "Removing CMake cache"
    Remove-Item -LiteralPath $Cache -Force
}

Invoke-Cmd "Configuring with ArrayFire/OpenCL ($Config)" {
    cmake -S $Root -B $BuildDir `
        -DCMAKE_BUILD_TYPE=$Config `
        -DNUNN_ENABLE_OPENCL=ON `
        -DArrayFire_DIR="$ArrayFireDir" `
        -DNUNN_BUILD_TESTS=ON
}

$buildArgs = @('--build', $BuildDir, '--config', $Config)
if ($Target) {
    $buildArgs += @('--target', $Target)
}

Invoke-Cmd "Building$(if ($Target) { " target '$Target'" } else { '' })" {
    cmake @buildArgs
}

Write-Host "`nArrayFire root : $ArrayFireRoot" -ForegroundColor Green
Write-Host "ArrayFire_DIR  : $ArrayFireDir" -ForegroundColor Green
Write-Host "Build complete." -ForegroundColor Green
