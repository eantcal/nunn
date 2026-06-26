# Common paths and helpers sourced by every training script.
# Usage: . "$PSScriptRoot\_common.ps1"

$RepoRoot  = Resolve-Path "$PSScriptRoot\..\..\"
$ExeRelease = "$RepoRoot\build\examples\mnist_test\Release\mnist_test.exe"
$ExeDebug   = "$RepoRoot\build\examples\mnist_test\Debug\mnist_test.exe"
$DataPath   = "$RepoRoot\build\examples\mnist_test\"
$ModelsDir  = "$RepoRoot\scripts\mnist\models"

# Prefer Release for speed; fall back to Debug.
if (Test-Path $ExeRelease) {
    $Exe = $ExeRelease
    # Add DLL directories to PATH so Windows can locate mnist.dll and nunn.dll.
    $env:PATH = "$RepoRoot\build\mnist\Release;$RepoRoot\build\nunn\Release;$env:PATH"
} elseif (Test-Path $ExeDebug) {
    $Exe = $ExeDebug
    $env:PATH = "$RepoRoot\build\mnist\Debug;$RepoRoot\build\nunn\Debug;$env:PATH"
} else {
    Write-Error "mnist_test.exe not found. Build the project first."
    exit 1
}

New-Item -ItemType Directory -Force $ModelsDir | Out-Null

function Run-Training {
    param(
        [string]   $Label,
        [string[]] $Args,
        [string]   $LogFile = ""
    )

    $banner = "=" * 60
    Write-Host ""
    Write-Host $banner -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Yellow
    Write-Host "  $Exe $Args" -ForegroundColor DarkGray
    Write-Host $banner -ForegroundColor Cyan
    Write-Host ""

    $start = Get-Date
    if ($LogFile) {
        & $Exe @Args 2>&1 | Tee-Object -FilePath $LogFile
    } else {
        & $Exe @Args
    }
    $elapsed = (Get-Date) - $start
    Write-Host ""
    Write-Host "Finished in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}
