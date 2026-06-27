# Common paths and helpers sourced by every training script.
# Usage: . "$PSScriptRoot\_common.ps1"
#
# mnist is now a static library; no DLL path manipulation is needed.

$RepoRoot   = Resolve-Path "$PSScriptRoot\..\.."
$ExeRelease = "$RepoRoot\build\examples\mnist_test\Release\mnist_test.exe"
$ExeDebug   = "$RepoRoot\build\examples\mnist_test\Debug\mnist_test.exe"
$DataPath   = "$RepoRoot\build\examples\mnist_test"
$ModelsDir  = "$RepoRoot\scripts\mnist\models"

if (Test-Path $ExeRelease) {
    $Exe = $ExeRelease
} elseif (Test-Path $ExeDebug) {
    $Exe = $ExeDebug
} else {
    Write-Error "mnist_test.exe not found. Build the project first."
    exit 1
}

New-Item -ItemType Directory -Force $ModelsDir | Out-Null

# NOTE: parameter is named $CmdArgs (not $Args) to avoid shadowing
# PowerShell's automatic $args variable, which would break @-splatting.
function Run-Training {
    param(
        [string]   $Label,
        [string[]] $CmdArgs,
        [string]   $LogFile = ""
    )

    $banner = "=" * 60
    Write-Host ""
    Write-Host $banner -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Yellow
    Write-Host "  $Exe $CmdArgs" -ForegroundColor DarkGray
    Write-Host $banner -ForegroundColor Cyan
    Write-Host ""

    $start = Get-Date
    if ($LogFile) {
        & $Exe @CmdArgs 2>&1 | Tee-Object -FilePath $LogFile
    } else {
        & $Exe @CmdArgs
    }
    $elapsed = (Get-Date) - $start
    Write-Host ""
    Write-Host "Finished in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}
