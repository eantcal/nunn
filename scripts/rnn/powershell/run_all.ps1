# run_all.ps1 — run all RNN examples and comparisons.
#
# Usage:
#   .\run_all.ps1            # full run (may take several minutes)
#   .\run_all.ps1 -Quick     # reduced epochs, fast smoke-test

param(
    [switch] $Quick
)

. "$PSScriptRoot\_common.ps1"

$banner = "#" * 60

Write-Host ""
Write-Host $banner                                    -ForegroundColor Magenta
Write-Host "  nunn RNN benchmark suite"               -ForegroundColor Magenta
Write-Host ("  Quick mode: " + $(if ($Quick) { "ON" } else { "OFF" })) -ForegroundColor Magenta
Write-Host $banner                                    -ForegroundColor Magenta

# --- Sine-wave prediction ---
Write-Host ""
Write-Host "  [1/3]  Sine-wave prediction (VanillaRnn / GRU / LSTM)" -ForegroundColor Cyan
$sineEpochs = if ($Quick) { 400 } else { 1500 }
& "$PSScriptRoot\run_sine.ps1" -Epochs $sineEpochs -Hidden 32

# --- Adding problem benchmark ---
Write-Host ""
Write-Host "  [2/3]  Adding problem benchmark (all models)" -ForegroundColor Cyan
$addEpochs = if ($Quick) { 100 } else { 500 }
& "$PSScriptRoot\run_adding.ps1" -SeqLen 20 -Hidden 32 -Epochs $addEpochs

# --- Character-level language model ---
Write-Host ""
Write-Host "  [3/3]  Character-level language model (VanillaRnn / GRU / LSTM)" -ForegroundColor Cyan
$charEpochs = if ($Quick) { 150 } else { 800 }
& "$PSScriptRoot\run_char.ps1" -Epochs $charEpochs -Hidden 64

Write-Host ""
Write-Host $banner   -ForegroundColor Magenta
Write-Host "  Done." -ForegroundColor Magenta
Write-Host $banner   -ForegroundColor Magenta
Write-Host ""
