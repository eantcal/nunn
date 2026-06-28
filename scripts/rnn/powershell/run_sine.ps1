# run_sine.ps1 — train and compare VanillaRnn / GRU / LSTM on sine-wave prediction.
#
# Usage:
#   .\run_sine.ps1
#   .\run_sine.ps1 -Quick
#   .\run_sine.ps1 -Epochs 2000 -Hidden 64 -Lr 0.003
#   .\run_sine.ps1 -Model gru
#
# Parameters:
#   -Quick    400 epochs instead of 1500 (fast smoke-test).
#   -Model    vanilla | gru | lstm | all  (default: all)
#   -Epochs   Override epoch count.
#   -Hidden   Hidden units (default 32).
#   -Lr       Learning rate (default 0.005).

param(
    [switch] $Quick,
    [string] $Model  = "all",
    [int]    $Epochs = 0,
    [int]    $Hidden = 32,
    [double] $Lr     = 0.005
)

. "$PSScriptRoot\_common.ps1"
$Exe = Find-Exe "rnn_sine"

$DefaultEpochs = if ($Quick) { 400 } else { 1500 }
if ($Epochs -eq 0) { $Epochs = $DefaultEpochs }

$Models = switch ($Model.ToLower()) {
    "all"     { @("vanilla", "gru", "lstm") }
    "vanilla" { @("vanilla") }
    "gru"     { @("gru") }
    "lstm"    { @("lstm") }
    default   { Write-Error "Unknown model '$Model'. Use: vanilla | gru | lstm | all"; exit 1 }
}

$TmpDir  = [System.IO.Path]::GetTempPath()
$Results = @()

foreach ($m in $Models) {
    $flag   = if ($m -eq "vanilla") { @() } else { @("--$m") }
    $label  = "rnn_sine -- $($m.ToUpper())  epochs=$Epochs  hidden=$Hidden  lr=$Lr"
    $logTmp = Join-Path $TmpDir "nunn_rnn_sine_$m.log"

    $banner = "=" * 60
    Write-Host ""
    Write-Host $banner                                         -ForegroundColor Cyan
    Write-Host "  $label"                                     -ForegroundColor Yellow
    Write-Host "  $Exe $flag $Epochs $Hidden $Lr"            -ForegroundColor DarkGray
    Write-Host $banner                                         -ForegroundColor Cyan
    Write-Host ""

    $t0 = Get-Date
    & $Exe @flag $Epochs $Hidden $Lr | Tee-Object -FilePath $logTmp
    $elapsed = (Get-Date) - $t0

    # Parse max autoregressive error from log
    $mae = "n/a"
    $line = Get-Content $logTmp -ErrorAction SilentlyContinue |
            Where-Object { $_ -match "Max absolute error" } |
            Select-Object -Last 1
    if ($line -and $line -match '[\d.]+$') { $mae = $Matches[0] }

    Write-Host ""
    Write-Host ("Finished in " + $elapsed.ToString('mm\:ss')) -ForegroundColor Green

    $Results += [PSCustomObject]@{
        Model    = $m.ToUpper()
        MaxAE    = $mae
        Time     = $elapsed.ToString('mm\:ss')
    }
}

if ($Results.Count -gt 1) {
    Write-Host ""
    Write-Host ("=" * 60)                                       -ForegroundColor Cyan
    Write-Host ("  Comparison  (epochs=$Epochs  hidden=$Hidden  lr=$Lr)") -ForegroundColor Cyan
    Write-Host ("=" * 60)                                       -ForegroundColor Cyan
    $Results | Format-Table -AutoSize
}
exit 0
