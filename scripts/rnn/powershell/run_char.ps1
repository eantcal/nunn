# run_char.ps1 — train a character-level language model and generate text.
#
# Usage:
#   .\run_char.ps1
#   .\run_char.ps1 -Quick
#   .\run_char.ps1 -Model gru -Epochs 1200 -Hidden 128
#   .\run_char.ps1 -Model all
#
# Parameters:
#   -Quick       Use 150 epochs for a fast smoke-test.
#   -Model       vanilla | gru | lstm | all (default: all)
#   -Epochs      Override epoch count.
#   -Hidden      Hidden units (default 64).
#   -GenLen      Characters to generate (default 120).
#   -Temperature Sampling temperature (default 0.8).

param(
    [switch] $Quick,
    [string] $Model       = "all",
    [int]    $Epochs      = 0,
    [int]    $Hidden      = 64,
    [int]    $GenLen      = 120,
    [double] $Temperature = 0.8
)

. "$PSScriptRoot\_common.ps1"
$Exe = Find-Exe "rnn_char"

$DefaultEpochs = if ($Quick) { 150 } else { 800 }
if ($Epochs -eq 0) { $Epochs = $DefaultEpochs }

$Models = switch ($Model.ToLower()) {
    "all"     { @("vanilla", "gru", "lstm") }
    "vanilla" { @("vanilla") }
    "gru"     { @("gru") }
    "lstm"    { @("lstm") }
    default   { Write-Error "Unknown model '$Model'. Use: vanilla | gru | lstm | all"; exit 1 }
}

foreach ($m in $Models) {
    $flag  = if ($m -eq "vanilla") { @() } else { @("--$m") }
    $label = "rnn_char -- $($m.ToUpper())  epochs=$Epochs  hidden=$Hidden  temp=$Temperature"
    Run-Example $label $Exe (@($flag) + @($Epochs, $Hidden, $GenLen, $Temperature) | Where-Object { $_ -ne $null })
}
