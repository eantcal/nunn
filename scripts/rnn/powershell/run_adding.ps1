# run_adding.ps1 — run the adding problem benchmark (VanillaRnn vs GRU vs LSTM).
#
# rnn_adding already trains all three architectures and prints a side-by-side
# comparison; this script just sets convenient defaults and calls it.
#
# Usage:
#   .\run_adding.ps1
#   .\run_adding.ps1 -Quick
#   .\run_adding.ps1 -SeqLen 30 -Hidden 64 -Epochs 800

param(
    [switch] $Quick,
    [int]    $SeqLen = 20,
    [int]    $Hidden = 32,
    [int]    $Epochs = 0,
    [double] $Lr     = 0.005
)

. "$PSScriptRoot\_common.ps1"
$Exe = Find-Exe "rnn_adding"

$DefaultEpochs = if ($Quick) { 100 } else { 500 }
if ($Epochs -eq 0) { $Epochs = $DefaultEpochs }

$label = "rnn_adding  seq_len=$SeqLen  hidden=$Hidden  epochs=$Epochs  lr=$Lr"
Run-Example $label $Exe @($SeqLen, $Hidden, $Epochs, $Lr)
