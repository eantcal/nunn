# Baseline: Sigmoid hidden + Sigmoid output, MSE cost.
# Classic configuration; typically reaches ~97% accuracy in 30 epochs.
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "Sigmoid + MSE  |  LR=0.025  M=0.9  HL=300  epochs=30" `
    -LogFile "$ModelsDir\sigmoid_mse.log" `
    -Args    @(
        "-p", $DataPath,
        "-a", "sigmoid",
        "-r", "0.025",
        "-m", "0.9",
        "-e", "30",
        "-hl", "300",
        "-s", "$ModelsDir\sigmoid_mse.json"
    )
