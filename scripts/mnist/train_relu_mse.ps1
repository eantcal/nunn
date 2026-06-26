# ReLU hidden + Sigmoid output, MSE cost.
# ReLU avoids vanishing gradients in the hidden layers; needs a lower LR
# because the derivative is either 0 or 1 (no natural dampening like sigmoid).
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "ReLU + MSE  |  LR=0.01  M=0.9  HL=512  epochs=30" `
    -LogFile "$ModelsDir\relu_mse.log" `
    -CmdArgs @(
        "-p", $DataPath,
        "-a", "relu",
        "-r", "0.01",
        "-m", "0.9",
        "-e", "30",
        "-hl", "512",
        "-s", "$ModelsDir\relu_mse.json"
    )
