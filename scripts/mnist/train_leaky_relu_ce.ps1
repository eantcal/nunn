# LeakyReLU hidden + Sigmoid output, Cross-Entropy cost.
# Avoids the "dying ReLU" problem (neurons stuck at 0) because the negative
# slope (alpha=0.01) keeps a small gradient flowing for negative inputs.
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "LeakyReLU + CrossEntropy  |  LR=0.01  M=0.9  HL=512  epochs=30" `
    -LogFile "$ModelsDir\leaky_relu_ce.log" `
    -CmdArgs @(
        "-p", $DataPath,
        "-a", "leaky_relu",
        "-c",
        "-r", "0.01",
        "-m", "0.9",
        "-e", "30",
        "-hl", "512",
        "-s", "$ModelsDir\leaky_relu_ce.json"
    )
