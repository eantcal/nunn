# Deep network: 3 hidden layers (784-512-256-128-10), ReLU + CrossEntropy.
# ReLU + depth is the closest this network gets to a modern DNN style.
# Low LR is essential to avoid exploding gradients through 3 ReLU layers.
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "Deep ReLU + CrossEntropy  |  LR=0.005  M=0.9  HL=512+256+128  epochs=50" `
    -LogFile "$ModelsDir\deep_relu_ce.log" `
    -CmdArgs @(
        "-p", $DataPath,
        "-a", "relu",
        "-c",
        "-r", "0.005",
        "-m", "0.9",
        "-e", "50",
        "-hl", "512",
        "-hl", "256",
        "-hl", "128",
        "-s", "$ModelsDir\deep_relu_ce.json"
    )
