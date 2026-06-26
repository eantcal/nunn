# Deep network: 3 hidden layers (784-512-256-128-10), Tanh + CrossEntropy.
# Tests whether depth helps on MNIST with this backprop implementation.
# Needs more epochs and a lower LR to avoid early divergence.
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "Deep Tanh + CrossEntropy  |  LR=0.02  M=0.9  HL=512+256+128  epochs=50" `
    -LogFile "$ModelsDir\deep_tanh_ce.log" `
    -Args    @(
        "-p", $DataPath,
        "-a", "tanh",
        "-c",
        "-r", "0.02",
        "-m", "0.9",
        "-e", "50",
        "-hl", "512",
        "-hl", "256",
        "-hl", "128",
        "-s", "$ModelsDir\deep_tanh_ce.json"
    )
