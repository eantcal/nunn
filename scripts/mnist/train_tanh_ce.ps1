# Tanh hidden + Sigmoid output, Cross-Entropy cost.
# Tanh centres activations around zero, which often speeds up early learning
# compared to Sigmoid.  Good default for single-hidden-layer experiments.
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "Tanh + CrossEntropy  |  LR=0.05  M=0.9  HL=512  epochs=30" `
    -LogFile "$ModelsDir\tanh_ce.log" `
    -Args    @(
        "-p", $DataPath,
        "-a", "tanh",
        "-c",
        "-r", "0.05",
        "-m", "0.9",
        "-e", "30",
        "-hl", "512",
        "-s", "$ModelsDir\tanh_ce.json"
    )
