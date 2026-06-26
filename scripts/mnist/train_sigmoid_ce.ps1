# Sigmoid hidden + Sigmoid output, Cross-Entropy cost.
# CE + Sigmoid is the theoretically correct pairing; gradient flows better
# than MSE because the sigmoid derivative cancels in the output delta.
. "$PSScriptRoot\_common.ps1"

Run-Training `
    -Label   "Sigmoid + CrossEntropy  |  LR=0.05  M=0.9  HL=300  epochs=30" `
    -LogFile "$ModelsDir\sigmoid_ce.log" `
    -Args    @(
        "-p", $DataPath,
        "-a", "sigmoid",
        "-c",
        "-r", "0.05",
        "-m", "0.9",
        "-e", "30",
        "-hl", "300",
        "-s", "$ModelsDir\sigmoid_ce.json"
    )
