# ReLU hidden + Sigmoid output, MSE cost.
# ReLU avoids vanishing gradients in the hidden layers; needs a lower LR
# because the derivative is either 0 or 1 (no natural dampening like sigmoid).
#
# Flags:
#   -NoMatrix        Use MlpNN instead of MlpMatrixNN (Eigen)
#   -BatchSize <N>   Mini-batch size when using MlpMatrixNN (default 32)
param(
    [switch] $NoMatrix,
    [int]    $BatchSize = 32
)
. "$PSScriptRoot\_common.ps1"

$matrixArgs = if (-not $NoMatrix) { @("-M", "-b", "$BatchSize") } else { @() }
$saveArgs   = if ($NoMatrix)      { @("-s", "$ModelsDir\relu_mse.json") } else { @() }
$backend    = if ($NoMatrix)      { "MlpNN" } else { "MlpMatrixNN batch=$BatchSize" }

Run-Training `
    -Label   "ReLU + MSE  |  LR=0.01  M=0.9  HL=512  epochs=30  [$backend]" `
    -LogFile "$ModelsDir\relu_mse.log" `
    -CmdArgs (@(
        "-p", $DataPath,
        "-a", "relu",
        "-r", "0.01",
        "-m", "0.9",
        "-e", "30",
        "-hl", "512"
    ) + $saveArgs + $matrixArgs)
