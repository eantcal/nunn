# LeakyReLU hidden + Sigmoid output, Cross-Entropy cost.
# Avoids the "dying ReLU" problem (neurons stuck at 0) because the negative
# slope (alpha=0.01) keeps a small gradient flowing for negative inputs.
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
$saveArgs   = if ($NoMatrix)      { @("-s", "$ModelsDir\leaky_relu_ce.json") } else { @() }
$backend    = if ($NoMatrix)      { "MlpNN" } else { "MlpMatrixNN batch=$BatchSize" }

Run-Training `
    -Label   "LeakyReLU + CrossEntropy  |  LR=0.01  M=0.9  HL=512  epochs=30  [$backend]" `
    -LogFile "$ModelsDir\leaky_relu_ce.log" `
    -CmdArgs (@(
        "-p", $DataPath,
        "-a", "leaky_relu",
        "-c",
        "-r", "0.01",
        "-m", "0.9",
        "-e", "30",
        "-hl", "512"
    ) + $saveArgs + $matrixArgs)
