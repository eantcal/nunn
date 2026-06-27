# Baseline: Sigmoid hidden + Sigmoid output, MSE cost.
# Classic configuration; typically reaches ~97% accuracy in 30 epochs.
#
# Flags:
#   -NoMatrix        Use MlpNN instead of MlpMatrixNN (Eigen)
#   -BatchSize <N>   Mini-batch size when using MlpMatrixNN (default 32)
#   -OpenCL          Use ArrayFire/OpenCL GPU backend (implies MlpMatrixNN)
param(
    [switch] $NoMatrix,
    [int]    $BatchSize = 32,
    [switch] $OpenCL
)
. "$PSScriptRoot\_common.ps1"

$matrixArgs = if ($OpenCL)       { @("-g", "-b", "$BatchSize") }
              elseif (-not $NoMatrix) { @("-M", "-b", "$BatchSize") }
              else                { @() }
$saveArgs   = if ($NoMatrix -and -not $OpenCL) { @("-s", "$ModelsDir\sigmoid_mse.json") } else { @() }
$backend    = if ($OpenCL)       { "MlpMatrixNN/OpenCL batch=$BatchSize" }
              elseif ($NoMatrix)  { "MlpNN" }
              else                { "MlpMatrixNN batch=$BatchSize" }

Run-Training `
    -Label   "Sigmoid + MSE  |  LR=0.025  M=0.9  HL=300  epochs=30  [$backend]" `
    -LogFile "$ModelsDir\sigmoid_mse.log" `
    -CmdArgs (@(
        "-p", $DataPath,
        "-a", "sigmoid",
        "-r", "0.025",
        "-m", "0.9",
        "-e", "30",
        "-hl", "300"
    ) + $saveArgs + $matrixArgs)
