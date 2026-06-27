# Tanh hidden + Sigmoid output, Cross-Entropy cost.
# Tanh centres activations around zero, which often speeds up early learning
# compared to Sigmoid.  Good default for single-hidden-layer experiments.
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

$matrixArgs = if ($OpenCL)            { @("-g", "-b", "$BatchSize") }
              elseif (-not $NoMatrix) { @("-M", "-b", "$BatchSize") }
              else                    { @() }
$saveArgs   = if ($NoMatrix -and -not $OpenCL) { @("-s", "$ModelsDir\tanh_ce.json") } else { @() }
$backend    = if ($OpenCL)     { "MlpMatrixNN/OpenCL batch=$BatchSize" }
              elseif ($NoMatrix){ "MlpNN" }
              else              { "MlpMatrixNN batch=$BatchSize" }

Run-Training `
    -Label   "Tanh + CrossEntropy  |  LR=0.05  M=0.9  HL=512  epochs=30  [$backend]" `
    -LogFile "$ModelsDir\tanh_ce.log" `
    -CmdArgs (@(
        "-p", $DataPath,
        "-a", "tanh",
        "-c",
        "-r", "0.05",
        "-m", "0.9",
        "-e", "30",
        "-hl", "512"
    ) + $saveArgs + $matrixArgs)
