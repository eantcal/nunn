#!/usr/bin/env bash
# ReLU hidden + Sigmoid output, MSE cost.
# ReLU avoids vanishing gradients in the hidden layers; needs a lower LR
# because the derivative is either 0 or 1 (no natural dampening like sigmoid).
#
# Options:
#   --no-matrix        Use MlpNN instead of MlpMatrixNN (Eigen)
#   --batch-size N     Mini-batch size when using MlpMatrixNN (default 32)
#   --opencl           Use ArrayFire/OpenCL GPU backend (implies MlpMatrixNN)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

no_matrix=0
opencl=0
batch_size=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-matrix)      no_matrix=1;         shift ;;
        --opencl)         opencl=1;             shift ;;
        --batch-size)     batch_size="$2";      shift 2 ;;
        --batch-size=*)   batch_size="${1#*=}"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if   (( opencl ));    then backend="MlpMatrixNN/OpenCL batch=$batch_size"; matrix_args=("-g" "-b" "$batch_size")
elif (( no_matrix )); then backend="MlpNN";                                matrix_args=()
else                       backend="MlpMatrixNN batch=$batch_size";         matrix_args=("-M" "-b" "$batch_size")
fi

save_args=()
if (( no_matrix && !opencl )); then save_args=("-s" "$MODELS_DIR/relu_mse.json"); fi

run_training \
    "ReLU + MSE  |  LR=0.01  M=0.9  HL=512  epochs=30  [$backend]" \
    "$MODELS_DIR/relu_mse.log" \
    -p "$DATA_PATH" -a relu -r 0.01 -m 0.9 -e 30 -hl 512 \
    "${save_args[@]}" "${matrix_args[@]}"
