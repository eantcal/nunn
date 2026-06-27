#!/usr/bin/env bash
# Deep network: 3 hidden layers (784-512-256-128-10), Tanh + CrossEntropy.
# Tests whether depth helps on MNIST with this backprop implementation.
# Needs more epochs and a lower LR to avoid early divergence.
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
if (( no_matrix && !opencl )); then save_args=("-s" "$MODELS_DIR/deep_tanh_ce.json"); fi

run_training \
    "Deep Tanh + CrossEntropy  |  LR=0.02  M=0.9  HL=512+256+128  epochs=50  [$backend]" \
    "$MODELS_DIR/deep_tanh_ce.log" \
    -p "$DATA_PATH" -a tanh -c -r 0.02 -m 0.9 -e 50 -hl 512 -hl 256 -hl 128 \
    "${save_args[@]}" "${matrix_args[@]}"
