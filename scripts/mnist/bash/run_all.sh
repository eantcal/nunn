#!/usr/bin/env bash
# Run all MNIST training configurations in sequence and print a comparison table.
#
# Usage:
#   ./run_all.sh                       # all configs, MlpMatrixNN batch=32 (default)
#   ./run_all.sh --no-matrix           # all configs, classic MlpNN
#   ./run_all.sh --batch-size 64       # override batch size
#   ./run_all.sh --quick               # skip deep-network configs
#   ./run_all.sh --epochs 5            # override epoch count for all runs
#   ./run_all.sh --opencl              # use ArrayFire/OpenCL GPU backend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

quick=0
no_matrix=0
opencl=0
epochs_override=0
batch_size=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)          quick=1;                   shift ;;
        --no-matrix)      no_matrix=1;               shift ;;
        --opencl)         opencl=1;                   shift ;;
        --epochs)         epochs_override="$2";       shift 2 ;;
        --epochs=*)       epochs_override="${1#*=}";  shift ;;
        --batch-size)     batch_size="$2";            shift 2 ;;
        --batch-size=*)   batch_size="${1#*=}";       shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if   (( opencl ));    then backend="MlpMatrixNN/OpenCL batch=$batch_size"
elif (( no_matrix )); then backend="MlpNN"
else                       backend="MlpMatrixNN batch=$batch_size"
fi

# ---------------------------------------------------------------------------
# Config table
# Fields: label|act|ce(0/1)|lr|momentum|epochs|hl_csv|deep(0/1)
# ---------------------------------------------------------------------------
configs=(
    "Sigmoid + MSE       (baseline)|sigmoid|0|0.025|0.9|30|300|0"
    "Sigmoid + CE|sigmoid|1|0.05|0.9|30|300|0"
    "Tanh    + CE|tanh|1|0.05|0.9|30|512|0"
    "ReLU    + MSE|relu|0|0.01|0.9|30|512|0"
    "LeakyReLU + CE|leaky_relu|1|0.01|0.9|30|512|0"
    "Deep Tanh + CE     (512-256-128)|tanh|1|0.02|0.9|50|512,256,128|1"
    "Deep ReLU + CE     (512-256-128)|relu|1|0.005|0.9|50|512,256,128|1"
)

if (( quick )); then
    _yellow "Quick mode: skipping deep-network configs."; echo ""
    filtered=()
    for cfg in "${configs[@]}"; do
        deep="${cfg##*|}"
        if [[ "$deep" == "0" ]]; then filtered+=("$cfg"); fi
    done
    configs=("${filtered[@]}")
fi

_cyan "Backend: $backend"; echo ""

# ---------------------------------------------------------------------------
# Results accumulator (parallel arrays)
# ---------------------------------------------------------------------------
res_labels=()
res_costs=()
res_hl=()
res_epochs=()
res_ber=()
res_thru=()

for cfg in "${configs[@]}"; do
    IFS='|' read -r label act ce lr momentum epochs hl_csv deep <<< "$cfg"

    epoch_count=$(( epochs_override > 0 ? epochs_override : epochs ))
    cost_tag=$(( ce )) && cost_tag="ce" || cost_tag="mse"
    [[ "$ce" == "1" ]] && cost_tag="ce" || cost_tag="mse"
    hl_tag="${hl_csv//,/-}"

    if   (( opencl ));    then backend_tag="ocl_b${batch_size}"
    elif (( no_matrix )); then backend_tag="mlpnn"
    else                       backend_tag="mat_b${batch_size}"
    fi

    log_file="$MODELS_DIR/${act}_${cost_tag}_hl${hl_tag}_${backend_tag}.log"

    # Build argument list
    args=(-p "$DATA_PATH" -a "$act" -r "$lr" -m "$momentum" -e "$epoch_count")

    IFS=',' read -ra hl_arr <<< "$hl_csv"
    for hl in "${hl_arr[@]}"; do args+=(-hl "$hl"); done

    [[ "$ce" == "1" ]] && args+=(-c)

    if (( opencl )); then
        args+=(-g -b "$batch_size")
    elif (( no_matrix )); then
        model_file="$MODELS_DIR/${act}_${cost_tag}_hl${hl_tag}.json"
        args+=(-s "$model_file")
    else
        args+=(-M -b "$batch_size")
    fi

    run_training "$label  [$backend]" "$log_file" "${args[@]}"

    # Parse BER and throughput from log
    ber="N/A"
    thru="N/A"
    if [[ -f "$log_file" ]]; then
        ber_line=$(grep "^BER" "$log_file" | tail -1)
        if [[ "$ber_line" =~ ([0-9]+\.[0-9]+)% ]]; then
            ber="${BASH_REMATCH[1]}"
        fi

        thru_line=$(grep "^Throughput" "$log_file" | tail -1)
        if [[ "$thru_line" =~ ([0-9]+)[[:space:]]samples ]]; then
            thru="${BASH_REMATCH[1]}"
        fi
    fi

    res_labels+=("$label")
    res_costs+=("${cost_tag^^}")
    res_hl+=("$hl_tag")
    res_epochs+=("$epoch_count")
    res_ber+=("$ber")
    res_thru+=("$thru")
done

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
echo ""
_cyan "$(printf '=%.0s' {1..75})"; echo ""
_yellow "  MNIST Training Results  --  $backend"; echo ""
_cyan "$(printf '=%.0s' {1..75})"; echo ""
echo ""

# Header
printf "%-35s %-5s %-12s %-7s %-8s %-10s\n" \
    "Configuration" "Cost" "HiddenLayers" "Epochs" "BER(%)" "Thru(s/s)"
printf "%-35s %-5s %-12s %-7s %-8s %-10s\n" \
    "$(printf -- '-%.0s' {1..35})" "-----" "------------" "-------" "--------" "----------"

count=${#res_labels[@]}
for (( i=0; i<count; i++ )); do
    printf "%-35s %-5s %-12s %-7s %-8s %-10s\n" \
        "${res_labels[$i]}" "${res_costs[$i]}" "${res_hl[$i]}" \
        "${res_epochs[$i]}" "${res_ber[$i]}" "${res_thru[$i]}"
done

echo ""
_dgray "Logs saved to: $MODELS_DIR"; echo ""
