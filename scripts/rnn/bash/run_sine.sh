#!/usr/bin/env bash
# run_sine.sh — train and compare VanillaRnn / GRU / LSTM on sine-wave prediction.
#
# Usage:
#   ./run_sine.sh
#   ./run_sine.sh --quick
#   ./run_sine.sh --model gru --epochs 2000 --hidden 64 --lr 0.003
#   ./run_sine.sh --model all
#
# Options:
#   --quick          400 epochs (fast smoke-test)
#   --model NAME     vanilla | gru | lstm | all  (default: all)
#   --epochs N       override epoch count
#   --hidden N       hidden units (default 32)
#   --lr F           learning rate (default 0.005)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="all"
EPOCHS=0
HIDDEN=32
LR=0.005
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)          QUICK=1 ;;
        --model)          MODEL="$2"; shift ;;
        --epochs)         EPOCHS="$2"; shift ;;
        --hidden)         HIDDEN="$2"; shift ;;
        --lr)             LR="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[ "$EPOCHS" -eq 0 ] && EPOCHS=$([ "$QUICK" -eq 1 ] && echo 400 || echo 1500)

case "$MODEL" in
    all)     MODELS=("vanilla" "gru" "lstm") ;;
    vanilla) MODELS=("vanilla") ;;
    gru)     MODELS=("gru") ;;
    lstm)    MODELS=("lstm") ;;
    *) echo "Unknown model '$MODEL'. Use: vanilla | gru | lstm | all" >&2; exit 1 ;;
esac

find_exe rnn_sine
SINE_EXE="$EXE"

# Collect results for comparison table
declare -a RES_MODEL RES_MAE RES_TIME

for m in "${MODELS[@]}"; do
    flag=""
    [ "$m" != "vanilla" ] && flag="--$m"

    label="rnn_sine -- ${m^^}  epochs=$EPOCHS  hidden=$HIDDEN  lr=$LR"
    echo ""
    _cyan "$(printf '=%.0s' {1..60})"; echo ""
    _yellow "  $label"; echo ""
    _dgray "  $SINE_EXE $flag $EPOCHS $HIDDEN $LR"; echo ""
    _cyan "$(printf '=%.0s' {1..60})"; echo ""
    echo ""

    t0=$(date +%s)
    if [ -n "$flag" ]; then
        output=$("$SINE_EXE" $flag "$EPOCHS" "$HIDDEN" "$LR" 2>&1)
    else
        output=$("$SINE_EXE" "$EPOCHS" "$HIDDEN" "$LR" 2>&1)
    fi
    printf '%s\n' "$output"
    elapsed=$(( $(date +%s) - t0 ))

    # Extract Max absolute error (autoregressive)
    mae="n/a"
    if [[ "$output" =~ Max\ absolute\ error.*:\ ([0-9.]+) ]]; then
        mae="${BASH_REMATCH[1]}"
    fi

    _green "Finished in $(printf '%02d:%02d' $(( elapsed/60 )) $(( elapsed%60 )) )"; echo ""

    RES_MODEL+=("${m^^}")
    RES_MAE+=("$mae")
    RES_TIME+=("$(printf '%02d:%02d' $(( elapsed/60 )) $(( elapsed%60 )) )")
done

# Print comparison table
if [ "${#MODELS[@]}" -gt 1 ]; then
    echo ""
    _cyan "$(printf '=%.0s' {1..60})"; echo ""
    _cyan "  Comparison  (epochs=$EPOCHS  hidden=$HIDDEN  lr=$LR)"; echo ""
    _cyan "$(printf '=%.0s' {1..60})"; echo ""
    printf "\n%-12s  %-14s  %-8s\n" "Model" "Max AE (autoreg)" "Time"
    printf "%-12s  %-14s  %-8s\n" "------------" "--------------" "--------"
    for i in "${!RES_MODEL[@]}"; do
        printf "%-12s  %-14s  %-8s\n" "${RES_MODEL[$i]}" "${RES_MAE[$i]}" "${RES_TIME[$i]}"
    done
    echo ""
fi
