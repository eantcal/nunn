#!/usr/bin/env bash
# run_adding.sh — run the adding problem benchmark (VanillaRnn vs GRU vs LSTM).
#
# rnn_adding already trains all three architectures and prints a side-by-side
# comparison; this script just sets convenient defaults and calls it.
#
# Usage:
#   ./run_adding.sh
#   ./run_adding.sh --quick
#   ./run_adding.sh --seq-len 30 --hidden 64 --epochs 800

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SEQ_LEN=20
HIDDEN=32
EPOCHS=0
LR=0.005
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)    QUICK=1 ;;
        --seq-len)  SEQ_LEN="$2"; shift ;;
        --hidden)   HIDDEN="$2"; shift ;;
        --epochs)   EPOCHS="$2"; shift ;;
        --lr)       LR="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[ "$EPOCHS" -eq 0 ] && EPOCHS=$([ "$QUICK" -eq 1 ] && echo 100 || echo 500)

find_exe rnn_adding

label="rnn_adding  seq_len=$SEQ_LEN  hidden=$HIDDEN  epochs=$EPOCHS  lr=$LR"
run_example "$label" "$EXE" "$SEQ_LEN" "$HIDDEN" "$EPOCHS" "$LR"
