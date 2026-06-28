#!/usr/bin/env bash
# run_char.sh — train a character-level language model and generate text.
#
# Usage:
#   ./run_char.sh
#   ./run_char.sh --quick
#   ./run_char.sh --model gru --epochs 1200 --hidden 128
#   ./run_char.sh --model all
#
# Options:
#   --quick          150 epochs (fast smoke-test)
#   --model NAME     vanilla | gru | lstm | all  (default: all)
#   --epochs N       override epoch count
#   --hidden N       hidden units (default 64)
#   --gen-len N      characters to generate (default 120)
#   --temperature F  sampling temperature (default 0.8)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="all"
EPOCHS=0
HIDDEN=64
GEN_LEN=120
TEMPERATURE=0.8
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)          QUICK=1 ;;
        --model)          MODEL="$2"; shift ;;
        --epochs)         EPOCHS="$2"; shift ;;
        --hidden)         HIDDEN="$2"; shift ;;
        --gen-len)        GEN_LEN="$2"; shift ;;
        --temperature)    TEMPERATURE="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[ "$EPOCHS" -eq 0 ] && EPOCHS=$([ "$QUICK" -eq 1 ] && echo 150 || echo 800)

case "$MODEL" in
    all)     MODELS=("vanilla" "gru" "lstm") ;;
    vanilla) MODELS=("vanilla") ;;
    gru)     MODELS=("gru") ;;
    lstm)    MODELS=("lstm") ;;
    *) echo "Unknown model '$MODEL'. Use: vanilla | gru | lstm | all" >&2; exit 1 ;;
esac

find_exe rnn_char
CHAR_EXE="$EXE"

for m in "${MODELS[@]}"; do
    flag=""
    [ "$m" != "vanilla" ] && flag="--$m"

    label="rnn_char -- ${m^^}  epochs=$EPOCHS  hidden=$HIDDEN  temp=$TEMPERATURE"
    args=("$EPOCHS" "$HIDDEN" "$GEN_LEN" "$TEMPERATURE")
    [ -n "$flag" ] && run_example "$label" "$CHAR_EXE" "$flag" "${args[@]}" \
                   || run_example "$label" "$CHAR_EXE" "${args[@]}"
done
