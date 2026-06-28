#!/usr/bin/env bash
# run_all.sh — run all RNN examples and comparisons.
#
# Usage:
#   ./run_all.sh            # full run (may take several minutes)
#   ./run_all.sh --quick    # reduced epochs, fast smoke-test

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

BANNER=$(printf '#%.0s' {1..60})

echo ""
_bold "$BANNER"; echo ""
_bold "  nunn RNN benchmark suite"; echo ""
_bold "  Quick mode: $([ "$QUICK" -eq 1 ] && echo ON || echo OFF)"; echo ""
_bold "$BANNER"; echo ""

QUICK_FLAG=""
[ "$QUICK" -eq 1 ] && QUICK_FLAG="--quick"

# --- Sine-wave prediction ---
echo ""
_cyan "  [1/3]  Sine-wave prediction (VanillaRnn / GRU / LSTM)"; echo ""
"$SCRIPT_DIR/run_sine.sh" $QUICK_FLAG --model all

# --- Adding problem benchmark ---
echo ""
_cyan "  [2/3]  Adding problem benchmark (all models)"; echo ""
"$SCRIPT_DIR/run_adding.sh" $QUICK_FLAG

# --- Character-level language model ---
echo ""
_cyan "  [3/3]  Character-level language model (VanillaRnn / GRU / LSTM)"; echo ""
"$SCRIPT_DIR/run_char.sh" $QUICK_FLAG --model all

echo ""
_bold "$BANNER"; echo ""
_bold "  Done."; echo ""
_bold "$BANNER"; echo ""
