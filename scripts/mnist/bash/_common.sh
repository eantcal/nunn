#!/usr/bin/env bash
# Common paths and helpers sourced by every training script.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXE_RELEASE="$REPO_ROOT/build/examples/mnist_test/mnist_test"
EXE_DEBUG="$REPO_ROOT/build/examples/mnist_test/Debug/mnist_test"
DATA_PATH="$REPO_ROOT/build/examples/mnist_test"
MODELS_DIR="$REPO_ROOT/scripts/mnist/models"

if [ -x "$EXE_RELEASE" ]; then
    EXE="$EXE_RELEASE"
elif [ -x "$EXE_DEBUG" ]; then
    EXE="$EXE_DEBUG"
else
    echo "Error: mnist_test not found. Build the project first." >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"

# ANSI colour helpers (no-op when stdout is not a terminal)
_cyan()    { [ -t 1 ] && printf '\033[36m%s\033[0m' "$*" || printf '%s' "$*"; }
_yellow()  { [ -t 1 ] && printf '\033[33m%s\033[0m' "$*" || printf '%s' "$*"; }
_green()   { [ -t 1 ] && printf '\033[32m%s\033[0m' "$*" || printf '%s' "$*"; }
_dgray()   { [ -t 1 ] && printf '\033[90m%s\033[0m' "$*" || printf '%s' "$*"; }

# run_training <label> <log_file> [exe_args...]
#   log_file may be empty ("") to skip logging to file.
run_training() {
    local label="$1"
    local log_file="$2"
    shift 2

    local banner
    banner=$(printf '=%.0s' {1..60})

    echo ""
    _cyan "$banner"; echo ""
    _yellow "  $label"; echo ""
    _dgray "  $EXE $*"; echo ""
    _cyan "$banner"; echo ""
    echo ""

    local start
    start=$(date +%s)

    if [ -n "$log_file" ]; then
        "$EXE" "$@" 2>&1 | tee "$log_file"
    else
        "$EXE" "$@"
    fi

    local elapsed=$(( $(date +%s) - start ))
    printf "\n"
    _green "Finished in $(printf '%02d:%02d:%02d' \
        $(( elapsed/3600 )) $(( (elapsed%3600)/60 )) $(( elapsed%60 )) )"
    echo ""
}
