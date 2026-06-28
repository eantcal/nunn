#!/usr/bin/env bash
# Common paths and helpers for RNN scripts.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/examples"

# ANSI colour helpers (no-op when stdout is not a terminal)
_cyan()   { [ -t 1 ] && printf '\033[36m%s\033[0m' "$*" || printf '%s' "$*"; }
_yellow() { [ -t 1 ] && printf '\033[33m%s\033[0m' "$*" || printf '%s' "$*"; }
_green()  { [ -t 1 ] && printf '\033[32m%s\033[0m' "$*" || printf '%s' "$*"; }
_dgray()  { [ -t 1 ] && printf '\033[90m%s\033[0m' "$*" || printf '%s' "$*"; }
_bold()   { [ -t 1 ] && printf '\033[1m%s\033[0m'  "$*" || printf '%s' "$*"; }

# find_exe <name>
# Sets global EXE to the path of the Release (or Debug) binary.
find_exe() {
    local name="$1"
    local rel="$BUILD_DIR/$name/Release/$name"
    local dbg="$BUILD_DIR/$name/Debug/$name"
    # Windows: also try .exe extension
    [ -x "$rel.exe" ] && { EXE="$rel.exe"; return 0; }
    [ -x "$rel"     ] && { EXE="$rel";     return 0; }
    [ -x "$dbg.exe" ] && { EXE="$dbg.exe"; return 0; }
    [ -x "$dbg"     ] && { EXE="$dbg";     return 0; }
    echo "Error: $name not found. Build the project first." >&2
    exit 1
}

# run_example <label> <exe> [args...]
run_example() {
    local label="$1"; local exe="$2"; shift 2
    local banner
    banner=$(printf '=%.0s' {1..60})
    echo ""
    _cyan "$banner"; echo ""
    _yellow "  $label"; echo ""
    _dgray "  $exe $*"; echo ""
    _cyan "$banner"; echo ""
    echo ""
    local t0; t0=$(date +%s)
    "$exe" "$@"
    local elapsed=$(( $(date +%s) - t0 ))
    echo ""
    _green "Finished in $(printf '%02d:%02d:%02d' \
        $(( elapsed/3600 )) $(( (elapsed%3600)/60 )) $(( elapsed%60 )) )"
    echo ""
}
