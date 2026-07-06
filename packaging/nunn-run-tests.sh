#!/usr/bin/env sh

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nunn_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
nunn_bin="$nunn_root/bin"

PATH="$nunn_bin:$PATH"
export PATH
export NUNN_ROOT="$nunn_root"
export NUNN_BIN="$nunn_bin"

if [ ! -x "$nunn_bin/nunn_tests" ]; then
    echo "nunn_tests was not installed in $nunn_bin." >&2
    echo "Reinstall Nunn with the runtime component enabled." >&2
    exit 1
fi

cd "$nunn_bin" || exit 1
exec "$nunn_bin/nunn_tests" "$@"
