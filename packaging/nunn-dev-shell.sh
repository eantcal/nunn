#!/usr/bin/env sh

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nunn_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
nunn_bin="$nunn_root/bin"

PATH="$nunn_bin:$PATH"
export PATH
export NUNN_ROOT="$nunn_root"
export NUNN_BIN="$nunn_bin"

echo "Nunn developer shell"
echo "Install root: $NUNN_ROOT"
echo
echo "Common commands:"
echo "  nunn_tests"
echo "  mnist_test"
echo "  xor_test"
echo "  tictactoe"
echo "  net2json"
echo
echo "Installed executables:"
find "$NUNN_BIN" -maxdepth 1 -type f -perm -111 -printf '  %f\n' 2>/dev/null || ls "$NUNN_BIN"
echo

if [ -n "$SHELL" ] && [ -x "$SHELL" ]; then
    exec "$SHELL"
fi

exec /bin/sh
