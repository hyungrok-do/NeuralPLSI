#!/bin/sh
# Builds the engine with the stricter stopping rule of the "converged" arm: L-BFGS
# stops when the objective falls by less than a relative 1e-9 over 20 iterations
# (the package's rule is 1e-5 over 10). Usage: build_converged.sh <library path>.
# Point NEURALPLSI_LIBRARY at the result to use it.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$root"/core/src/*.cpp "$work/"
sed -e 's/window=10;/window=20;/' -e 's/tolerance=1e-5;/tolerance=1e-9;/' "$root/core/src/neuralplsi.cpp" > "$work/neuralplsi.cpp"
grep -q 'tolerance=1e-9;' "$work/neuralplsi.cpp" || { echo "the stopping rule was not found in core/src/neuralplsi.cpp" >&2; exit 1; }
"${CXX:-c++}" -O3 -std=c++17 -shared -fPIC -ffp-contract=off -pthread -I "$root/core/include" \
  "$work/neuralplsi.cpp" "$work/spline.cpp" -o "$1"
echo "built $1"
