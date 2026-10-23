#!/bin/sh
# Compiles the C++ engine to WebAssembly and assembles the static site in web/dist.
set -eu
web=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$web")
dist=$web/dist

# Homebrew's emscripten bundles its own LLVM and Binaryen but writes no config
# file; point a local one at them instead of editing ~/.emscripten.
if ! em++ --version >/dev/null 2>&1 && command -v brew >/dev/null 2>&1 &&
   prefix=$(brew --prefix emscripten 2>/dev/null) && [ -d "$prefix/libexec/llvm/bin" ]; then
  mkdir -p "$web/.cache"
  printf "LLVM_ROOT = '%s'\nBINARYEN_ROOT = '%s'\nNODE_JS = '%s'\n" \
    "$prefix/libexec/llvm/bin" "$prefix/libexec/binaryen" "$(command -v node)" > "$web/.cache/emscripten.py"
  export EM_CONFIG="$web/.cache/emscripten.py"
fi
em++ --version | head -n 1

# Every function the header declares, plus malloc and free.
exports=$(grep -oE '(nplsi|splsi)_[a-z_]+\(' "$root/core/include/neuralplsi.h" | tr -d '(' | sort -u |
          sed 's/^/_/' | tr '\n' ',')_malloc,_free

rm -rf "$dist"
mkdir -p "$dist"
# The engine's native flags. No pthreads: GitHub Pages cannot send the headers
# SharedArrayBuffer needs, so the page runs one single-threaded instance per Web
# Worker and calls the engine with n_jobs = 1.
em++ -O3 -std=c++17 -ffp-contract=off -fwasm-exceptions --no-entry \
  -I "$root/core/include" "$root/core/src/neuralplsi.cpp" "$root/core/src/spline.cpp" \
  -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createNeuralPLSI -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH -sEXPORTED_FUNCTIONS="$exports" \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPF32,HEAPF64,UTF8ToString \
  -o "$dist/neuralplsi.js"

cp "$web"/src/*.html "$web"/src/*.css "$web"/src/*.js "$web"/src/*.svg "$dist/"
cp "$root/NHANES/12940_2020_644_MOESM2_ESM.csv" "$dist/nhanes.csv"
# The service worker caches every file; a new cache name per build drops stale ones.
rm "$dist/sw.js"
files=$(cd "$dist" && ls | sed 's/.*/"&"/' | tr '\n' ',')
digest() { if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi; }
build=$(cd "$dist" && cat $(ls) | digest | cut -c1-12)
sed -e "s/__BUILD__/$build/" -e "s/__FILES__/[\".\/\",${files%,}]/" "$web/src/sw.js" > "$dist/sw.js"
echo "Built $dist ($build)"
