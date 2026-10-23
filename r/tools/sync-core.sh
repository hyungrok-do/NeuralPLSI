#!/bin/sh
# Copy the engine sources from the checkout's core/ into src/, which R CMD build
# needs to be self-contained. With --check, change nothing and fail if the copies
# differ from core/ or if DESCRIPTION, neuralplsi.h and _version.py disagree.
package=$(cd "$(dirname "$0")/.." && pwd)
root=$(dirname "$package")
if [ ! -f "$root/core/include/neuralplsi.h" ]; then
  echo "core/ not found next to $package; run this from a NeuralPLSI checkout" >&2
  exit 1
fi
status=0
for file in src/neuralplsi.cpp src/spline.cpp include/neuralplsi.h; do
  copy="$package/src/$(basename "$file")"
  if [ "$1" = "--check" ]; then
    if ! cmp -s "$root/core/$file" "$copy"; then
      echo "r/src/$(basename "$file") differs from core/$file" >&2
      status=1
    fi
  else
    cp "$root/core/$file" "$copy"
  fi
done
description=$(sed -n 's/^Version: *//p' "$package/DESCRIPTION")
header=$(sed -n 's/^#define NEURALPLSI_VERSION "\(.*\)"$/\1/p' "$root/core/include/neuralplsi.h")
python=$(sed -n "s/^__version__ = ['\"]\(.*\)['\"]$/\1/p" "$root/python/neuralplsi/_version.py")
if [ "$description" != "$header" ] || [ "$description" != "$python" ]; then
  echo "Versions differ: DESCRIPTION $description, neuralplsi.h $header, _version.py $python" >&2
  status=1
fi
exit $status
