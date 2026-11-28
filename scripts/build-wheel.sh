#!/bin/sh
set -eu

out="${1:-dist}"
python="${PYTHON:-python3}"
maturin_version="1.16.0"
compat="${WHEEL_COMPAT:-}"

"$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "build-wheel.sh needs Python 3.11 or newer, got $("$python" --version 2>&1)" >&2
  exit 2
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
venv="$work/build-venv"
built="$work/wheel"
"$python" -m venv "$venv"
if [ -x "$venv/bin/python" ]; then
  vpy="$venv/bin/python"
else
  vpy="$venv/Scripts/python.exe"
fi
"$vpy" -m pip install --quiet "maturin==$maturin_version"

set -- build --release --strip --locked \
  --manifest-path crates/troupekit-py/Cargo.toml \
  --interpreter "$vpy" \
  --out "$built"
if [ -n "$compat" ]; then
  set -- "$@" --compatibility "$compat"
fi
"$vpy" -m maturin "$@"

count="$(find "$built" -maxdepth 1 -name 'troupekit-*-abi3-*.whl' | wc -l | tr -d ' ')"
if [ "$count" != "1" ]; then
  echo "expected exactly one abi3 wheel from this build, found $count" >&2
  exit 1
fi
wheel="$(find "$built" -maxdepth 1 -name 'troupekit-*-abi3-*.whl')"
mkdir -p "$out"
cp "$wheel" "$out/"
test -f "$out/$(basename "$wheel")"
echo "$out/$(basename "$wheel")"
