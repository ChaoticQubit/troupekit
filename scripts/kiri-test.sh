#!/bin/sh
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
target="troupekit-target-$(basename "$root")"
tag="troupekit-test:$(shasum -a 256 "$root/Dockerfile.test" | cut -c1-12)"
if ! kiri image inspect "$tag" >/dev/null 2>&1; then
  kiri build -q -f "$root/Dockerfile.test" -t "$tag" "$root" >/dev/null
fi
exec kiri run --rm \
  -v "$root":/src \
  -v troupekit-cargo-registry:/usr/local/cargo/registry \
  -v "$target":/target \
  "$tag" "$@"
