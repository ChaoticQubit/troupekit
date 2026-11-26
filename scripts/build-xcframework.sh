#!/bin/sh
set -eu

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

check=0
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    *) echo "usage: build-xcframework.sh [--check]" >&2; exit 2 ;;
  esac
done

targets="aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin"
out="bindings/swift/TroupekitFFI.xcframework"
sources="bindings/swift/Sources/Troupekit"
work="target/xcframework"
lib="libtroupekit_ffi.a"
export IPHONEOS_DEPLOYMENT_TARGET=16.0
export MACOSX_DEPLOYMENT_TARGET=13.0
grep -q '\.iOS(\.v16), \.macOS(\.v13)' Package.swift

for target in $targets; do
  cargo build -p troupekit-ffi --release --locked --target "$target"
  strip -S "target/$target/release/$lib"
done

rm -rf "$work"
mkdir -p "$work/headers" "$work/swift"
cargo run -p troupekit-ffi --features bindgen --release --locked --bin uniffi-bindgen -- \
  generate --library "target/aarch64-apple-darwin/release/$lib" \
  --language swift --out-dir "$work/swift"

test -f "$work/swift/Troupekit.swift"
test -f "$work/swift/troupekitFFI.h"
mv "$work/swift/troupekitFFI.h" "$work/headers/"
mv "$work/swift/troupekitFFI.modulemap" "$work/headers/module.modulemap"

if [ "$check" = 1 ]; then
  if ! cmp -s "$work/swift/Troupekit.swift" "$sources/Troupekit.swift"; then
    echo "bindings/swift/Sources/Troupekit/Troupekit.swift is stale: rerun build-xcframework.sh" >&2
    exit 1
  fi
else
  cp "$work/swift/Troupekit.swift" "$sources/Troupekit.swift"
fi

rm -rf "$out"
set --
for target in $targets; do
  set -- "$@" -library "target/$target/release/$lib" -headers "$work/headers"
done
xcodebuild -create-xcframework "$@" -output "$out"

count="$(find "$out" -name "$lib" | wc -l | tr -d ' ')"
if [ "$count" != 3 ]; then
  echo "expected 3 static libraries in $out, found $count" >&2
  exit 1
fi
echo "$out"
