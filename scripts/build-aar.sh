#!/bin/sh
set -eu

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

check=0
abis="arm64-v8a"
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    --emulator) abis="arm64-v8a x86_64" ;;
    *) echo "usage: build-aar.sh [--check] [--emulator]" >&2; exit 2 ;;
  esac
done

: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the Android NDK}"
api=26
host="$(uname -s | tr 'A-Z' 'a-z')-x86_64"
bin="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$host/bin"
test -x "$bin/llvm-strip"
module="bindings/kotlin/troupekit"
kotlin="$module/src/main/kotlin"
lib="libtroupekit_ffi.so"

triple_of() {
  case "$1" in
    arm64-v8a) echo "aarch64-linux-android" ;;
    x86_64) echo "x86_64-linux-android" ;;
    *) echo "unknown ABI $1" >&2; exit 2 ;;
  esac
}

for abi in $abis; do
  triple="$(triple_of "$abi")"
  upper="$(echo "$triple" | tr 'a-z-' 'A-Z_')"
  clang="$bin/$triple$api-clang"
  test -x "$clang"
  env "CARGO_TARGET_${upper}_LINKER=$clang" "CC_$(echo "$triple" | tr '-' '_')=$clang" \
    "AR_$(echo "$triple" | tr '-' '_')=$bin/llvm-ar" \
    cargo build -p troupekit-ffi --release --locked --target "$triple"
  mkdir -p "$module/src/main/jniLibs/$abi"
  cp "target/$triple/release/$lib" "$module/src/main/jniLibs/$abi/$lib"
  "$bin/llvm-strip" "$module/src/main/jniLibs/$abi/$lib"
done

work="target/aar-bindings"
rm -rf "$work"
first="$(triple_of "${abis%% *}")"
cargo run -p troupekit-ffi --features bindgen --release --locked --bin uniffi-bindgen -- \
  generate --library "target/$first/release/$lib" --language kotlin --out-dir "$work"
generated="$work/dev/troupekit/troupekit_ffi.kt"
test -f "$generated"

if [ "$check" = 1 ]; then
  if ! cmp -s "$generated" "$kotlin/dev/troupekit/troupekit_ffi.kt"; then
    echo "$kotlin/dev/troupekit/troupekit_ffi.kt is stale: rerun build-aar.sh" >&2
    exit 1
  fi
else
  mkdir -p "$kotlin/dev/troupekit"
  cp "$generated" "$kotlin/dev/troupekit/troupekit_ffi.kt"
fi

(cd bindings/kotlin && ./gradlew --no-daemon ${TROUPEKIT_GRADLE_FLAGS:-} :troupekit:assembleRelease)
find bindings/kotlin/troupekit/build/outputs/aar -name '*.aar'
