#!/bin/sh
set -eu

dry=0
label="${SIZE_LABEL:-unlabelled}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry=1 ;;
    --label=*) label="${arg#--label=}" ;;
    *) echo "usage: size-report.sh [--dry-run] [--label=quiet|loaded]" >&2; exit 2 ;;
  esac
done

rows="a: b:--no-default-features+--features+sqlite c:--no-default-features+--features+sqlite,llama"
targets="aarch64-apple-ios aarch64-linux-android"

artifact() {
  case "$1" in
    aarch64-apple-ios) echo "libtroupekit_ffi.a" ;;
    aarch64-linux-android) echo "libtroupekit_ffi.so" ;;
    *) echo "unknown target $1" >&2; exit 2 ;;
  esac
}

strip_tool() {
  case "$1" in
    aarch64-apple-ios) echo "xcrun strip -S" ;;
    aarch64-linux-android) echo "${ANDROID_NDK_HOME:?ANDROID_NDK_HOME unset}/toolchains/llvm/prebuilt/$(uname -s | tr 'A-Z' 'a-z')-x86_64/bin/llvm-strip" ;;
  esac
}

has_llama() {
  grep -q '^llama' crates/troupekit-ffi/Cargo.toml
}

builds=0
echo "| Target | Row | Stripped bytes | Build seconds | Condition |"
echo "|---|---|---|---|---|"
for target in $targets; do
  for row in $rows; do
    name="${row%%:*}"
    flags="$(echo "${row#*:}" | tr '+' ' ')"
    dir="target/size/$target-$name"
    cmd="cargo build -p troupekit-ffi --release --locked --target $target $flags"
    builds=$((builds + 1))
    if [ "$dry" = 1 ]; then
      echo "build $target $name: CARGO_TARGET_DIR=$dir $cmd"
      continue
    fi
    if [ "$name" = c ] && ! has_llama; then
      echo "| $target | ($name) | missing: feature llama not in troupekit-ffi (0.3) | | $label |"
      continue
    fi
    if [ "$target" = aarch64-linux-android ] && [ -z "${CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER:-}" ]; then
      echo "set CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER to the NDK clang (API 26) first" >&2
      exit 2
    fi
    start="$(date +%s)"
    RUSTC_WRAPPER= CARGO_TARGET_DIR="$dir" $cmd
    end="$(date +%s)"
    file="$dir/$target/release/$(artifact "$target")"
    cp "$file" "$file.stripped"
    $(strip_tool "$target") "$file.stripped"
    bytes="$(wc -c < "$file.stripped" | tr -d ' ')"
    echo "| $target | ($name) | $bytes | $((end - start)) | $label |"
  done
done
if [ "$builds" != 6 ]; then
  echo "size-report.sh covered $builds builds, expected 6" >&2
  exit 1
fi
