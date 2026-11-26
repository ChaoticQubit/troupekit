#!/bin/sh
set -eu

TAG=v1.0.0
SHA256=3f5f750e36326aea4d65c7623e7d597a33e395d65f4d76e5e0532171f252532b
URL="https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz"

root=$(cd "$(dirname "$0")/.." && pwd)
dest="$root/crates/troupekit-local/llama.cpp"
pin="$TAG $SHA256"

if [ -f "$dest/.troupekit-pin" ] && [ "$(cat "$dest/.troupekit-pin")" = "$pin" ]; then
  echo "fetch-llama: llama.cpp $TAG already at $dest"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

curl --proto '=https' --tlsv1.2 -fsSL --max-filesize 104857600 -o "$tmp/src.tar.gz" "$URL"

if command -v sha256sum >/dev/null 2>&1; then
  got=$(sha256sum "$tmp/src.tar.gz" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$tmp/src.tar.gz" | cut -d' ' -f1)
fi
if [ "$got" != "$SHA256" ]; then
  echo "fetch-llama: SHA-256 mismatch for $URL: got $got, want $SHA256" >&2
  exit 1
fi

mkdir "$tmp/out"
tar -xzf "$tmp/src.tar.gz" -C "$tmp/out"
unpacked="$tmp/out/llama.cpp-${TAG#v}"
if [ ! -f "$unpacked/include/llama.h" ]; then
  echo "fetch-llama: $URL did not unpack to llama.cpp-${TAG#v}/" >&2
  exit 1
fi

rm -rf "$dest"
mv "$unpacked" "$dest"
printf '%s\n' "$pin" > "$dest/.troupekit-pin"
echo "fetch-llama: llama.cpp $TAG unpacked to $dest"
