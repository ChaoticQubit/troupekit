#!/bin/sh
set -eu

ID=smollm2-135m-instruct-q2k
BYTES=88202080
SHA256=741ad12b64088fedc17c33aacb22e48be1972ef36a39f03666dd68bd15614fb9
URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q2_K.gguf"

if [ -n "${TROUPEKIT_MODEL_DIR:-}" ]; then
  dir="$TROUPEKIT_MODEL_DIR"
elif [ -n "${HOME:-}" ]; then
  dir="$HOME/.cache/troupekit/models"
else
  echo "fetch-test-model: set TROUPEKIT_MODEL_DIR or HOME" >&2
  exit 1
fi
dest="$dir/$ID-$SHA256.gguf"

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

if [ -f "$dest" ] && [ "$(sha_of "$dest")" = "$SHA256" ]; then
  echo "$dest"
  exit 0
fi

umask 077
mkdir -p "$dir"
part=$(mktemp "$dest.XXXXXX")
trap 'rm -f "$part"' EXIT
trap 'exit 1' INT TERM
curl --proto '=https' --tlsv1.2 -fsSL --max-filesize "$BYTES" -o "$part" "$URL"

got=$(sha_of "$part")
if [ "$got" != "$SHA256" ]; then
  echo "fetch-test-model: SHA-256 mismatch for $URL: got $got, want $SHA256" >&2
  exit 1
fi
mv "$part" "$dest"
trap - EXIT INT TERM
echo "$dest"
