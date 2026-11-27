#!/bin/sh
set -eu

ID=smollm2-135m-instruct-q2k
BYTES=88202080
SHA256=741ad12b64088fedc17c33aacb22e48be1972ef36a39f03666dd68bd15614fb9
URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q2_K.gguf"

case "${1:-chat}" in
  chat) ;;
  embed)
    ID=all-minilm-l6-v2-q8-0
    BYTES=25008064
    SHA256=263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed
    URL="https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q8_0.gguf"
    ;;
  *)
    echo "fetch-test-model: unknown model $1, want chat or embed" >&2
    exit 2
    ;;
esac

if [ -n "${TROUPEKIT_MODEL_DIR:-}" ]; then
  dir="$TROUPEKIT_MODEL_DIR"
elif [ -n "${HOME:-}" ]; then
  dir="$HOME/.cache/troupekit/models"
else
  echo "fetch-test-model: set TROUPEKIT_MODEL_DIR or HOME" >&2
  exit 1
fi
dest="$dir/$ID.gguf"

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
part="$dest.part"
trap 'rm -f "$part"' EXIT INT TERM
curl --proto '=https' --tlsv1.2 -fsSL --max-filesize "$BYTES" -o "$part" "$URL"

got=$(sha_of "$part")
if [ "$got" != "$SHA256" ]; then
  echo "fetch-test-model: SHA-256 mismatch for $URL: got $got, want $SHA256" >&2
  exit 1
fi
mv "$part" "$dest"
trap - EXIT INT TERM
echo "$dest"
