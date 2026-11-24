#!/bin/sh
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
if grep -rniI --exclude-dir=target --exclude-dir=.git 'a[r]oh' "$root"; then
  echo "troupekit must not reference its host app" >&2
  exit 1
fi
echo "boundary ok"
