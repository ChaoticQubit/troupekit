#!/bin/sh
set -eu
base="${1:-main}"
from="$(git merge-base "$base" HEAD)"
crates="$({ git diff --name-only "$from" -- crates; git ls-files --others --exclude-standard -- crates; } | cut -d/ -f2 | sort -u)"
if [ -z "$crates" ]; then
  echo "no crate changed since $base" >&2
  exit 3
fi
printf -- '-p %s ' $crates
