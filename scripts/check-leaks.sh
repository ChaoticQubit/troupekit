#!/bin/sh
set -eu
dir="${1:-.}"
pattern='(^|/)(claude|agents)\.md$|(^|/)\.claude/|^docs/|^graphify-out/'
tracked="$(git -C "$dir" ls-files)"
leaks="$(printf '%s\n' "$tracked" | grep -iE "$pattern" || true)"
if [ -n "$leaks" ]; then
  echo "internal agent files are tracked; untrack them and keep them in .gitignore:" >&2
  echo "$leaks" >&2
  exit 1
fi
echo "leaks ok"
