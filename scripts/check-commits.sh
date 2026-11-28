#!/bin/sh
set -eu
[ "$#" -eq 2 ] || { echo "usage: sh scripts/check-commits.sh <base> <head>" >&2; exit 2; }
base="$1"
head="$2"
revert_pattern='^Revert ".+"$'
pattern='^(feat|fix|docs|chore|ci|test|refactor|perf|build|style|revert)(\([a-z0-9][a-z0-9._/,-]*\))?!?: [^ ]'
shas="$(git rev-list --no-merges "$base..$head")"
bad=0
for sha in $shas; do
  subject="$(git log -1 --format=%s "$sha")"
  if ! printf '%s\n' "$subject" | grep -qE "$pattern|$revert_pattern"; then
    echo "not a Conventional Commit: $(git rev-parse --short "$sha") $subject" >&2
    bad=$((bad + 1))
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "$bad commit subject(s) must read type(scope)!: subject or Revert \"...\", see CONTRIBUTING.md" >&2
  exit 1
fi
echo "commits ok"
