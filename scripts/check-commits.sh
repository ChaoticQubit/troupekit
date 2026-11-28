#!/bin/sh
set -eu
[ "$#" -eq 2 ] || { echo "usage: sh scripts/check-commits.sh <base> <head>" >&2; exit 2; }
base="$1"
head="$2"
revert_pattern='^Revert ".+"$'
pattern='^(feat|fix|docs|chore|ci|test|refactor|perf|build|style|revert)(\([a-z0-9][a-z0-9._/,-]*\))?!?: [^ ]'
tools='(claude( code)?|anthropic|(github )?copilot|cursor( agent)?|codex|chatgpt|devin|aider|windsurf)'
trailer="^[[:space:]]*(co-authored-by|assisted-by|generated-by|co-developed-by):.*($tools([^a-z]|$)|@anthropic\\.com|@openai\\.com|@cursor\\.com)"
credit="(generated|written|authored|coded|drafted|assisted) (with|by|using|via) (the help of )?\\[?$tools([^a-z]|$)"
author="^$tools(\\[bot\\])? <|<[^>]*@(anthropic|openai|cursor)\\.com>|<[^>]*copilot[^>]*@"
shas="$(git rev-list --no-merges "$base..$head")"
bad=0
for sha in $shas; do
  short="$(git rev-parse --short "$sha")"
  subject="$(git log -1 --format=%s "$sha")"
  if ! printf '%s\n' "$subject" | grep -qE "$pattern|$revert_pattern"; then
    echo "not a Conventional Commit: $short $subject" >&2
    bad=$((bad + 1))
  fi
  lines="$(git log -1 --format=%B "$sha" | grep -iE "$trailer|$credit" || true)"
  if [ -n "$lines" ]; then
    printf 'tool attribution in %s: %s\n' "$short" "$lines" >&2
    bad=$((bad + 1))
  fi
  people="$(git log -1 --format='%an <%ae>%n%cn <%ce>' "$sha" | grep -iE "$author" || true)"
  if [ -n "$people" ]; then
    printf 'tool as author or committer in %s: %s\n' "$short" "$people" >&2
    bad=$((bad + 1))
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "$bad problem(s): subjects must read type(scope)!: subject or Revert \"...\", and messages must not credit a coding tool, see CONTRIBUTING.md" >&2
  exit 1
fi
echo "commits ok"
