#!/bin/sh
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
check="$here/check-leaks.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

repo() {
  dir="$work/$1"
  shift
  git init -q "$dir"
  for path in "$@"; do
    mkdir -p "$dir/$(dirname "$path")"
    echo x > "$dir/$path"
    git -C "$dir" add -f "$path"
  done
  echo "$dir"
}

clean="$(repo clean README.md \
  crates/troupekit/examples/support/docs/account-delete.md \
  crates/troupekit-graph/src/patterns/handoff.rs \
  crates/troupekit-context/tests/fixtures/load/notes.txt \
  website/docs/intro.md \
  scripts/notes-free.sh)"
sh "$check" "$clean" >/dev/null || fail "clean tree rejected"

n=0
for leak in .claude/settings.json CLAUDE.md AGENTS.md crates/troupekit-core/CLAUDE.md \
  crates/troupekit-core/.claude/settings.local.json docs/agent-decisions.md \
  docs/tasks/oss/brief-oss-01.md graphify-out/graph.json crates/x/Claude.md; do
  n=$((n + 1))
  dir="$(repo "leak$n" README.md "$leak")"
  if out="$(sh "$check" "$dir" 2>&1)"; then
    fail "leak passed: $leak"
  fi
  case "$out" in
    *"$leak"*) ;;
    *) fail "leak not named in output: $leak" ;;
  esac
done

sh "$check" "$here/.." >/dev/null || fail "this repository has a tracked internal file"
echo "check-leaks-test: PASS"
