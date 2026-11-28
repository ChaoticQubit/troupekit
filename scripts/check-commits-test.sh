#!/bin/sh
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
check="$here/check-commits.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

repo="$work/repo"
git init -q -b main "$repo"
g() { git -C "$repo" -c user.name=t -c user.email=t@example.com "$@"; }
run() { (cd "$repo" && sh "$check" "$@"); }
c() { g commit -q --allow-empty -m "$1"; }
c "chore: root"
base="$(g rev-parse HEAD)"

for subject in "feat: add a tool" "fix(store-sqlite): close the file" "fix(local,kit): two scopes" "docs(readme)!: drop the old API" \
  "chore(0.3): bump" "ci: pin the runner" "test: cover the queue" "refactor(kit): split the builder" \
  "perf: skip a copy" "build: cargo profile" "style: rustfmt" "revert: feat: add a tool"; do
  c "$subject"
done
g checkout -q -b side
c "feat: on a side branch"
g checkout -q main
g merge -q --no-ff -m "Merge pull request #1 from someone/side" side
run "$base" HEAD >/dev/null 2>&1 || fail "valid range rejected: $(run "$base" HEAD 2>&1)"

for subject in "Add a tool" "feat:no space" "feat add a tool" "feature: add" "Feat: add" \
  "feat(): empty scope" "feat(kit) : space" "fixup! feat: add a tool" "wip" "feat:"; do
  good="$(g rev-parse HEAD)"
  c "$subject"
  if out="$(run "$base" HEAD 2>&1)"; then
    fail "bad subject passed: $subject"
  fi
  case "$out" in
    *"$subject"*) ;;
    *) fail "bad subject not named: $subject" ;;
  esac
  g reset -q --hard "$good"
done

if run "$base" no-such-ref >/dev/null 2>&1; then
  fail "unknown ref passed"
fi
if run >/dev/null 2>&1; then
  fail "missing arguments passed"
fi
echo "check-commits-test: PASS"
