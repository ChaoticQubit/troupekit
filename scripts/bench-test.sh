#!/bin/sh
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
bench="$here/bench.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

mkdir -p "$work/bin"
cat > "$work/bin/cargo" <<'STUB'
#!/bin/sh
sleep "${STUB_SLEEP:-0.2}"
mkdir -p target
echo "$*" >> target/stub.log
[ "${STUB_FAIL:-0}" = 1 ] && exit 1
exit 0
STUB
chmod +x "$work/bin/cargo"
PATH="$work/bin:$PATH"
export PATH

trees=""
for n in 1 2 3; do
  t="$work/tree-$n"
  mkdir -p "$t/crates/troupekit-core/src"
  printf 'pub mod clock;\n' > "$t/crates/troupekit-core/src/lib.rs"
  git -C "$t" init -q
  git -C "$t" -c user.email=t@t -c user.name=t add .
  git -C "$t" -c user.email=t@t -c user.name=t commit -qm init
  cp "$t/crates/troupekit-core/src/lib.rs" "$work/lib-$n.orig"
  trees="$trees $t"
done
BENCH_TREES="${trees# }"
BENCH_OUT="$work/bench.csv"
export BENCH_TREES BENCH_OUT

same_libs() {
  for n in 1 2 3; do
    cmp -s "$work/tree-$n/crates/troupekit-core/src/lib.rs" "$work/lib-$n.orig" || fail "lib.rs changed in tree-$n ($1)"
  done
}
rows() { if [ -f "$BENCH_OUT" ]; then tail -n +2 "$BENCH_OUT" | grep -c "^$1," || true; else echo 0; fi; }

sh "$bench" t-best incr best 1 >/dev/null 2>&1 || fail "best run exited non-zero"
[ "$(rows t-best)" = 1 ] || fail "best wrote $(rows t-best) rows, want 1"
same_libs best

sh "$bench" t-worst test worst 1 >/dev/null 2>&1 || fail "worst run exited non-zero"
[ "$(rows t-worst)" = 3 ] || fail "worst wrote $(rows t-worst) rows, want 3"
same_libs worst

head -1 "$BENCH_OUT" | grep -qx 'label,scenario,condition,run,tree,wall_s,peak_rss_kb,target_kb,load1_before,load1_after,jobs,wrapper,commit,iso_time,incremental' || fail "bad header"
awk -F, 'NR > 1 && NF != 15 { exit 1 }' "$BENCH_OUT" || fail "row without 15 fields"
awk -F, 'NR > 1 && !($6 >= 0.2) { exit 1 }' "$BENCH_OUT" || fail "wall_s below 0.2"
awk -F, 'NR > 1 && !($8 > 0) { exit 1 }' "$BENCH_OUT" || fail "target_kb not positive"
CARGO_BUILD_JOBS=4 sh "$bench" t-jobs check best 1 >/dev/null 2>&1 || fail "jobs run exited non-zero"
grep -q '^t-jobs,check,best,1,tree-1,.*,4,,' "$BENCH_OUT" || fail "CARGO_BUILD_JOBS not recorded"

if STUB_FAIL=1 sh "$bench" t-fail incr worst 1 >/dev/null 2>&1; then fail "failed cargo did not fail the run"; fi
[ "$(rows t-fail)" = 0 ] || fail "failed run wrote rows"
same_libs failure

STUB_SLEEP=5 sh "$bench" t-int incr worst 1 >/dev/null 2>&1 &
pid=$!
sleep 7
kill -TERM "$pid"
wait "$pid" 2>/dev/null && fail "interrupted run exited zero"
[ "$(rows t-int)" = 0 ] || fail "interrupted run wrote rows"
same_libs interrupt

if sh "$bench" t-bad nope best 1 >/dev/null 2>&1; then fail "unknown scenario accepted"; fi

known="$work/known.csv"
head -1 "$BENCH_OUT" > "$known"
for r in "1,a,1.0" "2,a,5.0" "3,a,3.0" "1,b,9.0" "1,c,2.0" "2,b,1.0" "2,c,4.0" "3,b,7.0" "3,c,8.0"; do
  run="${r%%,*}"; rest="${r#*,}"; tree="${rest%%,*}"; wall="${rest#*,}"
  cond=worst; [ "$tree" = a ] && cond=best
  echo "k,incr,$cond,$run,$tree,$wall,100,200,1.5,1.6,,,abc,2026-01-01T00:00:00Z," >> "$known"
done
out="$(sh "$bench" report "$known")"
echo "$out" | grep -Eq '^k +incr +best +3 +3\.00 +1\.00 +5\.00 ' || fail "best median wrong: $out"
echo "$out" | grep -Eq '^k +incr +worst +3 +8\.00 +4\.00 +9\.00 ' || fail "worst median wrong: $out"

echo "bench-test: PASS"
