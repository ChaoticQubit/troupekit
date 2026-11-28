#!/bin/sh
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
lib=crates/troupekit-core/src/lib.rs
out="${BENCH_OUT:-$root/docs/tasks/build-speed/raw/bench.csv}"
header=label,scenario,condition,run,tree,wall_s,peak_rss_kb,target_kb,load1_before,load1_after,jobs,wrapper,commit,iso_time,incremental
trees="${BENCH_TREES:-}"

usage() {
  echo "usage: sh scripts/bench.sh <label> <clean|incr|check|test> <best|worst> <run>" >&2
  echo "       sh scripts/bench.sh report [csv]" >&2
  echo "       sh scripts/bench.sh fetch" >&2
  exit 2
}

report() {
  [ -f "$1" ] || { echo "no csv: $1" >&2; exit 2; }
  awk -F, '
    function median(list, n,   a, i, j, k, v) {
      n = split(list, a, " ")
      for (i = 2; i <= n; i++) { v = a[i]; for (j = i - 1; j >= 1 && a[j] + 0 > v + 0; j--) a[j + 1] = a[j]; a[j + 1] = v }
      lo = a[1]; hi = a[n]
      if (n % 2) return a[(n + 1) / 2]
      return (a[n / 2] + a[n / 2 + 1]) / 2
    }
    NR == 1 { next }
    {
      key = $1 FS $2 FS $3
      run = key FS $4
      if (!(key in seen)) { seen[key] = 1; order[++keys] = key }
      if (!(run in wall)) { runs[key] = runs[key] " " run; wall[run] = $6 }
      else if ($6 + 0 > wall[run] + 0) wall[run] = $6
      tkb[key] = tkb[key] " " $8
      load[key] = load[key] " " $9
    }
    END {
      printf "%-8s %-6s %-6s %2s %8s %8s %8s %12s %6s\n", "label", "scen", "cond", "N", "median_s", "min_s", "max_s", "target_kb", "load1"
      for (k = 1; k <= keys; k++) {
        key = order[k]
        n = split(substr(runs[key], 2), r, " ")
        vals = ""
        for (i = 1; i <= n; i++) vals = vals " " wall[r[i]]
        m = median(vals); mlo = lo; mhi = hi
        split(key, f, FS)
        printf "%-8s %-6s %-6s %2d %8.2f %8.2f %8.2f %12d %6.2f\n", f[1], f[2], f[3], n, m, mlo, mhi, median(tkb[key]), median(load[key])
      }
    }' "$1"
}

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }

case "${1:-}" in
  report) report "${2:-$out}"; exit 0 ;;
esac
[ -n "$trees" ] || { echo "set BENCH_TREES to the work trees to build" >&2; exit 2; }
case "${1:-}" in
  fetch) for t in $trees; do (cd "$t" && cargo fetch --locked); done; exit 0 ;;
esac
[ "$#" -eq 4 ] || usage
label="$1"; scenario="$2"; condition="$3"; run="$4"
case "$label" in ''|*[!a-z0-9-]*) echo "bad label: $label" >&2; exit 2 ;; esac
case "$scenario" in clean|incr|check|test) ;; *) echo "bad scenario: $scenario" >&2; exit 2 ;; esac
case "$run" in ''|*[!0-9]*) echo "bad run: $run" >&2; exit 2 ;; esac
case "$condition" in
  best) set -- $trees; sel="$1" ;;
  worst) sel="$trees" ;;
  *) echo "bad condition: $condition" >&2; exit 2 ;;
esac
for t in $sel; do [ -f "$t/$lib" ] || { echo "not a tree: $t" >&2; exit 2; }; done
for t in $sel; do git -C "$t" diff --quiet HEAD -- "$lib" || { echo "tree has local edits to $lib: $t" >&2; exit 2; }; done

case "$scenario" in
  clean|incr) build="cargo build --workspace --all-targets --locked" ;;
  check) build="cargo check --workspace --all-targets --locked" ;;
  test) build="cargo test --workspace --no-run --locked" ;;
esac
if [ "$scenario" = clean ]; then prime="cargo clean"; else prime="$build"; fi

tmp="$(mktemp -d)"
pids=""
restore() { for t in $sel; do git -C "$t" checkout -q -- "$lib" 2>/dev/null || true; done; }
finish() { status=$?; restore; rm -rf "$tmp"; exit "$status"; }
interrupt() {
  trap - EXIT INT TERM
  for p in $pids; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  restore
  rm -rf "$tmp"
  exit 130
}
trap finish EXIT
trap interrupt INT TERM
set -m

phase() {
  pids=""
  i=0
  for t in $sel; do
    i=$((i + 1))
    ( cd "$t" && BENCH_TIME="$tmp/time.$i" && export BENCH_TIME && eval "$1" ) > "$tmp/$2.$i" 2>&1 &
    pids="$pids $!"
  done
  failed=0
  i=0
  for p in $pids; do
    i=$((i + 1))
    if ! wait "$p"; then failed=1; echo "$2 failed in tree $i:" >&2; tail -20 "$tmp/$2.$i" >&2; fi
  done
  pids=""
  [ "$failed" = 0 ]
}

phase "git checkout -q -- $lib && $prime" prime
before="$(load1)"
if [ "$scenario" = clean ]; then
  measured="/usr/bin/time -l -o \"\$BENCH_TIME\" $build"
else
  measured="printf 'pub const BENCH_EDIT: u64 = %s;\n' \"\$(perl -MTime::HiRes=time -e 'printf \"%d\", time * 1e9')\" >> $lib && /usr/bin/time -l -o \"\$BENCH_TIME\" $build"
fi
phase "$measured" log
after="$(load1)"

mkdir -p "$(dirname "$out")"
[ -f "$out" ] || echo "$header" > "$out"
now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rows=""
i=0
for t in $sel; do
  i=$((i + 1))
  wall="$(awk '/ real /{print $1; exit}' "$tmp/time.$i")"
  rss="$(awk '/maximum resident set size/{printf "%d", $1 / 1024; exit}' "$tmp/time.$i")"
  [ -n "$wall" ] && [ -n "$rss" ] || { echo "no timing for tree $i" >&2; exit 1; }
  tkb=0
  [ -d "$t/target" ] && tkb="$(du -sk "$t/target" | awk '{print $1}')"
  commit="$(git -C "$t" rev-parse --short HEAD)"
  rows="$rows$label,$scenario,$condition,$run,$(basename "$t"),$wall,$rss,$tkb,$before,$after,${CARGO_BUILD_JOBS:-},${RUSTC_WRAPPER:-},$commit,$now,${CARGO_INCREMENTAL:-}
"
done
printf '%s' "$rows" >> "$out"
printf '%s' "$rows"
