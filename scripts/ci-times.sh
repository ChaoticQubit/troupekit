#!/bin/sh
set -eu
[ "$#" -eq 1 ] || { echo "usage: sh scripts/ci-times.sh <run-id>" >&2; exit 2; }
run_id="$1"
case "$run_id" in
  *[!0-9]*|'') echo "run id must be numeric: $run_id" >&2; exit 2 ;;
esac
repo="repos/ChaoticQubit/troupekit/actions/runs/$run_id"
out="${CI_TIMES_OUT:-docs/tasks/build-speed/raw}/ci-$run_id.csv"
tmp="$out.tmp"
echo "run_id,job,created_at,started_at,completed_at,queue_s,compute_s,conclusion" > "$tmp"
gh api "$repo/jobs?per_page=100" --paginate --jq '
  .jobs[] | [
    '"$run_id"', .name, .created_at, .started_at, .completed_at,
    ((.started_at | fromdateiso8601) - (.created_at | fromdateiso8601)),
    (if .completed_at and .conclusion != "skipped" then (.completed_at | fromdateiso8601) - (.started_at | fromdateiso8601) else "" end),
    (.conclusion // "")
  ] | @csv' >> "$tmp"
gh api "$repo" --jq '
  [ .id, "run", .created_at, .run_started_at, .updated_at, "",
    ((.updated_at | fromdateiso8601) - (.run_started_at | fromdateiso8601)),
    (.conclusion // "") ] | @csv' >> "$tmp"
mv "$tmp" "$out"
echo "$out"
