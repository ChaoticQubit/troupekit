#!/bin/sh
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
fetch="$here/fetch-test-model.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

want=$(sed -n 's/^SHA256=//p' "$fetch")
id=$(sed -n 's/^ID=//p' "$fetch")
[ -n "$want" ] && [ -n "$id" ] || fail "cannot read ID and SHA256 from $fetch"

mkdir -p "$work/bin"
cat > "$work/bin/curl" <<'STUB'
#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
[ -n "$out" ] || exit 2
printf '%s' "${STUB_BODY:-junk}" > "$out"
exit 0
STUB
cat > "$work/bin/sha256sum" <<STUB
#!/bin/sh
if [ "\$(cat "\$1")" = good ]; then echo "$want  \$1"; else echo "0000  \$1"; fi
STUB
chmod +x "$work/bin/curl" "$work/bin/sha256sum"
PATH="$work/bin:$PATH"
export PATH

models="$work/models"
mkdir -p "$models"
final="$models/$id-$want.gguf"
foreign="$final.part"
printf 'another run in progress' > "$foreign"

if TROUPEKIT_MODEL_DIR="$models" STUB_BODY=junk sh "$fetch" >/dev/null 2>&1; then
  fail "a digest mismatch must fail"
fi
[ "$(cat "$foreign")" = "another run in progress" ] || fail "a failed run touched another run's staging file"
[ ! -e "$final" ] || fail "a failed run published a file"
left=$(ls "$models" | grep -v "^$id-$want.gguf.part\$" || true)
[ -z "$left" ] || fail "a failed run left files: $left"

got=$(TROUPEKIT_MODEL_DIR="$models" STUB_BODY=good sh "$fetch") || fail "a good download must pass"
[ "$got" = "$final" ] || fail "printed $got, want $final"
[ "$(cat "$final")" = good ] || fail "published the wrong file"
[ "$(cat "$foreign")" = "another run in progress" ] || fail "a good run moved another run's staging file"
left=$(ls "$models" | grep -v "^$id-$want.gguf\$" | grep -v "^$id-$want.gguf.part\$" || true)
[ -z "$left" ] || fail "a good run left files: $left"

echo "fetch-test-model-test: PASS"
