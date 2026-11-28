"""Checks every rust, toml and text code block under website/content against
the first files its Source line names: the block's non-empty lines, stripped,
must appear as one contiguous run in one named file or file folder.
Input: repo root as argv[1]. Output: one line per block, exit 1 on a miss."""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
fence = re.compile(r"^```(\w*)\n(.*?)^```", re.S | re.M)
bad = 0
for md in sorted((root / "website/content").rglob("*.md")):
    text = md.read_text()
    for m in fence.finditer(text):
        lang, body = m.group(1), m.group(2)
        if lang not in ("rust", "toml", "text"):
            continue
        after = text[m.end():]
        src = re.search(r"^Source: (.*)$", after, re.M)
        nxt = fence.search(after)
        if src and src.group(1).startswith("composed"):
            continue
        if not src or (nxt and nxt.start() < src.start()):
            print(f"NOSOURCE {md.relative_to(root)}:{text[:m.start()].count(chr(10))+1}")
            bad += 1
            continue
        paths = re.findall(r"`([^`]+)`", src.group(1))
        files = []
        for p in paths:
            p = p.split(" ")[0]
            f = root / p
            if f.is_dir():
                files += [x for x in f.rglob("*") if x.is_file()]
            elif f.is_file():
                files.append(f)
            elif "*" in p:
                files += list(root.glob(p))
        want = [l.strip() for l in body.splitlines() if l.strip()]
        ok = False
        for f in files:
            raw = f.read_text(errors="ignore")
            have = [l.strip() for l in raw.splitlines() if l.strip()]
            n = len(want)
            if any(have[i:i + n] == want for i in range(len(have) - n + 1)):
                ok = True
                break
            esc = [l.strip().replace('\\"', '"') for l in raw.replace("\\n", "\n").splitlines() if l.strip()]
            if lang == "text" and all(any(w in h for h in esc) for w in want):
                ok = True
                break
        line = text[:m.start()].count("\n") + 1
        if not ok:
            bad += 1
            print(f"MISS {md.relative_to(root)}:{line} files={[str(x.relative_to(root)) for x in files][:3]}")
print("bad", bad)
sys.exit(1 if bad else 0)
