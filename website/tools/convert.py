"""Turns the tool-neutral pages in website/content into an mdBook source folder.

Input: argv[1] the content folder, argv[2] the output folder (wiped first).
Output: every page listed in _outline.md, front matter stripped, plus a
SUMMARY.md in outline order. Exits non-zero when a page and the outline
disagree, or a page has front matter other than title and description.
"""
import pathlib
import re
import shutil
import sys

ROW = re.compile(r"^\| (?P<title>[^|]+?) \| `(?P<file>[^`]+\.md)` \|")
FRONT = re.compile(r"\A---\n(?P<body>.*?)\n---\n\n?", re.S)
SECTIONS = ("Start here", "Concepts", "Guides", "Reference", "Project")


def outline(path):
    """Reads _outline.md. Input: its path. Output: a list of (section, file)
    pairs in page order, for the sections named in SECTIONS."""
    rows = []
    section = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = ROW.match(line)
        if m and section in SECTIONS:
            rows.append((section, m.group("file")))
    assert rows, "no page rows in the outline"
    files = [f for _, f in rows]
    assert len(files) == len(set(files)), "a page is listed twice in the outline"
    return rows


def split_page(path):
    """Splits one page. Input: its path. Output: (title, body) where body is
    the Markdown with front matter removed and title comes from front matter."""
    text = path.read_text()
    m = FRONT.match(text)
    assert m, f"{path}: no front matter"
    meta = {}
    for line in m.group("body").splitlines():
        key, sep, value = line.partition(": ")
        assert sep, f"{path}: front matter line {line!r}"
        meta[key] = value.strip()
    assert set(meta) == {"title", "description"}, f"{path}: front matter keys {sorted(meta)}"
    body = text[m.end():]
    assert body.startswith("# "), f"{path}: page does not open with its H1"
    return meta["title"], body


def summary(rows, titles):
    """Builds SUMMARY.md. Input: outline rows and a file-to-title map.
    Output: the SUMMARY.md text: the first page as a prefix chapter, the rest
    of Start here numbered, Concepts nested under a draft chapter, Guides and
    Reference as parts, and Project as suffix chapters after a separator."""
    by = {s: [f for sec, f in rows if sec == s] for s in SECTIONS}
    assert all(by[s] for s in SECTIONS), "an outline section is empty"
    first, *start = by["Start here"]
    out = ["# Summary", "", f"[{titles[first]}]({first})", ""]
    out += [f"- [{titles[f]}]({f})" for f in start]
    out.append("- [Concepts]()")
    out += [f"    - [{titles[f]}]({f})" for f in by["Concepts"]]
    for part in ("Guides", "Reference"):
        out += ["", f"# {part}", ""]
        out += [f"- [{titles[f]}]({f})" for f in by[part]]
    out += ["", "---", ""]
    for f in by["Project"]:
        out += [f"[{titles[f]}]({f})", ""]
    return "\n".join(out)


def main():
    """Runs the conversion. Input: argv as in the module docstring.
    Output: the mdBook source folder on disk, and a count on stdout."""
    assert len(sys.argv) == 3, "usage: convert.py <content dir> <out dir>"
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    rows = outline(src / "_outline.md")
    listed = {f for _, f in rows}
    present = {p.relative_to(src).as_posix() for p in src.rglob("*.md")} - {"_outline.md"}
    assert listed == present, f"outline and pages differ: {sorted(listed ^ present)}"
    if dst.exists():
        shutil.rmtree(dst)
    titles = {}
    for _, f in rows:
        title, body = split_page(src / f)
        titles[f] = title
        out = dst / f
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
    (dst / "SUMMARY.md").write_text(summary(rows, titles))
    written = len(list(dst.rglob("*.md")))
    assert written == len(rows) + 1, f"wrote {written} files for {len(rows)} pages"
    print(f"converted {len(rows)} pages into {dst}")


if __name__ == "__main__":
    main()
