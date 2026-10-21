"""Builds the paper from paper/paper.md. Every {{key}} is replaced with the value in
build/numbers.json and every {{table:name}} with build/tables/name.md, so the text
can never disagree with the analysis. Outputs build/paper.docx and build/paper.pdf.
  python tools/render.py"""
import json
import re
import sys
from pathlib import Path

import pypandoc
import typst

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
PAPER = ROOT / "paper" / "paper.md"
BIB = ROOT / "paper" / "references.bib"


def fill(text, numbers):
    missing = []

    def table(m):
        path = BUILD / "tables" / f"{m.group(1)}.md"
        if not path.exists():
            missing.append(f"table:{m.group(1)}")
            return m.group(0)
        return path.read_text(encoding="utf-8").strip()

    def value(m):
        key = m.group(1)
        if key not in numbers:
            missing.append(key)
            return m.group(0)
        v = numbers[key]
        return f"{v:,}" if isinstance(v, int) else str(v)

    text = re.sub(r"\{\{table:([\w\-]+)\}\}", table, text)
    text = re.sub(r"\{\{([\w.\-]+)\}\}", value, text)
    return text, missing


def main():
    numbers = json.loads((BUILD / "numbers.json").read_text(encoding="utf-8"))
    text, missing = fill(PAPER.read_text(encoding="utf-8"), numbers)
    if missing:
        sys.exit("paper.md uses values the analysis does not produce:\n  " + "\n  ".join(sorted(set(missing))))
    filled = BUILD / "paper.md"
    filled.write_text(text, encoding="utf-8")

    args = ["--citeproc", f"--bibliography={BIB}", f"--resource-path={BUILD}", "--standalone"]
    pypandoc.convert_file(str(filled), "docx", outputfile=str(BUILD / "paper.docx"), extra_args=args)
    typ = BUILD / "paper.typ"
    pypandoc.convert_file(str(filled), "typst", outputfile=str(typ), extra_args=args)
    typst.compile(str(typ), output=str(BUILD / "paper.pdf"), root=str(BUILD))
    print("Built build/paper.docx and build/paper.pdf")


if __name__ == "__main__":
    main()
