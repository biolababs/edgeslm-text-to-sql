"""Checks that every reference in paper/references.bib is real and that the paper
cites only references that exist. Uses free public APIs (Crossref, arXiv), no keys.
  python tools/verify_citations.py"""
import re
import sys
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper" / "references.bib"
PAPER = ROOT / "paper" / "paper.md"
HEADERS = {"User-Agent": "slm-paper-citation-check/1.0"}
TITLE_MATCH = 0.85


def get(url, **kw):
    """The arXiv and Crossref endpoints rate-limit and time out intermittently.
    A transient failure here is indistinguishable in its output from a genuinely
    missing reference, so retry before letting it surface as one."""
    kw.setdefault("headers", HEADERS)
    kw.setdefault("timeout", 30)
    last = None
    for attempt in range(4):
        try:
            r = get_session().get(url, **kw)
            if r.status_code == 200:
                return r
            last = requests.RequestException(f"HTTP {r.status_code}")
        except requests.RequestException as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


_SESSION = None


def get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", m.group(3), re.S | re.M):
            fields[f.group(1).lower()] = re.sub(r"[{}]", "", " ".join(f.group(2).split()))
        entries[m.group(2)] = fields
    return entries


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def similarity(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def lookup_doi(doi):
    r = get(f"https://api.crossref.org/works/{doi}")
    if r.status_code != 200:
        return None
    msg = r.json()["message"]
    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": (msg.get("title") or [""])[0], "year": year, "source": f"doi:{doi}"}


def lookup_arxiv(arxiv_id):
    r = get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id})
    if r.status_code != 200:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(r.text).find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    published = entry.findtext("a:published", default="", namespaces=ns)
    return {"title": entry.findtext("a:title", namespaces=ns), "year": int(published[:4]) if published else None,
            "source": f"arXiv:{arxiv_id}"}


def search_title(title):
    r = get("https://api.crossref.org/works", params={"query.bibliographic": title, "rows": 5},
                     headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    best = None
    for item in r.json()["message"]["items"]:
        t = (item.get("title") or [""])[0]
        if best is None or similarity(t, title) > similarity(best["title"], title):
            year = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
            best = {"title": t, "year": year, "source": f"doi:{item.get('DOI')} (title search)"}
    return best


def arxiv_id(fields):
    if fields.get("eprint"):
        return fields["eprint"]
    m = re.search(r"arxiv\.org/abs/([\w.\-/]+?)(v\d+)?$", fields.get("url", ""))
    return m.group(1) if m else None


def check(key, fields):
    title = fields.get("title", "")
    if not title:
        return False, "no title in bib entry"
    found = None
    try:
        if fields.get("doi"):
            found = lookup_doi(fields["doi"])
        aid = arxiv_id(fields)
        if found is None and aid:
            found = lookup_arxiv(aid)
            if found is None:
                # Do not fall through to a title search for an entry that names
                # an arXiv id. Preprints frequently have no DOI, so the search
                # matches the nearest indexed paper instead and reports a
                # confident mismatch against a different work - which invites
                # deleting a citation that is in fact correct.
                return False, (f"arXiv:{aid} returned no record; inconclusive, "
                               f"re-run when online rather than treating as wrong")
        if found is None:
            found = search_title(title)
    except requests.RequestException as e:
        return False, f"lookup failed ({e.__class__.__name__}); re-run when online"
    if found is None:
        return False, "not found in Crossref or arXiv"
    score = similarity(found["title"], title)
    if score < TITLE_MATCH:
        return False, f"title mismatch ({score:.2f}) vs {found['source']}: \"{found['title']}\""
    year = fields.get("year")
    if year and found["year"] and abs(int(year) - int(found["year"])) > 1:
        return False, f"year {year} but {found['source']} says {found['year']}"
    return True, f"verified via {found['source']}"


def main():
    entries = parse_bib(BIB.read_text(encoding="utf-8")) if BIB.exists() else {}
    cited = set(re.findall(r"@([\w:.\-]+[\w])", PAPER.read_text(encoding="utf-8"))) if PAPER.exists() else set()
    cited = {c for c in cited if not re.fullmatch(r"(fig|tbl|sec|eq):.*", c)}
    failures = 0
    for key in sorted(cited - entries.keys()):
        print(f"FAIL  {key}: cited in paper.md but missing from references.bib")
        failures += 1
    for key in sorted(entries.keys() - cited):
        print(f"WARN  {key}: in references.bib but never cited")
    verified = 0
    for key, fields in entries.items():
        ok, detail = check(key, fields)
        print(f"{'OK  ' if ok else 'FAIL'}  {key}: {detail}")
        verified += ok
        failures += not ok
        time.sleep(1)  # stay polite to the free APIs
    print(f"\n{verified}/{len(entries)} references verified")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
