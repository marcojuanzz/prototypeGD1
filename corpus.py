"""Loading and cleaning the abstract collection (Section 4.4)."""
import csv
import re
from dataclasses import dataclass

import config


@dataclass
class Doc:
    doc_id: str
    title: str
    abstract: str
    query_role: str  # "dev", "final" or ""


def clean_abstract(text: str) -> str:
    """Mechanical cleaning only: merge line breaks into continuous text and collapse
    repeated whitespace. Wording is left unchanged (Section 4.4)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_corpus(path=None):
    """Returns (docs, excluded). Rows whose status is 'Unusable' or whose abstract is
    empty are excluded and reported, as Section 4.4 requires."""
    path = path or config.CORPUS_CSV
    if not path.exists():
        raise SystemExit(f"Corpus file not found: {path}\n"
                         "Export the Corpus tab of the Google Sheet as CSV and save it there.")
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    # Some exports start with a '## Sheet name:' line; skip anything before the header.
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("doc_id"))
    reader = csv.DictReader(lines[start:])
    docs, excluded, seen = [], [], set()
    for row in reader:
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id or not doc_id[0].isalnum():
            if doc_id.startswith("##"):
                break  # the next sheet of a multi-sheet export
            continue
        abstract = clean_abstract(row.get("abstract") or "")
        status = (row.get("status") or "").strip().lower()
        if doc_id in seen:
            excluded.append((doc_id, "duplicate doc_id"))
            continue
        seen.add(doc_id)
        if status == "unusable":
            excluded.append((doc_id, "marked Unusable"))
            continue
        if not abstract:
            excluded.append((doc_id, "no abstract"))
            continue
        role = (row.get("query_role") or "").strip().lower()
        if role not in ("dev", "final"):
            role = ""
        docs.append(Doc(doc_id, (row.get("title") or "").strip(), abstract, role))
    docs.sort(key=lambda d: d.doc_id)
    return docs, excluded
