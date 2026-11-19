"""Pools, independent judgments, and consensus (Sections 4.2.3, 4.5, 4.5.1)."""
import csv
import random
from datetime import datetime

import config

JDIR = config.OUT_DIR / "judgments"
PDIR = config.OUT_DIR / "pools"


def pool_path(phase):
    return PDIR / f"{phase}_pool.csv"


def write_pool(phase, rankings_by_query):
    """rankings_by_query: {qid: [ranking, ranking, ...]}. Pools the top POOL_DEPTH of every
    ranking and removes duplicates, so each query-document pair is judged once."""
    PDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, rankings in rankings_by_query.items():
        seen = set()
        for ranking in rankings:
            for doc_id, _ in ranking[: config.POOL_DEPTH]:
                if doc_id not in seen:
                    seen.add(doc_id)
                    rows.append((qid, doc_id))
    with open(pool_path(phase), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id"])
        w.writerows(sorted(rows))
    return rows


def read_pool(phase):
    p = pool_path(phase)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [(r["query_id"], r["doc_id"]) for r in csv.DictReader(f)]


def _jfile(phase, annotator):
    safe = "".join(c if c.isalnum() else "_" for c in annotator)
    return JDIR / f"{phase}_{safe}.csv"


def read_judgments(phase, annotator):
    p = _jfile(phase, annotator)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        out = {}
        for r in csv.DictReader(f):
            out[(r["query_id"], r["doc_id"])] = int(r["score"])   # last entry wins
        return out


def save_judgment(phase, annotator, qid, did, score):
    JDIR.mkdir(parents=True, exist_ok=True)
    p = _jfile(phase, annotator)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["query_id", "doc_id", "score", "timestamp"])
        w.writerow([qid, did, int(score), datetime.now().isoformat(timespec="seconds")])


def annotation_order(phase, annotator):
    """Random order of pooled pairs, seeded per annotator so it is reproducible."""
    pairs = read_pool(phase)
    rnd = random.Random(f"{config.RANDOM_SEED}-{phase}-{annotator}")
    rnd.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------- consensus
def _res_file(phase):
    return JDIR / f"{phase}_resolutions.csv"


def read_resolutions(phase):
    p = _res_file(phase)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {(r["query_id"], r["doc_id"]): int(r["score"]) for r in csv.DictReader(f)}


def save_resolution(phase, qid, did, score):
    JDIR.mkdir(parents=True, exist_ok=True)
    p = _res_file(phase)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["query_id", "doc_id", "score", "timestamp"])
        w.writerow([qid, did, int(score), datetime.now().isoformat(timespec="seconds")])


def consensus(phase):
    """Final score per pooled pair: the agreed score, or the score recorded after discussing
    a disagreement. Returns (qrels, status) where status counts what is still missing."""
    a, b = (read_judgments(phase, x) for x in config.ANNOTATORS)
    res = read_resolutions(phase)
    qrels, status = {}, {"pairs": 0, "agreed": 0, "resolved": 0,
                         "unresolved": 0, "incomplete": 0}
    for pair in read_pool(phase):
        status["pairs"] += 1
        if pair in a and pair in b:
            if a[pair] == b[pair]:
                qrels[pair] = a[pair]
                status["agreed"] += 1
            elif pair in res:
                qrels[pair] = res[pair]
                status["resolved"] += 1
            else:
                status["unresolved"] += 1
        else:
            status["incomplete"] += 1
    return qrels, status


def disagreements(phase):
    a, b = (read_judgments(phase, x) for x in config.ANNOTATORS)
    res = read_resolutions(phase)
    return [(p, a[p], b[p], res.get(p)) for p in read_pool(phase)
            if p in a and p in b and a[p] != b[p]]
