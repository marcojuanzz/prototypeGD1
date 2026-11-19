"""Command-line pipeline, in the fixed order of Section 4.2.

  python cli.py prepare        load + clean corpus, count word pieces, encode abstracts
  python cli.py dev            development runs, percentile references, development pool
  (judge the development pool in the web app)
  python cli.py select-alpha   choose a* from development judgments   [--provisional 0.5]
  python cli.py final          freeze settings, run the 4 configurations, final pool
  (judge the final pool in the web app, resolve disagreements)
  python cli.py evaluate       metrics, agreement, significance, query-level analysis
  python cli.py status         where the pipeline is
"""
import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from importlib import metadata

import config
import evaluation as ev
import judgments as jd
from corpus import load_corpus
from retrieval import Engine, build_embeddings, load_embeddings

STATE = config.OUT_DIR / "state.json"
RUNS = config.OUT_DIR / "runs"
RESULTS = config.OUT_DIR / "results"


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def slug(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def write_run(phase, name, rankings):
    """Run file: query_id, doc_id, rank, score (Section 4.2.3)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / f"{phase}_{slug(name)}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "doc_id", "rank", "score"])
        for qid, ranking in rankings.items():
            for r, (did, s) in enumerate(ranking, start=1):
                w.writerow([qid, did, r, f"{s:.6f}"])


def get_engine():
    docs, _ = load_corpus()
    emb, meta = load_embeddings(docs)
    return Engine(docs, emb), docs, meta


def single_source(phase, name):
    """Demo/pilot only: one judgment source (an annotator, or the 'AI pilot' file)
    instead of the two-annotator consensus."""
    pool = set(jd.read_pool(phase))
    j = {p: v for p, v in jd.read_judgments(phase, name).items() if p in pool}
    if not j:
        sys.exit(f"No {phase} judgments found for '{name}' "
                 f"(expected outputs/judgments/{phase}_{name.replace(' ', '_')}.csv).")
    return j


def queries(docs, role):
    return [d.doc_id for d in docs if d.query_role == role]


# ------------------------------------------------------------------ prepare
def cmd_prepare(args, embedder=None):
    docs, excluded = load_corpus()
    dev, final = queries(docs, "dev"), queries(docs, "final")
    print(f"Abstracts included: {len(docs)}   excluded: {len(excluded)}")
    for did, why in excluded:
        print(f"  excluded {did}: {why}")
    print(f"Queries: {len(dev)} development, {len(final)} final")
    if not dev or not final:
        sys.exit("Mark the query abstracts as 'dev' or 'final' in the query_role column.")
    words = {d.doc_id: len(d.abstract.split()) for d in docs}
    print(f"Abstract length: {min(words.values())}-{max(words.values())} words "
          f"(mean {sum(words.values()) / len(words):.0f})")
    _, counts = build_embeddings(docs, embedder)
    over = [(d.doc_id, c) for d, c in zip(docs, counts) if c > config.SBERT_MAX_TOKENS]
    print(f"Word pieces over the {config.SBERT_MAX_TOKENS} limit (truncated): "
          f"{len(over)} of {len(docs)} ({100 * len(over) / len(docs):.0f}%)")
    for did, c in over:
        print(f"  {did}: {c} word pieces")
    s = load_state()
    s["prepared"] = {"at": datetime.now().isoformat(timespec="seconds"),
                     "included": len(docs), "excluded": excluded,
                     "dev_queries": dev, "final_queries": final,
                     "truncated": over}
    save_state(s)
    print("Saved embeddings. Next: python cli.py dev")


# ------------------------------------------------------------------ development
def cmd_dev(args):
    eng, docs, _ = get_engine()
    s = load_state()
    if s.get("frozen"):
        sys.exit("Settings are frozen; the development phase cannot be re-run.")
    dev = queries(docs, "dev")
    ref = {"length": [eng.query_length(q) for q in dev],
           "specificity": [eng.term_specificity(q) for q in dev]}
    rankings = {}
    runs = {"BM25": {}, "SBERT": {}}
    for q in dev:
        runs["BM25"][q] = eng.rank_bm25(q)
        runs["SBERT"][q] = eng.rank_sbert(q)
        rankings[q] = [runs["BM25"][q], runs["SBERT"][q]]
        for a in config.ALPHA_GRID:
            r = eng.rank_hybrid(q, a)
            runs.setdefault(f"Hybrid a={a}", {})[q] = r
            rankings[q].append(r)
    for name, rk in runs.items():
        write_run("dev", name, rk)
    pairs = jd.write_pool("dev", rankings)
    s["dev"] = {"at": datetime.now().isoformat(timespec="seconds"), "queries": dev,
                "percentile_reference": ref, "pool_pairs": len(pairs)}
    save_state(s)
    print("Development query signals (percentile reference):")
    for q, l, sp in zip(dev, ref["length"], ref["specificity"]):
        print(f"  {q}: length={l:4d}  specificity={sp:.3f}")
    print(f"Development pool: {len(pairs)} query-document pairs "
          f"(11 rankings x top {config.POOL_DEPTH}, duplicates removed).")
    print("Next: judge the 'dev' pool in the web app (python app.py), "
          "then python cli.py select-alpha")


# ------------------------------------------------------------------ alpha selection
def cmd_select_alpha(args):
    s = load_state()
    if "dev" not in s:
        sys.exit("Run `python cli.py dev` first.")
    if s.get("frozen"):
        sys.exit("Settings are frozen; a* cannot be changed.")
    if args.provisional is not None:
        if args.provisional not in config.ALPHA_GRID:
            sys.exit(f"Choose a value from {config.ALPHA_GRID}.")
        s["alpha_star"] = args.provisional
        s["alpha_source"] = "provisional (development judgments not finished)"
        save_state(s)
        print(f"a* = {args.provisional} set PROVISIONALLY. Report it as such.")
        return
    eng, docs, _ = get_engine()
    qrels, status = jd.consensus("dev")
    if args.annotator:
        qrels = single_source("dev", args.annotator)
        status = {"unresolved": 0, "incomplete": 0}
        print(f"Using judgments from '{args.annotator}' only (pilot).")
    if status["unresolved"] or status["incomplete"]:
        print(f"Warning: {status['incomplete']} pairs not judged by both annotators and "
              f"{status['unresolved']} disagreements unresolved; they count as not relevant.")
    by_q = ev.qrels_by_query(qrels)
    table = []
    for a in config.ALPHA_GRID:
        vals = []
        for q in s["dev"]["queries"]:
            v = ev.ndcg_at_k(eng.rank_hybrid(q, a), by_q.get(q, {}))
            if v is not None:
                vals.append(v)
        table.append((a, sum(vals) / len(vals) if vals else float("nan"), len(vals)))
    valid = [t for t in table if t[2] > 0]
    if not valid:
        sys.exit("No development query has a relevant document yet. Judge the dev pool first.")
    best = max(t[1] for t in valid)
    tied = [t[0] for t in valid if abs(t[1] - best) < 1e-12]
    alpha = min(tied, key=lambda a: (abs(a - 0.5), a))   # closest to 0.5, then lower
    print(" alpha   mean nDCG@10   queries")
    for a, m, n in table:
        print(f"  {a:.1f}     {m:.4f}        {n}{'   <- a*' if a == alpha else ''}")
    s["alpha_star"] = alpha
    s["alpha_source"] = ("development judgments" if not args.annotator
                         else f"development judgments from '{args.annotator}' (pilot)")
    s["alpha_table"] = table
    save_state(s)
    print(f"Selected a* = {alpha}. Next: python cli.py final")


# ------------------------------------------------------------------ final runs
def _versions():
    out = {"python": platform.python_version()}
    for pkg in ("bm25s", "sentence-transformers", "torch", "transformers", "numpy", "scipy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def cmd_final(args):
    s = load_state()
    if "alpha_star" not in s:
        sys.exit("Select a* first: python cli.py select-alpha  (or --provisional 0.5)")
    eng, docs, meta = get_engine()
    if not s.get("frozen"):
        s["frozen"] = {"at": datetime.now().isoformat(timespec="seconds"),
                       "alpha_star": s["alpha_star"], "alpha_source": s["alpha_source"],
                       "bm25": {"k1": config.BM25_K1, "b": config.BM25_B, "variant": "lucene"},
                       "model": meta["model"], "max_tokens": config.SBERT_MAX_TOKENS,
                       "pool_depth": config.POOL_DEPTH, "seed": config.RANDOM_SEED,
                       "versions": _versions()}
    ref = s["dev"]["percentile_reference"]
    final = queries(docs, "final")
    runs = {c: {} for c in config.CONFIGS}
    info_rows = []
    for q in final:
        r, info = eng.run_all(q, s["alpha_star"], ref)
        for c in config.CONFIGS:
            runs[c][q] = r[c]
        info_rows.append({"query_id": q, **info})
    for c in config.CONFIGS:
        write_run("final", c, runs[c])
    pairs = jd.write_pool("final", {q: [runs[c][q] for c in config.CONFIGS] for q in final})
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "final_query_signals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(info_rows[0]))
        w.writeheader()
        w.writerows(info_rows)
    s["final"] = {"at": datetime.now().isoformat(timespec="seconds"), "queries": final,
                  "pool_pairs": len(pairs)}
    save_state(s)
    print(f"a* = {s['alpha_star']} ({s['alpha_source']})")
    print(" query   length  specificity  P_len  P_spec  alpha_q")
    for r in info_rows:
        print(f"  {r['query_id']}   {r['length']:5d}   {r['specificity']:.3f}      "
              f"{r['p_length']:.2f}   {r['p_specificity']:.2f}    {r['alpha_q']:.2f}")
    print(f"Final pool: {len(pairs)} pairs. Next: judge the 'final' pool, resolve "
          "disagreements, then python cli.py evaluate")


# ------------------------------------------------------------------ evaluation
def cmd_evaluate(args):
    from scipy.stats import spearmanr
    s = load_state()
    if "final" not in s:
        sys.exit("Run `python cli.py final` first.")
    eng, docs, _ = get_engine()
    RESULTS.mkdir(parents=True, exist_ok=True)
    qrels, status = jd.consensus("final")
    if args.annotator:   # prototype demo only: one judgment source instead of consensus
        qrels = single_source("final", args.annotator)
        status = dict(status, source=f"{args.annotator} only (pilot, not the consensus)")
    by_q = ev.qrels_by_query(qrels)
    ref = s["dev"]["percentile_reference"]
    a_star = s["frozen"]["alpha_star"]
    final = s["final"]["queries"]

    per_query = {c: {} for c in config.CONFIGS}
    signals = {}
    no_relevant = []
    for q in final:
        runs, info = eng.run_all(q, a_star, ref)
        signals[q] = info
        judged = by_q.get(q, {})
        if not any(v >= config.RELEVANT_THRESHOLD for v in judged.values()):
            no_relevant.append(q)
        for c in config.CONFIGS:
            per_query[c][q] = {
                "nDCG@10": ev.ndcg_at_k(runs[c], judged),
                "P@5": ev.precision_at_k(runs[c], judged),
                "Recall (pool)": ev.pool_recall(runs[c], judged),
                "unjudged_top10": ev.unjudged_in_top(runs[c], judged)}
    used = [q for q in final if q not in no_relevant]
    for c in config.CONFIGS:          # queries without a relevant document are excluded
        for q in no_relevant:
            per_query[c][q] = {k: None if k != "unjudged_top10" else v
                               for k, v in per_query[c][q].items()}

    measures = ["nDCG@10", "P@5", "Recall (pool)"]
    with open(RESULTS / "per_query_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "configuration", *measures, "unjudged_in_top10"])
        for q in final:
            for c in config.CONFIGS:
                v = per_query[c][q]
                w.writerow([q, c, *[("" if v[m] is None else f"{v[m]:.4f}") for m in measures],
                            v["unjudged_top10"]])
    summary = {}
    for c in config.CONFIGS:
        summary[c] = {}
        for m in measures:
            vals = [per_query[c][q][m] for q in used if per_query[c][q][m] is not None]
            summary[c][m] = sum(vals) / len(vals) if vals else None
    tests = [row for m in measures for row in ev.pairwise_tests(per_query, m)]
    with open(RESULTS / "significance_tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0]))
        w.writeheader()
        w.writerows(tests)

    agreement = {}
    for phase in ("dev", "final"):
        ja, jb = (jd.read_judgments(phase, a) for a in config.ANNOTATORS)
        agreement[phase] = ev.agreement_report(ja, jb)

    # Query-level analysis (Section 4.6), exploratory
    def nd(c, q):
        return per_query[c][q]["nDCG@10"]
    rows = []
    for q in used:
        rows.append({"query_id": q, **signals[q],
                     "delta_LS": nd("BM25", q) - nd("SBERT", q),
                     "delta_AF": nd("Query-Adaptive Hybrid", q) - nd("Fixed-Weight Hybrid", q),
                     "alpha_distance": abs(signals[q]["alpha_q"] - a_star)})
    corr = {}
    if len(rows) >= 3:
        for x in ("length", "specificity", "alpha_q"):
            rho, p = spearmanr([r[x] for r in rows], [r["delta_LS"] for r in rows])
            corr[f"delta_LS vs {x}"] = (rho, p)
        rho, p = spearmanr([r["alpha_distance"] for r in rows], [r["delta_AF"] for r in rows])
        corr["delta_AF vs |alpha_q - a*|"] = (rho, p)
    wins = sum(r["delta_AF"] > 1e-12 for r in rows)
    losses = sum(r["delta_AF"] < -1e-12 for r in rows)
    ties = len(rows) - wins - losses
    with open(RESULTS / "query_level_analysis.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    _plots(rows)

    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "alpha_star": a_star, "alpha_source": s["frozen"]["alpha_source"],
              "judgment_status": status, "queries_used": used,
              "queries_without_relevant": no_relevant, "summary": summary,
              "tests": tests, "agreement": agreement,
              "correlations": {k: {"rho": v[0], "p": v[1]} for k, v in corr.items()},
              "adaptive_vs_fixed": {"higher": wins, "lower": losses, "equal": ties}}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=float))
    _print_report(report)


def _plots(rows):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [("alpha_q", "delta_LS", "Query-adaptive weight a_q", "nDCG@10 BM25 - SBERT"),
             ("length", "delta_LS", "Query length (tokens)", "nDCG@10 BM25 - SBERT"),
             ("specificity", "delta_LS", "Term specificity (mean IDF)", "nDCG@10 BM25 - SBERT"),
             ("alpha_distance", "delta_AF", "|a_q - a*|", "nDCG@10 adaptive - fixed")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (x, y, xl, yl) in zip(axes.flat, specs):
        ax.scatter([r[x] for r in rows], [r[y] for r in rows])
        for r in rows:
            ax.annotate(r["query_id"], (r[x], r[y]), fontsize=7, alpha=.7)
        ax.axhline(0, lw=.8, color="gray")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.suptitle("Query-level analysis (exploratory)")
    fig.tight_layout()
    fig.savefig(RESULTS / "query_level_scatter.png", dpi=150)


def _fmt(v, d=4):
    return "-" if v is None else f"{v:.{d}f}"


def _print_report(r):
    st = r["judgment_status"]
    print(f"a* = {r['alpha_star']} ({r['alpha_source']})")
    print(f"Final pool: {st['pairs']} pairs | agreed {st['agreed']} | resolved {st['resolved']}"
          f" | unresolved {st['unresolved']} | not judged by both {st['incomplete']}")
    if st.get("source"):
        print(f"  SOURCE: {st['source']}")
    if st["unresolved"] or st["incomplete"] or st.get("source"):
        print("  NOTE: judging is incomplete; treat these numbers as pilot results only.")
    print(f"Queries used: {len(r['queries_used'])}; without any relevant document "
          f"(excluded): {r['queries_without_relevant']}")
    print(f"\n{'Configuration':24s} {'nDCG@10':>8s} {'P@5':>8s} {'Recall':>8s}")
    for c, m in r["summary"].items():
        print(f"{c:24s} {_fmt(m['nDCG@10']):>8s} {_fmt(m['P@5']):>8s} "
              f"{_fmt(m['Recall (pool)']):>8s}")
    print("\nPaired randomization tests (Holm-adjusted within each measure)")
    for t in r["tests"]:
        if t["n_queries"] == 0:
            continue
        print(f"  {t['measure']:13s} {t['config_1']:>22s} vs {t['config_2']:22s} "
              f"diff={t['mean_difference']:+.4f}  p_holm={t['p_holm']:.4f}"
              f"{'  *' if t['significant'] else ''}")
    for ph, a in r["agreement"].items():
        print(f"Agreement ({ph}): {a['pairs']} pairs, weighted kappa={_fmt(a['kappa'], 3)}, "
              f"percent agreement={_fmt(a['percent_agreement'], 3)}")
    for k, v in r["correlations"].items():
        if v["rho"] != v["rho"]:
            print(f"Spearman {k}: undefined (no variation)")
            continue
        print(f"Spearman {k}: rho={v['rho']:+.3f} (p={v['p']:.3f}, exploratory)")
    w = r["adaptive_vs_fixed"]
    print(f"Adaptive vs fixed: higher on {w['higher']}, lower on {w['lower']}, "
          f"equal on {w['equal']} queries")
    print(f"\nFiles: {RESULTS}")


def cmd_status(args):
    s = load_state()
    print(json.dumps({k: v for k, v in s.items() if k in
                      ("prepared", "dev", "alpha_star", "alpha_source", "final")},
                     indent=2)[:3000])
    for phase in ("dev", "final"):
        if jd.read_pool(phase):
            _, st = jd.consensus(phase)
            print(phase, st)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("dev")
    sa = sub.add_parser("select-alpha")
    sa.add_argument("--provisional", type=float, default=None,
                    help="set a* without development judgments (prototype demo only)")
    sa.add_argument("--annotator", default=None,
                    help="pilot only: use one judgment source instead of the consensus")
    sub.add_parser("final")
    e = sub.add_parser("evaluate")
    e.add_argument("--annotator", default=None,
                   help="demo only: use one annotator's scores instead of the consensus")
    sub.add_parser("status")
    args = p.parse_args(argv)
    {"prepare": cmd_prepare, "dev": cmd_dev, "select-alpha": cmd_select_alpha,
     "final": cmd_final, "evaluate": cmd_evaluate, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
