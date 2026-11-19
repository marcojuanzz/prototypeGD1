"""Experimental platform (Section 4.3): retrieval demo, blind annotation, disagreement
resolution, and results. Run:  python app.py   then open http://127.0.0.1:5000"""
import json
from functools import lru_cache

from flask import Flask, abort, redirect, render_template, request, url_for

import config
import judgments as jd
from cli import RESULTS, get_engine, load_state

app = Flask(__name__)

LABELS = {2: "Highly Relevant", 1: "Partially Relevant", 0: "Not Relevant"}


@lru_cache(maxsize=1)
def engine():
    eng, docs, meta = get_engine()
    return eng, docs, meta


def phase_progress(phase):
    pool = jd.read_pool(phase)
    out = {"pairs": len(pool), "annotators": {}}
    for a in config.ANNOTATORS:
        done = jd.read_judgments(phase, a)
        out["annotators"][a] = sum(1 for p in pool if p in done)
    if pool:
        out["consensus"] = jd.consensus(phase)[1]
    return out


@app.route("/")
def home():
    s = load_state()
    eng, docs, meta = engine()
    return render_template("home.html", s=s, n_docs=len(docs), meta=meta,
                           progress={p: phase_progress(p) for p in ("dev", "final")},
                           annotators=config.ANNOTATORS)


@app.route("/search")
def search():
    eng, docs, _ = engine()
    s = load_state()
    qid = request.args.get("q") or next((d.doc_id for d in docs if d.query_role == "final"),
                                        docs[0].doc_id)
    if qid not in eng.by_id:
        abort(404)
    k = int(request.args.get("k", config.POOL_DEPTH))
    alpha_star = s.get("alpha_star")
    ref = s.get("dev", {}).get("percentile_reference")
    columns, info = [], None
    if alpha_star is not None and ref:
        runs, info = eng.run_all(qid, alpha_star, ref)
        labels = {"Fixed-Weight Hybrid": f"Fixed-Weight Hybrid (a* = {alpha_star})",
                  "Query-Adaptive Hybrid": f"Query-Adaptive Hybrid (a_q = {info['alpha_q']:.2f})"}
        for c in config.CONFIGS:
            columns.append((labels.get(c, c), runs[c][:k]))
    else:   # before the development phase only the two baselines are available
        columns = [("BM25", eng.rank_bm25(qid)[:k]), ("SBERT", eng.rank_sbert(qid)[:k])]
    return render_template("search.html", docs=docs, q=eng.by_id[qid], columns=columns,
                           info=info, by_id=eng.by_id, alpha_star=alpha_star,
                           alpha_source=s.get("alpha_source"), k=k)


@app.route("/annotate/<phase>")
def annotate(phase):
    if phase not in ("dev", "final"):
        abort(404)
    who = request.args.get("who")
    if who not in config.ANNOTATORS:
        return render_template("pick_annotator.html", phase=phase, annotators=config.ANNOTATORS)
    eng, _, _ = engine()
    order = jd.annotation_order(phase, who)
    done = jd.read_judgments(phase, who)
    redo = request.args.get("redo")
    if redo:
        qid, did = redo.split("|")
        pair = (qid, did)
    else:
        pair = next((p for p in order if p not in done), None)
    if pair is None:
        return render_template("done.html", phase=phase, who=who, n=len(order))
    idx = order.index(pair)
    prev = order[idx - 1] if idx > 0 else None   # lets the annotator correct a mistake
    return render_template("annotate.html", phase=phase, who=who, pair=pair,
                           q=eng.by_id[pair[0]], d=eng.by_id[pair[1]], labels=LABELS,
                           position=idx + 1, total=len(order), n_done=len(
                               [p for p in order if p in done]),
                           current=done.get(pair), prev=prev)


@app.post("/annotate/<phase>")
def save(phase):
    who = request.form["who"]
    if who not in config.ANNOTATORS or phase not in ("dev", "final"):
        abort(400)
    score = int(request.form["score"])
    if score not in LABELS:
        abort(400)
    pair = (request.form["qid"], request.form["did"])
    if pair not in set(jd.read_pool(phase)):
        abort(400)
    jd.save_judgment(phase, who, pair[0], pair[1], score)
    return redirect(url_for("annotate", phase=phase, who=who))


@app.route("/resolve/<phase>", methods=["GET", "POST"])
def resolve(phase):
    if phase not in ("dev", "final"):
        abort(404)
    if request.method == "POST":
        score = int(request.form["score"])
        if score in LABELS:
            jd.save_resolution(phase, request.form["qid"], request.form["did"], score)
        return redirect(url_for("resolve", phase=phase))
    eng, _, _ = engine()
    items = jd.disagreements(phase)
    open_items = [i for i in items if i[3] is None]
    return render_template("resolve.html", phase=phase, items=items, open_items=open_items,
                           by_id=eng.by_id, labels=LABELS, annotators=config.ANNOTATORS)


@app.route("/results")
def results():
    p = RESULTS / "report.json"
    report = json.loads(p.read_text()) if p.exists() else None
    has_plot = (RESULTS / "query_level_scatter.png").exists()
    return render_template("results.html", r=report, has_plot=has_plot,
                           configs=config.CONFIGS)


@app.route("/results/plot.png")
def plot():
    from flask import send_file
    return send_file(RESULTS / "query_level_scatter.png", mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
