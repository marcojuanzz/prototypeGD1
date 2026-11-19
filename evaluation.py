"""Effectiveness measures, agreement, and significance tests (Sections 4.5, 4.6)."""
import math
from itertools import combinations

import numpy as np

import config


def qrels_by_query(qrels):
    out = {}
    for (qid, did), s in qrels.items():
        out.setdefault(qid, {})[did] = s
    return out


def ndcg_at_k(ranking, judged, k=config.NDCG_K):
    """Gains are the consensus scores 0/1/2, discount log2(rank+1). The ideal ranking is
    formed from all judged documents of the query. Unjudged documents count as 0."""
    dcg = sum(judged.get(d, 0) / math.log2(r + 1)
              for r, (d, _) in enumerate(ranking[:k], start=1))
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else None


def precision_at_k(ranking, judged, k=config.PRECISION_K):
    return sum(1 for d, _ in ranking[:k]
               if judged.get(d, 0) >= config.RELEVANT_THRESHOLD) / k


def pool_recall(ranking, judged, k=config.POOL_DEPTH):
    rel = {d for d, s in judged.items() if s >= config.RELEVANT_THRESHOLD}
    if not rel:
        return None
    return sum(1 for d, _ in ranking[:k] if d in rel) / len(rel)


def unjudged_in_top(ranking, judged, k=config.NDCG_K):
    return sum(1 for d, _ in ranking[:k] if d not in judged)


# ------------------------------------------------------------------ agreement
def weighted_kappa(a, b, categories=(0, 1, 2)):
    """Linearly weighted Cohen's kappa (disagreement 0-2 counts twice 0-1)."""
    n = len(a)
    if n == 0:
        return None
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= n
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    denom = (w * exp).sum()
    if denom == 0:
        return 1.0
    return 1 - (w * obs).sum() / denom


def agreement_report(ja, jb):
    common = sorted(set(ja) & set(jb))
    a = [ja[p] for p in common]
    b = [jb[p] for p in common]
    if not common:
        return {"pairs": 0, "kappa": None, "percent_agreement": None}
    return {"pairs": len(common),
            "kappa": weighted_kappa(a, b),
            "percent_agreement": sum(x == y for x, y in zip(a, b)) / len(common)}


# ------------------------------------------------------------------ significance
def randomization_test(x, y, permutations=config.PERMUTATIONS, seed=config.RANDOM_SEED):
    """Two-sided paired randomization test on per-query scores. Each permutation swaps
    the two configurations' scores within a query with probability 0.5.
    p = (count(|perm mean| >= |observed mean|) + 1) / (permutations + 1)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    count, done, batch = 0, 0, 10_000
    while done < permutations:
        m = min(batch, permutations - done)
        signs = rng.choice([-1.0, 1.0], size=(m, len(d)))
        perm = (signs * d).mean(axis=1)
        count += int(np.sum(np.abs(perm) >= abs(obs) - 1e-12))
        done += m
    return obs, (count + 1) / (permutations + 1)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def pairwise_tests(per_query, measure):
    """per_query: {config: {qid: {measure: value}}}. Six pairwise tests, Holm within the
    measure. Queries where the measure is undefined are left out of every comparison."""
    qids = sorted(set.intersection(*[
        {q for q, v in per_query[c].items() if v[measure] is not None} for c in config.CONFIGS]))
    rows = []
    for c1, c2 in combinations(config.CONFIGS, 2):
        x = [per_query[c1][q][measure] for q in qids]
        y = [per_query[c2][q][measure] for q in qids]
        diff, p = randomization_test(x, y) if qids else (float("nan"), float("nan"))
        rows.append({"measure": measure, "config_1": c1, "config_2": c2,
                     "mean_difference": diff, "p_value": p, "n_queries": len(qids)})
    adj = holm([r["p_value"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["significant"] = a < config.SIGNIFICANCE_LEVEL
    return rows
