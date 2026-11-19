"""The four retrieval configurations (Chapter 3, Section 4.2.2).

BM25            L(d,q)  - BM25S, Lucene variant, k1=1.2, b=0.75           (Eq. 3.1)
Sentence-BERT   S(d,q)  - cosine similarity of all-mpnet-base-v2 vectors
Fixed hybrid    H(d,q)  = a* L'(d,q) + (1-a*) S'(d,q)                     (Eq. 3.3)
Adaptive hybrid HA(d,q) = a_q L'(d,q) + (1-a_q) S'(d,q)                   (Eq. 3.9)
L', S' are min-max normalized per query over the candidate set D_q        (Eq. 3.2)
"""
import json
import math
import re
from collections import Counter

import numpy as np

import config

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stopwords():
    from bm25s.stopwords import STOPWORDS_EN
    return set(STOPWORDS_EN)


STOPWORDS = _stopwords()


def tokenize(text: str):
    """Section 4.2.2.1: lowercase, split at spaces and punctuation, remove the BM25
    library's English stopwords, no stemming. Repeated tokens are kept."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# ----------------------------------------------------------------------------- embeddings
EMB_FILE = config.OUT_DIR / "embeddings.npy"
EMB_META = config.OUT_DIR / "embeddings_meta.json"


def build_embeddings(docs, embedder=None):
    """Encodes every cleaned abstract once (Section 4.2.2.3) and counts word pieces with
    the model's own tokenizer. Returns (matrix, token_counts)."""
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.SBERT_MODEL)
        model.max_seq_length = config.SBERT_MAX_TOKENS
        texts = [d.abstract for d in docs]
        counts = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
        emb = model.encode(texts, batch_size=16, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        name = config.SBERT_MODEL
    else:  # test double used only for automated checks
        emb, counts, name = embedder(docs)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, emb.astype(np.float32))
    EMB_META.write_text(json.dumps({
        "model": name, "doc_ids": [d.doc_id for d in docs],
        "token_counts": dict(zip([d.doc_id for d in docs], counts)),
        "max_tokens": config.SBERT_MAX_TOKENS}, indent=2))
    return emb, counts


def load_embeddings(docs):
    if not EMB_FILE.exists():
        raise SystemExit("No stored embeddings. Run `python cli.py prepare` first.")
    meta = json.loads(EMB_META.read_text())
    if meta["doc_ids"] != [d.doc_id for d in docs]:
        raise SystemExit("The corpus changed since embeddings were built. "
                         "Run `python cli.py prepare` again.")
    return np.load(EMB_FILE), meta


# ----------------------------------------------------------------------------- engine
def minmax(scores: np.ndarray) -> np.ndarray:
    """Eq. 3.2. If every candidate has the same score, all normalized scores are 0."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo == 0:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def percentile_rank(x, reference):
    """Eq. 3.7: share of development queries with a lower value, ties counted as half."""
    below = sum(1 for r in reference if r < x)
    equal = sum(1 for r in reference if r == x)
    return (below + 0.5 * equal) / len(reference)


class Engine:
    def __init__(self, docs, embeddings):
        import bm25s
        self.docs = docs
        self.ids = [d.doc_id for d in docs]
        self.index_of = {d: i for i, d in enumerate(self.ids)}
        self.by_id = {d.doc_id: d for d in docs}
        self.tokens = [tokenize(d.abstract) for d in docs]
        self.N = len(docs)
        self.bm25 = bm25s.BM25(method="lucene", k1=config.BM25_K1, b=config.BM25_B)
        self.bm25.index(self.tokens, show_progress=False)
        self.df = Counter(t for toks in self.tokens for t in set(toks))
        self.emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.emb = self.emb / np.where(norms == 0, 1, norms)

    # -- raw scores over the candidate set D_q (every abstract except the query) --
    def candidates(self, qid):
        return [d for d in self.ids if d != qid]

    def bm25_scores(self, qid):
        """BM25S's Lucene variant omits the constant (k1+1) factor of the textbook
        formula; this multiplies every score by the same constant, so rankings and
        min-max normalized scores are identical."""
        q = self.index_of[qid]
        s = np.asarray(self.bm25.get_scores(self.tokens[q]), dtype=float)
        return np.delete(s, q)

    def sbert_scores(self, qid):
        q = self.index_of[qid]
        s = self.emb @ self.emb[q]
        return np.delete(s.astype(float), q)

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)   # Eq. 3.1

    # -- query signals (Section 3.2.5.2 / 4.2.2.6) --
    def query_length(self, qid):
        return len(self.tokens[self.index_of[qid]])           # Eq. 3.5

    def term_specificity(self, qid):
        """Eq. 3.6: mean IDF of unique query terms that occur in at least one other
        abstract (document frequency >= 2)."""
        terms = [t for t in set(self.tokens[self.index_of[qid]]) if self.df[t] >= 2]
        if not terms:
            return 0.0
        return sum(self.idf(t) for t in terms) / len(terms)

    def adaptive_alpha(self, qid, ref):
        """Eqs. 3.7-3.8. `ref` holds the development queries' lengths and specificities."""
        p_len = percentile_rank(self.query_length(qid), ref["length"])
        p_spec = percentile_rank(self.term_specificity(qid), ref["specificity"])
        alpha = min(config.ALPHA_MAX, max(config.ALPHA_MIN, (p_len + p_spec) / 2))
        return alpha, p_len, p_spec

    # -- rankings --
    def _rank(self, qid, scores):
        cands = self.candidates(qid)
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i]))
        return [(cands[i], float(scores[i])) for i in order]   # ties: ascending doc_id

    def rank_bm25(self, qid):
        return self._rank(qid, self.bm25_scores(qid))

    def rank_sbert(self, qid):
        return self._rank(qid, self.sbert_scores(qid))

    def rank_hybrid(self, qid, alpha):
        h = alpha * minmax(self.bm25_scores(qid)) + (1 - alpha) * minmax(self.sbert_scores(qid))
        return self._rank(qid, h)

    def run_all(self, qid, alpha_star, ref):
        """The four final configurations for one query, plus the adaptive details."""
        a_q, p_len, p_spec = self.adaptive_alpha(qid, ref)
        runs = {
            "BM25": self.rank_bm25(qid),
            "SBERT": self.rank_sbert(qid),
            "Fixed-Weight Hybrid": self.rank_hybrid(qid, alpha_star),
            "Query-Adaptive Hybrid": self.rank_hybrid(qid, a_q),
        }
        info = {"length": self.query_length(qid), "specificity": self.term_specificity(qid),
                "p_length": p_len, "p_specificity": p_spec, "alpha_q": a_q}
        return runs, info
