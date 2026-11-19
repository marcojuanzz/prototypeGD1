"""Experimental settings. Everything here is recorded when the parameters are frozen
(Section 4.2.3). Change values BEFORE running `python cli.py final`, never after."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CORPUS_CSV = DATA_DIR / "corpus.csv"          # exported from the Google Sheet (Corpus tab)

# --- Lexical retrieval (Section 3.2.1 / 4.2.2.2) ---
BM25_K1 = 1.2
BM25_B = 0.75

# --- Semantic retrieval (Section 3.2.2 / 4.2.2.3) ---
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_MAX_TOKENS = 384                          # word pieces; longer abstracts are truncated

# --- Hybrid configurations (Sections 3.2.3 - 3.2.5) ---
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA_MIN, ALPHA_MAX = 0.1, 0.9

# --- Evaluation (Sections 4.2.3, 4.5) ---
POOL_DEPTH = 10            # top-k of each configuration pooled for judging
NDCG_K = 10
PRECISION_K = 5
RELEVANT_THRESHOLD = 1     # a consensus score >= 1 counts as relevant for P@5 and recall
PERMUTATIONS = 100_000
SIGNIFICANCE_LEVEL = 0.05

# --- Reproducibility ---
RANDOM_SEED = 2026

# --- Annotators (independent, blind). Rename to the real annotators. ---
ANNOTATORS = ["Tres Reyes", "Williams"]

CONFIGS = ["BM25", "SBERT", "Fixed-Weight Hybrid", "Query-Adaptive Hybrid"]
