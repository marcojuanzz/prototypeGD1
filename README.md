# Retrieval comparison prototype

Experimental platform for *A Comparative Study of Lexical, Semantic, and Query-Adaptive Hybrid Retrieval Models*.
It implements Chapters 3 and 4: BM25, Sentence-BERT, the fixed-weight hybrid, the query-adaptive hybrid, pooling, blind annotation, consensus, and evaluation.

| File | What it does | Manuscript section |
|---|---|---|
| `config.py` | All settings (k1, b, model, α grid, pool depth, seed, annotator names) | 3.2, 4.2 |
| `corpus.py` | Loads `data/corpus.csv`, cleans line breaks, excludes unusable rows | 4.4 |
| `retrieval.py` | Tokenizer, BM25, Sentence-BERT, min-max fusion, ℓ(q), σ(q), percentiles, α_q | 3.2.1–3.2.5, 4.2.2 |
| `judgments.py` | Pools, each annotator's scores, disagreements, consensus | 4.2.3, 4.5.1 |
| `evaluation.py` | nDCG@10, P@5, pool recall, weighted κ, randomization test, Holm | 4.5, 4.6 |
| `cli.py` | The pipeline, in the fixed order of Section 4.2 | 4.2 |
| `app.py` | Web platform: retrieval demo, blind annotation, resolving disagreements, results | 4.3 |

---

## Step 1 – Install (once)

Needs Python 3.10 or newer and internet access for the first run (the model is about 420 MB).

```bash
cd prototype
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

On a laptop without a GPU, installing the CPU build of PyTorch first is smaller and faster:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Step 2 – Put the corpus in place

`data/corpus.csv` already holds your 50 abstracts, with the four PDF leftovers fixed (D03 "offers"/"different", D28 "offline", D44 "different").
If you edit the Google Sheet again: **File → Download → Comma-separated values** on the *Corpus* tab, and save it as `data/corpus.csv`.

The code only needs the columns `doc_id`, `title`, `abstract`, `query_role` (`dev` / `final` / blank) and `status` (`Unusable` rows are excluded).

Set the real annotator names in `config.py` → `ANNOTATORS`.

## Step 3 – Prepare

```bash
python cli.py prepare
```
This loads and cleans the abstracts and reports how many were included or excluded. It counts word pieces with the model's own tokenizer, lists the abstracts truncated at 384, then encodes every abstract once and stores the vectors.

## Step 4 – Development phase

```bash
python cli.py dev
```
This records ℓ(q) and σ(q) of the 5 development queries as the percentile reference. It runs BM25, SBERT and the hybrid at α = 0.1 … 0.9 (11 rankings per query), writes the run files, and builds the development pool.

## Step 5 – Judge the development pool

```bash
python app.py
```
Open **http://127.0.0.1:5000** and choose **Judge dev pool**. Pick your name. Each pair shows the query abstract next to the candidate. You never see which configuration found it or at what rank, and the order is random for each annotator. You can click 2 / 1 / 0 or press the keys.
Both annotators judge alone. Then use **Resolve** (switch to the dev pool) to discuss each disagreement and record the agreed score.

## Step 6 – Select the fixed weight α*

```bash
python cli.py select-alpha
```
This picks the α with the highest mean development nDCG@10. On a tie it takes the value closest to 0.5, then the lower one.
**Short on time for the defense?** `python cli.py select-alpha --provisional 0.5` sets α* without development judgments. The platform marks it as provisional everywhere, so say so in the presentation.

## Step 7 – Freeze and run the final evaluation queries

```bash
python cli.py final
```
This freezes every setting and library version. It runs the four configurations on the 10 final queries, prints each query's ℓ, σ, P_ℓ, P_σ and α_q, writes the run files, and builds the final pool.

## Step 8 – Judge the final pool, then resolve disagreements

Do this in the web app, in the same way as Step 5.

## Step 9 – Evaluate

```bash
python cli.py evaluate
```
This produces:
- mean nDCG@10, P@5 and recall over the judged pool for each configuration
- six pairwise randomization tests per measure (100,000 permutations, Holm-adjusted)
- weighted κ and percent agreement for the dev and final pools
- Spearman correlations of ΔLS with ℓ, σ and α_q, and of ΔAF with |α_q − α*|, plus scatter plots

Everything appears on the **Results** page and is written to `outputs/results/`.
For a quick demo before both annotators finish: `python cli.py evaluate --annotator "Annotator A"` uses one person's scores, and the output is clearly labeled as a demo.

`python cli.py status` shows where you are at any time.

---

## For the defense demo

1. **Retrieval demo** page: pick a final query. The four rankings appear side by side, with ℓ(q), σ(q), both percentile ranks, α_q and α* shown. This is where you show what the query-adaptive weighting does.
2. **Judge** page: show the blind, side-by-side annotation.
3. **Results** page: show that the evaluation runs from start to finish, and say these are pilot numbers.

## Outputs

```
outputs/
  embeddings.npy, embeddings_meta.json   stored vectors + word-piece counts
  state.json                             percentile reference, α*, frozen settings, versions
  runs/  dev_*.tsv, final_*.tsv          run files: query_id, doc_id, rank, score
  pools/ dev_pool.csv, final_pool.csv    pairs to judge
  judgments/                             each annotator's scores + resolutions
  results/                               per-query scores, tests, query analysis, plot, report.json
```

## Implementation notes worth knowing for the manuscript

- **BM25 scores** come from BM25S's Lucene variant. Its IDF is Eq. 3.1. It leaves out the textbook (k1 + 1) factor, which multiplies every score by the same constant. Rankings and the min-max normalized scores used by the hybrids are identical.
- **The stopword list** is the BM25 library's English list (Lucene's 33 words).
- **Term specificity** keeps only terms with a document frequency of at least 2, meaning they appear in another abstract (Section 4.2.2.6).
- **Queries with no relevant document** in their pool are excluded from averages and tests, and they are listed in the output (Section 4.5).
- **Statistical power.** With 10 final queries, the smallest possible p-value of the randomization test is about 0.002, before Holm adjustment. Expect few or no significant differences in the prototype; that is a sample-size limit, not a fault.
