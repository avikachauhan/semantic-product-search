# Smart Product Search (semantic / hybrid)

A small but complete product search engine that understands *intent*, not just keywords.

| Query | Finds |
|---|---|
| `keep drinks cold on a hike` | Insulated water bottle |
| `help me sleep better` | Weighted blanket |
| `noise cancelling headphones for travel under 200` | Headphones, filtered to ≤ $200 |
| `camping gear under $70` | Sleeping bag, headlamp, ... |

## How it works

```
query ──► query parser ──► filters (price, rating, category)
              │                         │
              ▼                         ▼
       cleaned text ──► dense embeddings (cosine) ─┐
                    └─► BM25 keyword score ────────┼─► weighted fusion ─► rating tie-break ─► top-k
```

* **Query parser** (`query_parser.py`) – turns "under $100", "between 50 and 200", "4 stars" into structured filters.
* **Dense retrieval** (`embedder.py`) – `sentence-transformers` (MiniLM) when installed, otherwise an offline **LSA** fallback (TF-IDF + SVD) so the project runs with zero downloads.
* **BM25** (`bm25.py`) – exact-term matching for model names, brands and rare words.
* **Fusion** (`engine.py`) – `alpha * semantic + (1 - alpha) * keyword`, plus a relevance floor so unrelated queries return *no results* instead of noise.
* Every result includes `semantic_score`, `keyword_score` and `matched_terms` for explainability.

## Run it

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v     # tests
python cli.py "gift for a toddler"          # CLI (or no args for a REPL)
uvicorn api:app --reload                    # API + web UI at http://localhost:8000
```

For much better semantics, install the neural backend:

```bash
pip install sentence-transformers           # auto-detected on next start
```

API: `GET /search?q=...&k=10&category=Toys&max_price=50&min_rating=4&alpha=0.65`, `GET /categories`, `GET /health`.

## Use your own catalog

Replace `data/products.csv` (columns: `id,title,description,category,brand,price,rating`).
For catalogs beyond ~100k items, swap the NumPy dot product in `engine.py` for an ANN index (FAISS, hnswlib, pgvector).

## Known limits / next steps

* The LSA fallback is trained on the catalog only, so synonyms it hasn't seen won't match. Use sentence-transformers for real-world quality.
* Category words can match generically (e.g. "home" matches the Home category).
* Ideas: cross-encoder re-ranking, click-based learning-to-rank, typo tolerance, faceted filters, embedding cache on disk.
