# Notebook Design Plan: Vector-Based Product Recommendation Engine
## Elasticsearch Serverless + Jina AI + `query_vector_builder.lookup`

---

## Overview

This notebook demonstrates a production-grade product recommendation engine built on
Elasticsearch Serverless, using Jina Embeddings v5-text-small with behavioral signals
and Jina Reranker v3 — all composed into a single Elasticsearch API call. The central
feature showcase is `query_vector_builder.lookup`, introduced in Elasticsearch 9.4,
which eliminates the traditional two-request pattern for stored-vector search.

**Target audience:** Elasticsearch developers and e-commerce engineers evaluating
vector search for recommendation use cases.

**Notebook format:** Linear, executable top-to-bottom. Each section has a narrative
markdown cell followed by code. The primary evidence for `lookup` efficiency is
structural and noise-free: request count reduction (2 → 1), payload elimination (~8 KB
→ 0 KB on the wire), and identical result IDs (`ids_before == ids_after`). Section 6
adds a supporting steady-state latency benchmark (p50 / p95, 100 interleaved iterations,
10 warmup rounds discarded) for the two-request vs. lookup patterns only. Cache-clear is
not available on Serverless, so all measurements are cache-warm and explicitly labelled
as such. The Section 5 full pipeline is excluded from the benchmark — its latency is
inference-bound (Jina Reranker v3) and independent of the lookup optimization.

---

## Infrastructure

- **Elasticsearch:** Elastic Serverless, provisioned via Terraform
- **Embeddings:** Jina Embeddings v5-text-small (`.jina-embeddings-v5-text-small`) —
  pre-registered EIS endpoint; called at ingest time via `es.inference.inference()`;
  vectors stored as 1024-dim `dense_vector`, excluded from `_source` by default
- **Reranker:** Jina Reranker v3 (`.jina-reranker-v3`) — pre-registered EIS endpoint;
  used inside the `text_similarity_reranker` retriever at query time
- **Dataset:** Amazon ESCI Shopping Queries Dataset — English locale, 1,000 products
  streamed via HuggingFace `datasets` library
- **Augmentation:** Synthetic co-purchase titles, sponsored brand flags (~30%), and
  in-stock flags, clearly labelled throughout

---

## Jina Model Notes

| Role | Model | Dims | Called from |
|---|---|---|---|
| Embedding | `.jina-embeddings-v5-text-small` | 1024 | EIS at ingest (`es.inference.inference()`) |
| Reranking | `.jina-reranker-v3` | n/a | EIS at query time (inside `text_similarity_reranker`) |

Both models are pre-registered EIS endpoints — no `es.inference.put()` call needed.
Embedding happens once per product at ingest — a batch operation with `batch_size=16`.
Reranking happens at query time, fully inside the Elasticsearch API boundary.

---

## Section Structure

---

### Section 1 — Create Elastic Serverless Environment

Runs `terraform apply`, writes `ELASTIC_CLOUD_ID`, `ELASTIC_CLOUD_API_KEY`, and
`HF_TOKEN` to `.env`.

```bash
%%bash
terraform -chdir=terraform init -upgrade -input=false > /dev/null
terraform -chdir=terraform apply -auto-approve > /dev/null

cat > .env << EOF
ELASTIC_CLOUD_API_KEY=$(terraform -chdir=terraform output -raw elastic_cloud_api_key)
ELASTIC_CLOUD_ID=$(terraform -chdir=terraform output -raw elastic_cloud_id)
HF_TOKEN=$(terraform -chdir=terraform output -raw hf_token)
EOF
```

---

### Section 2 — Ingest Data

Combines all data preparation into a single cell: load, augment, embed, pick source
SKU, and index. The source SKU printed here is reused unchanged in Sections 3–6.

**Config and client init:**

```python
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from rec_engine.ingest import load_products, augment, bulk_index
from rec_engine.helpers import print_source

load_dotenv()

es = Elasticsearch(
    cloud_id=os.environ["ELASTIC_CLOUD_ID"],
    api_key=os.environ["ELASTIC_CLOUD_API_KEY"]
)

INDEX_NAME             = "products"
EMBEDDING_INFERENCE_ID = ".jina-embeddings-v5-text-small"
RERANKER_INFERENCE_ID  = ".jina-reranker-v3"
```

**Pipeline:**

```python
products = augment(load_products(), es, EMBEDDING_INFERENCE_ID)
source = products.sample(1, random_state=19).iloc[0]
print_source(source)

bulk_index(es, INDEX_NAME, products)
```

**`load_products(n=1_000)`** — streams the ESCI dataset, filters to English locale with
non-empty title and description, deduplicates by `product_id`, samples `n` products.

**`augment(products_df, es, inference_id)`** — adds four columns:

| Column | How | Production replacement |
|---|---|---|
| `co_purchase_titles` | Same-brand cluster sample (n=3) | Transaction log co-purchase graph |
| `is_sponsored` | ~30% of products, weighted by brand frequency | Sponsorship/contract data |
| `in_stock` | ~90% True, random | Inventory system |
| `product_text` | Concatenation of title, brand, description, bullet points, co-purchase titles | Same |
| `embedding` | Batch EIS call to `.jina-embeddings-v5-text-small`, `batch_size=16` | Same |

**`bulk_index(es, index_name, products_df)`** — calls `create_index` then bulk loads.
`dense_vector` fields are excluded from `_source` by default in Elasticsearch.
Section 3 retrieves the embedding explicitly via `_source_includes=["embedding"]`.

**`print_source(source)`** — prints SKU, title, brand, sponsored flag, in-stock flag
from the pandas Series. Does not display `product_text` or the embedding.

---

### Section 3 — The Before: Two Round Trips

Establishes the baseline that `query_vector_builder.lookup` replaces.

```python
from rec_engine.helpers import print_recommendations

doc = es.get(index=INDEX_NAME, id=source["product_id"], _source_includes=["embedding"])
stored_vector = doc["_source"]["embedding"]

print(f"Request 1 — GET (vector fetch)")
print(f"  Dims: {len(stored_vector)}")
print(f"  Payload: ~{len(stored_vector) * 4 / 1024:.1f} KB returned to client")

results_before = es.search(
    index=INDEX_NAME,
    body={
        "knn": {
            "field":          "embedding",
            "query_vector":   stored_vector,
            "k":              10,
            "num_candidates": 50,
            "filter": {"bool": {
                "must":     [{"term": {"in_stock": True}}],
                "must_not": [{"term": {"sku": source["product_id"]}}]
            }}
        },
        "_source": ["sku", "title", "brand", "is_sponsored"]
    }
)

print(f"\nRequest 2 — KNN search")
print(f"  Payload: ~{len(stored_vector) * 4 / 1024:.1f} KB sent to cluster")

print_recommendations(results_before, label="Before — two requests, raw KNN")
```

The vector (1024 dims × 4 bytes = ~4KB) crosses the network twice: once returned to
the client on the GET, once sent back in the search body. Latency is measured
separately in Section 6 (steady-state benchmark), not timed inline here.

---

### Section 4 — The After: `query_vector_builder.lookup`

```python
results_after = es.search(
    index=INDEX_NAME,
    body={
        "knn": {
            "field": "embedding",
            "query_vector_builder": {
                "lookup": {
                    "index": INDEX_NAME,
                    "id":    source["product_id"],
                    "path":  "embedding"
                }
            },
            "k":              10,
            "num_candidates": 50,
            "filter": {"bool": {
                "must":     [{"term": {"in_stock": True}}],
                "must_not": [{"term": {"sku": source["product_id"]}}]
            }}
        },
        "_source": ["sku", "title", "brand", "is_sponsored"]
    }
)

print(f"Request count: 1")
print(f"Client payload: 0 KB (no vector sent)")

print_recommendations(results_after, label="After — single request, lookup")

ids_before = [h["_id"] for h in results_before["hits"]["hits"]]
ids_after  = [h["_id"] for h in results_after["hits"]["hits"]]
print(f"Results identical: {ids_before == ids_after}")
```

The `ids_before == ids_after` assertion is intentional — it proves `lookup` is a
drop-in replacement, not a different algorithm. Keep it.

---

### Section 5 — Full Pipeline: lookup + Rerank + Sponsorship Boost

Two cells. The first shows results after reranking but before the boost; the second
shows the full pipeline. This makes the contribution of each layer visible separately.

**Pipeline anatomy:**

```
Single API call
│
└── rescorer retriever
    │   Painless script_score: sponsorship boost (+0.15 delta)
    │   rescore_query_weight: 0.15 — sponsorship delta (query_weight left at default 1.0)
    │
    └── text_similarity_reranker retriever
        │   Jina Reranker v3 (cross-encoder, via EIS)
        │   rank_window_size: 10
        │   Cross-attention sees source + candidate together — richer than cosine
        │
        └── knn retriever
                query_vector_builder.lookup → embedding field
                k=10, num_candidates=50
                filter: in_stock=true, exclude source SKU
```

**Cell 1 — rerank only (before boost):**

```python
results_reranked = es.search(
    index=INDEX_NAME,
    body={
        "retriever": {
            "text_similarity_reranker": {
                "retriever": {
                    "knn": {
                        "field": "embedding",
                        "query_vector_builder": {
                            "lookup": {
                                "index": INDEX_NAME,
                                "id":    source["product_id"],
                                "path":  "embedding"
                            }
                        },
                        "k":              10,
                        "num_candidates": 50,
                        "filter": {"bool": {
                            "must":     [{"term": {"in_stock": True}}],
                            "must_not": [{"term": {"sku": source["product_id"]}}]
                        }}
                    }
                },
                "field":            "product_text",
                "inference_id":     RERANKER_INFERENCE_ID,
                "inference_text":   source["product_text"],
                "rank_window_size": 10
            }
        },
        "_source": ["sku", "title", "brand", "is_sponsored"]
    }
)

print_recommendations(results_reranked, label="After rerank — before boost")
```

**Cell 2 — full pipeline with sponsorship boost:**

```python
results_full = es.search(
    index=INDEX_NAME,
    body={
        "retriever": {
            "rescorer": {
                "retriever": {
                    "text_similarity_reranker": {
                        "retriever": {
                            "knn": {
                                "field": "embedding",
                                "query_vector_builder": {
                                    "lookup": {
                                        "index": INDEX_NAME,
                                        "id":    source["product_id"],
                                        "path":  "embedding"
                                    }
                                },
                                "k":              10,
                                "num_candidates": 50,
                                "filter": {"bool": {
                                    "must":     [{"term": {"in_stock": True}}],
                                    "must_not": [{"term": {"sku": source["product_id"]}}]
                                }}
                            }
                        },
                        "field":          "product_text",
                        "inference_id":   RERANKER_INFERENCE_ID,
                        "inference_text": source["product_text"],
                        "rank_window_size": 10
                    }
                },
                "rescore": {
                    "window_size": 10,
                    "query": {
                        "rescore_query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "return doc['is_sponsored'].value ? 1.0 : 0.0;",
                                }
                            }
                        },
                        "rescore_query_weight": 0.15
                    }
                }
            }
        },
        "_source": ["sku", "title", "brand", "is_sponsored"]
    }
)

print_recommendations(results_full, label="Full pipeline — lookup + rerank + boost")
```

**Boost design:** `rescore_query_weight: 0.15` adds a fixed delta for sponsored items.
`query_weight` is left at its default (1.0), so the reranker score passes through
unmodified for all items. Any item that survived KNN + rerank is already relevant —
no additional floor check is needed. Non-sponsored scores are exact reranker scores.
Sponsored scores are reranker_score + 0.15.

---

### Section 6 — Full Comparison

Two cells. The first runs the steady-state latency benchmark for the two-request vs.
lookup patterns; the second prints the three-way result comparison.

**Cell 1 — latency benchmark:**

```python
from rec_engine.helpers import benchmark, print_benchmark

def run_before():
    doc = es.get(index=INDEX_NAME, id=source["product_id"], _source_includes=["embedding"])
    v = doc["_source"]["embedding"]
    es.search(index=INDEX_NAME, body={"knn": {
        "field": "embedding", "query_vector": v, "k": 10, "num_candidates": 50,
        "filter": {"bool": {"must": [{"term": {"in_stock": True}}],
                            "must_not": [{"term": {"sku": source["product_id"]}}]}}},
        "_source": ["sku", "title", "brand", "is_sponsored"]})

def run_after():
    es.search(index=INDEX_NAME, body={"knn": {
        "field": "embedding",
        "query_vector_builder": {"lookup": {"index": INDEX_NAME, "id": source["product_id"], "path": "embedding"}},
        "k": 10, "num_candidates": 50,
        "filter": {"bool": {"must": [{"term": {"in_stock": True}}],
                            "must_not": [{"term": {"sku": source["product_id"]}}]}}},
        "_source": ["sku", "title", "brand", "is_sponsored"]})

stats = benchmark(
    {"Two requests (before)": run_before, "lookup (after)": run_after},
    warmup=10,
    iterations=100,
)
print_benchmark(stats, baseline="Two requests (before)")
```

The full pipeline (Section 5) is deliberately excluded from the benchmark — its latency
is inference-bound (Jina Reranker v3) and independent of the lookup optimization.
Cache-clear is unavailable on Serverless, so all measurements are cache-warm.

**Cell 2 — three-way result comparison:**

```python
from rec_engine.helpers import compare_results

compare_results(
    before=results_before,
    after=results_after,
    full=results_full,
    source_title=source["product_title"]
)
```

Three-column table: Two Requests | lookup only | Full Pipeline. `✓` marks positions
where lookup matches before (drop-in equivalence). `★` marks sponsored items elevated
by the boost.

---

### Section 7 — Destroy Environment

```bash
%%bash
terraform -chdir=terraform destroy -auto-approve > /dev/null && echo "Done."
rm -f .env
```

---

## Helper Function Specifications

### `print_source(source)` — `helpers.py`

Accepts a pandas Series. Prints SKU (`product_id`), title, brand, sponsored flag,
and in-stock flag. Does not display `product_text` or embedding.

### `print_recommendations(results, label="")` — `helpers.py`

```
[label]
────────────────────────────────────────────────────────────
 1. [0.9823] Ruiqas Jewelry Case ...         ★
 2. [0.9741] RosinKing Earring Storage Box
...

★ = sponsored
```

### `compare_results(before, after, full, source_title)` — `helpers.py`

Three-column fixed-width table. `✓` = lookup result in same position as before.
`★` = sponsored product in full pipeline column.

### `benchmark(patterns, warmup=10, iterations=100)` — `helpers.py`

Accepts a dict of `{label: callable}`. Discards `warmup` rounds, then runs `iterations`
interleaved timed calls per pattern. Returns per-pattern stats (p50, p95, min ms).

### `print_benchmark(stats, baseline=None)` — `helpers.py`

Prints the benchmark table (p50 / p95 / min). When `baseline` is given, also reports the
relative speedup of the other patterns against it.

### `load_products(n=1_000, random_state=42)` — `ingest.py`

Streams `tasksource/esci` from HuggingFace, filters to English locale with non-empty
title and description, deduplicates by `product_id`, buffers `n*3` then samples `n`.

### `augment(products_df, es, inference_id, seed=42)` — `ingest.py`

Adds `co_purchase_titles`, `is_sponsored`, `in_stock`, `product_text`, `embedding`.
Sponsorship rate: 0.30, weighted by brand frequency, `replace=True` sampling.

### `embed_batch(texts, es, inference_id, batch_size=16)` — `ingest.py`

Calls `es.inference.inference(task_type="text_embedding", ...)`. Response key path:
`response["text_embedding"][i]["embedding"]`.

### `create_index(es, index_name)` — `ingest.py`

Creates index with `dense_vector` mapping (`dims=1024`, `similarity=cosine`,
`index=True`). No explicit `_source.excludes` needed — Elasticsearch excludes
`dense_vector` fields from `_source` by default.

### `bulk_index(es, index_name, products_df)` — `ingest.py`

Calls `create_index` then bulk-loads. Document `_id` = `product_id`.
Fields indexed: `sku`, `title`, `brand`, `product_text`, `embedding`,
`is_sponsored`, `in_stock`. Calls `es.indices.refresh()` after bulk load.

---

## Key Design Principles

**Honesty about synthetic data.** Every synthetic field has a production replacement
note in the docstring and in the Section 2 narrative table.

**Progressive complexity.** Sections 3 → 4 → 5 each add exactly one concept.
A reader stopping at Section 4 understands `lookup`. Finishing Section 5 gives
the full production picture.

**Single source SKU.** Sampled once in Section 2, used unchanged across Sections 3,
4, 5, and 6. Makes the before/after comparison unambiguous.

**Correctness assertion.** Section 4 explicitly asserts that `lookup` returns
identical results to the two-request pattern. Confirms drop-in replacement.

**Rescore weight design.** `rescore_query_weight: 0.15` adds a fixed delta for
qualifying sponsored items. `query_weight` is left at its default (1.0) so non-sponsored
items retain their exact reranker scores. Do not set `query_weight: 0.0` — that
discards reranker scores entirely.

---

## File Structure

```
rec-engine/
├── assets/
│   ├── plan.md                  ← this document
│   ├── article.md               ← the written walkthrough (deliverable)
│   ├── images/
│   │   ├── arch.png             ← architecture diagram
│   │   ├── cover.png            ← article cover image
│   │   ├── section1.png … section7.png  ← per-section banner images (900px wide)
│   │   └── step1.png            ← source render for the section banners
│   └── prompts/
│       └── prompt_section1.txt … prompt_section7.txt  ← image-generation prompts per section
├── demo.ipynb                   ← the demo notebook (deliverable)
├── index.html                   ← self-contained slide deck for live presentation (deliverable)
├── src/rec_engine/
│   ├── __init__.py              ← package stub
│   ├── helpers.py               ← print_source, print_recommendations, compare_results, benchmark, print_benchmark
│   └── ingest.py                ← load_products, augment, embed_batch, create_index, bulk_index
├── terraform/                   ← Serverless provisioning
├── README.md
├── LICENSE
├── pyproject.toml               ← managed by uv (Python 3.12)
└── uv.lock
```
