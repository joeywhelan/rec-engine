# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Jupyter notebook demonstration of a production-grade product recommendation engine on Elasticsearch Serverless. The central feature is `query_vector_builder.lookup` (ES 9.4), which eliminates the traditional two-request pattern for stored-vector KNN search. The full pipeline composes lookup → Jina Reranker v3 → Painless sponsorship boost into a single Elasticsearch API call.

## Environment Setup

Dependencies are managed with `uv` (Python 3.12 required):

```bash
uv sync
```

Infrastructure is provisioned via Terraform:

```bash
cp terraform/terraform.tfvars.sample terraform/terraform.tfvars  # fill in credentials
terraform -chdir=terraform init -upgrade -input=false
terraform -chdir=terraform apply -auto-approve
```

Generate the `.env` file from Terraform outputs:

```bash
cat > .env << EOF
ELASTIC_CLOUD_API_KEY=$(terraform -chdir=terraform output -raw elastic_cloud_api_key)
ELASTIC_CLOUD_ID=$(terraform -chdir=terraform output -raw elastic_cloud_id)
HF_TOKEN=$(terraform -chdir=terraform output -raw hf_token)
EOF
```

Tear down:

```bash
terraform -chdir=terraform destroy -auto-approve
```

## Running the Notebook

```bash
uv run jupyter notebook demo.ipynb
```

Run cells top-to-bottom; the notebook is designed for linear execution.

## Architecture

The notebook is structured in 7 sections that build progressively:

| Section | Purpose |
|---|---|
| 1 | Provision Elasticsearch Serverless via Terraform, write credentials to `.env` |
| 2 | Load 1,000 ESCI products; synthesize co-purchase, sponsorship, and stock signals; embed via EIS; pick source SKU; bulk index |
| 3 | **Before:** 2-request pattern (GET vector → KNN search with client payload) |
| 4 | **After:** `query_vector_builder.lookup` — same result, 1 request, 0KB client vector payload |
| 5 | **Full pipeline:** lookup + Jina Reranker v3 + Painless sponsorship boost, single API call |
| 6 | Steady-state latency benchmark (two-request vs. lookup) + side-by-side comparison of all three approaches |
| 7 | Teardown (destroy Terraform environment) |

**Data flow:**
```
ESCI stream → augment with synthetic signals → embed via EIS (Jina v5-text-small) → ES Serverless
                                                                                           ↓
query: knn(lookup) → text_similarity_reranker(Jina Reranker v3) → rescorer(Painless boost)
```

**Embedding vs. reranking split:** Both models are pre-registered EIS endpoints — no explicit registration needed. Embeddings are called at ingest time (batch, one per product) via `es.inference.inference()` and stored as `dense_vector`. In Elasticsearch, `dense_vector` fields are excluded from `_source` by default — Section 3 overrides this with `_source_includes=["embedding"]` to demonstrate the two-request pattern. Jina Reranker v3 is called server-side at query time inside the `text_similarity_reranker` retriever — no client round trip at query time.

**Synthetic data:** `co_purchase_titles`, `is_sponsored`, and `in_stock` fields are synthesized. Sponsorship rate is ~30%, weighted toward high-frequency brands. See `assets/plan.md` for the production replacement strategy for each.

## Key Files

- `demo.ipynb` — the main notebook (the deliverable)
- `assets/article.md` — the written walkthrough of the demo (deliverable); section banner images live in `assets/images/section1.png … section7.png`, with their generation prompts in `assets/prompts/prompt_section1.txt … prompt_section7.txt`
- `index.html` — self-contained dark-theme slide deck for presenting the demo live (deliverable); references the section images
- `assets/plan.md` — detailed specification for every notebook section, including exact code snippets and design rationale; consult this before adding or changing notebook sections
- `terraform/` — Elastic Cloud Serverless project provisioning
- `src/rec_engine/ingest.py` — `load_products`, `augment`, `embed_batch`, `create_index`, `bulk_index`
- `src/rec_engine/helpers.py` — `print_source`, `print_recommendations`, `compare_results`, `benchmark`, `print_benchmark`

## ES 9.4 Feature: `query_vector_builder.lookup`

The key pattern replacing the two-request baseline:

```python
# Old: GET embedding → send 4KB vector back in search body
# New: server resolves vector internally
"query_vector_builder": {
    "lookup": {
        "index": INDEX_NAME,
        "id":    source_sku,
        "path":  "embedding"
    }
}
```

This composes transparently with `text_similarity_reranker` and `rescorer` retrievers — Section 5 shows the full nesting.

## Design Constraints

- **Progressive disclosure.** Each of Sections 3→4→5 adds exactly one concept. Do not conflate them.
- **Single source SKU.** Sampled in Section 2, reused verbatim across 3, 4, 5, and 6.
- **Correctness assertion in Section 4.** The `ids_before == ids_after` check is intentional — it proves `lookup` is a drop-in replacement, not a different algorithm. Keep it.
- **Sponsorship boost is post-rerank only.** Any item that survived KNN + rerank is already relevant — no additional relevance floor is needed in the Painless script.
- **Rescore weights.** `rescore_query_weight: 0.15` — the Painless script contributes only the sponsorship delta; `query_weight` is left at its default (1.0) so reranker scores pass through untouched. Do not set `query_weight: 0.0` as that discards reranker scores entirely.
