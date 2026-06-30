import random

import pandas as pd
from datasets import load_dataset
from elasticsearch.helpers import bulk


def load_products(n=1_000, random_state=42):
    target = n * 3
    print(f"Streaming ESCI products (target buffer: {target})...")
    ds = load_dataset("tasksource/esci", split="train", streaming=True)
    records, seen_ids = [], set()
    for row in ds:
        if (
            row["product_locale"] == "us"
            and row["product_title"]
            and row["product_description"]
            and row["product_id"] not in seen_ids
        ):
            seen_ids.add(row["product_id"])
            records.append(row)
            if len(records) % 500 == 0:
                print(f"  {len(records)} / {target} buffered", end="\r", flush=True)
            if len(records) >= target:
                break
    products = pd.DataFrame(records)
    products = products.sample(n=n, random_state=random_state).reset_index(drop=True)
    print(f"  Done. {len(products)} products loaded.")
    return products


def synthesize_co_purchases(idx, products_df, brand_to_indices, all_indices, n=3, seed=None):
    """
    SYNTHETIC: samples n products from the same brand cluster as a proxy
    for co-purchase affinity. In production, replace with transaction log data
    or a co-purchase graph (e.g. Amazon SNAP co-purchasing network).
    """
    rng = random.Random(seed or idx)
    brand = products_df.loc[idx, "product_brand"]
    pool = [i for i in brand_to_indices.get(brand, []) if i != idx]
    if len(pool) < n:
        pool = [i for i in all_indices if i != idx]
    sampled = rng.sample(pool, min(n, len(pool)))
    return products_df.loc[sampled, "product_title"].tolist()


def synthesize_sponsorship(products_df, sponsored_rate=0.3, seed=42):
    """
    SYNTHETIC: ~15% of products marked as sponsored, weighted toward
    brands appearing frequently in the dataset (simulating brands with budget).
    In production, replace with your actual sponsorship/contract data.
    """
    brand_counts = products_df["product_brand"].value_counts()
    weights = products_df["product_brand"].map(brand_counts).fillna(1)
    weights = weights / weights.sum()
    n_sponsored = int(len(products_df) * sponsored_rate)
    sponsored_idx = products_df.sample(
        n=n_sponsored, weights=weights, random_state=seed, replace=True
    ).index
    products_df["is_sponsored"] = False
    products_df.loc[sponsored_idx, "is_sponsored"] = True
    return products_df


def build_product_text(row):
    """
    Constructs the text passed to EIS for embedding.

    Behavioral signal (co-purchase) is expressed as natural language so the
    embedding model encodes it semantically. Products with similar co-purchase
    patterns land closer together in vector space, complementing pure
    content similarity.
    """
    co_purchases = ", ".join(row["co_purchase_titles"])
    return (
        f"Product: {row['product_title']}\n"
        f"Brand: {row['product_brand']}\n"
        f"Description: {row['product_description']}\n"
        f"Key features: {row['product_bullet_point']}\n"
        f"Frequently bought with: {co_purchases}"
    ).strip()


def augment(products_df, es, inference_id, seed=42):
    """Add co_purchase_titles, is_sponsored, in_stock, product_text, and embedding columns."""
    rng = random.Random(seed)
    products_df = products_df.copy()

    print("Synthesizing co-purchase signals...")
    brand_to_indices = (
        products_df.groupby("product_brand")
        .apply(lambda g: g.index.tolist(), include_groups=False)
        .to_dict()
    )
    all_indices = products_df.index.tolist()
    products_df["co_purchase_titles"] = [
        synthesize_co_purchases(i, products_df, brand_to_indices, all_indices)
        for i in products_df.index
    ]

    print("Synthesizing sponsorship and stock signals...")
    products_df = synthesize_sponsorship(products_df, seed=seed)
    products_df["in_stock"] = [rng.random() > 0.1 for _ in range(len(products_df))]
    products_df["product_text"] = products_df.apply(build_product_text, axis=1)

    print("Embedding product texts via EIS...")
    products_df["embedding"] = embed_batch(products_df["product_text"].tolist(), es, inference_id)
    print("  Done.")
    return products_df


def embed_batch(texts, es, inference_id, batch_size=16):
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        response = es.inference.inference(
            task_type="text_embedding",
            inference_id=inference_id,
            input=batch,
        )
        all_embeddings.extend(e["embedding"] for e in response["text_embedding"])
        print(f"  {len(all_embeddings)}/{total} embedded", end="\r", flush=True)
    return all_embeddings


def create_index(es, index_name):
    mapping = {
        "mappings": {
            "properties": {
                "sku":          {"type": "keyword"},
                "title":        {"type": "text"},
                "brand":        {"type": "keyword"},
                "product_text": {"type": "text"},
                "embedding": {
                    "type":       "dense_vector",
                    "dims":       1024,
                    "index":      True,
                    "similarity": "cosine",
                },
                "is_sponsored": {"type": "boolean"},
                "in_stock":     {"type": "boolean"},
            }
        }
    }
    es.indices.delete(index=index_name, ignore_unavailable=True)
    es.indices.create(index=index_name, mappings=mapping["mappings"])
    print(f"\nIndex '{index_name}' created.")


def bulk_index(es, index_name, products_df):
    create_index(es, index_name)
    actions = [
        {
            "_index": index_name,
            "_id":    row["product_id"],
            "_source": {
                "sku":          row["product_id"],
                "title":        row["product_title"],
                "brand":        row["product_brand"] if pd.notna(row["product_brand"]) else None,
                "product_text": row["product_text"],
                "embedding":    row["embedding"],
                "is_sponsored": row["is_sponsored"],
                "in_stock":     row["in_stock"],
            },
        }
        for _, row in products_df.iterrows()
    ]
    success, failed = bulk(es, actions, raise_on_error=False)
    es.indices.refresh(index=index_name)
    print(f"Indexed: {success} | Failed: {failed}")
