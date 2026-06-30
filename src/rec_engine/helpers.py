import time
import numpy as np


def benchmark(patterns, warmup=10, iterations=100):
    """
    Time each callable in patterns (dict {name: fn}) over warmup + iterations rounds.
    Patterns are interleaved within each round so all see the same time-distributed
    network conditions.  Returns {name: {p50, p95, min, mean, n}} in milliseconds.
    """
    names = list(patterns)
    samples = {name: [] for name in names}

    for _ in range(warmup):
        for name in names:
            patterns[name]()

    for _ in range(iterations):
        for name in names:
            t0 = time.perf_counter()
            patterns[name]()
            t1 = time.perf_counter()
            samples[name].append((t1 - t0) * 1000)

    return {
        name: {
            "p50":  np.percentile(samples[name], 50),
            "p95":  np.percentile(samples[name], 95),
            "min":  np.min(samples[name]),
            "mean": np.mean(samples[name]),
            "n":    len(samples[name]),
        }
        for name in names
    }


def print_benchmark(stats, baseline=None):
    """Print a latency table (p50 / p95 / min) and p50 delta vs baseline."""
    COL = 24
    print(f"\n{'Pattern':<{COL}}  {'p50 (ms)':>10}  {'p95 (ms)':>10}  {'min (ms)':>10}")
    print("─" * 60)
    for name, s in stats.items():
        print(f"{name:<{COL}}  {s['p50']:>10.1f}  {s['p95']:>10.1f}  {s['min']:>10.1f}")

    if baseline and baseline in stats:
        base_p50 = stats[baseline]["p50"]
        print()
        for name, s in stats.items():
            if name == baseline:
                continue
            delta = base_p50 - s["p50"]
            pct   = (delta / base_p50) * 100 if base_p50 else 0
            sign  = "faster" if delta >= 0 else "slower"
            label = "reduction" if delta >= 0 else "increase"
            print(f"{name}: {abs(delta):.1f} ms {sign} at p50 ({abs(pct):.0f}% {label})")

    print()
    print("Note: Elastic Serverless does not support cache clear — all measurements are")
    print("steady-state (cache-warm). Patterns were interleaved within each round so both")
    print("see identical time-distributed network conditions.")


def print_source(source):
    print("\nSource product")
    print("─" * 60)
    print(f"SKU:       {source['product_id']}")
    print(f"Title:     {source['product_title']}")
    print(f"Brand:     {source['product_brand']}")
    print(f"Sponsored: {source['is_sponsored']}")
    print(f"In stock:  {source['in_stock']}")


def print_recommendations(results, label=""):
    hits = results["hits"]["hits"]
    if label:
        print(label)
    print("─" * 60)
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        flag = " ★" if src.get("is_sponsored") else ""
        print(f"{i:2}. [{hit['_score']:.4f}] {src['title']}{flag}")
    if any(hit["_source"].get("is_sponsored") for hit in hits):
        print("\n★ = sponsored")
    print()


def compare_results(before, after, full, source_title):
    COL = 30

    before_hits = before["hits"]["hits"]
    after_hits  = after["hits"]["hits"]
    full_hits   = full["hits"]["hits"]
    before_ids  = [h["_id"] for h in before_hits]

    def cell(hits, idx, flag=" "):
        if idx >= len(hits):
            return " " * (COL + 2)
        h = hits[idx]
        title = h["_source"]["title"][:20]
        return f"{title:<20} [{h['_score']:.4f}]{flag} "

    print(f"Source: {source_title}\n")
    print(f"{'Rank':<5} {'Two Requests (before)':<{COL+2}} {'lookup only (after)':<{COL+2}} {'Full Pipeline'}")
    print(f"{'─'*4}  {'─'*(COL+2)} {'─'*(COL+2)} {'─'*(COL+2)}")

    for i in range(max(len(before_hits), len(after_hits), len(full_hits))):
        same  = "✓" if i < len(after_hits) and i < len(before_ids) and after_hits[i]["_id"] == before_ids[i] else " "
        spons = "★" if i < len(full_hits) and full_hits[i]["_source"].get("is_sponsored") else " "
        print(f"{i+1:>3}.  {cell(before_hits, i)}{cell(after_hits, i, same)}{cell(full_hits, i, spons)}")

    print("\n✓ = same position as before   ★ = sponsored product")


