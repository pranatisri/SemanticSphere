"""
Small helper script to exercise the FastAPI semantic search API and show
real responses, including semantic cache behaviour.

Usage (from repo root, with venv active or via .venv explicitly):

    .\.venv\Scripts\python scripts\test_queries.py
"""

from __future__ import annotations

import json

import httpx


def pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    base_url = "http://127.0.0.1:8000"

    # Give generous timeout on first call, since the API may lazily load models
    # if you skipped running scripts/precompute.py.
    with httpx.Client(timeout=60.0) as client:
        # First query: cold cache, should be a miss.
        q1 = "How do graphics cards improve gaming performance?"
        r1 = client.post(f"{base_url}/query", json={"query": q1})
        data1 = r1.json()
        print("=== First query ===")
        print("cache_hit:", data1.get("cache_hit"))
        print("dominant_cluster:", data1.get("dominant_cluster"))
        print("top_result_category:", data1["result"][0]["category"] if data1["result"] else None)
        print()

        # Second query: paraphrase, should be a semantic cache hit if similarity is high enough.
        q2 = "Explain how GPUs help games run faster"
        r2 = client.post(f"{base_url}/query", json={"query": q2})
        data2 = r2.json()
        print("=== Second query (paraphrase) ===")
        print("cache_hit:", data2.get("cache_hit"))
        print("matched_query:", data2.get("matched_query"))
        print("similarity_score:", data2.get("similarity_score"))
        print("dominant_cluster:", data2.get("dominant_cluster"))
        print()

        # Show cache stats after the two lookups.
        r_stats = client.get(f"{base_url}/cache/stats")
        stats = r_stats.json()
        print("=== Cache stats ===")
        print(pretty(stats))


if __name__ == "__main__":
    main()

