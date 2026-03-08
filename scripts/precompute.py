"""
Offline precomputation script (recommended before running the API in production).

Why this exists:
- The API lazily builds artifacts on first request to keep startup fast.
- For a realistic system, you'd usually precompute embeddings + FAISS index + clustering once,
  then serve queries with low latency.

Run:
  python scripts/precompute.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on PYTHONPATH when run as a script (Windows-friendly).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pipeline import get_global_pipeline


def main() -> None:
    pipeline = get_global_pipeline()
    pipeline.ensure_ready()
    # We intentionally keep output ASCII-only so it works on default Windows consoles.
    print("Precompute complete.")
    print("- Embeddings + FAISS index are ready.")
    print("- Fuzzy C-means membership matrix is ready.")


if __name__ == "__main__":
    main()

