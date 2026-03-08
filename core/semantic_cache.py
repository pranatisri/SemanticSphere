from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DEFAULT_SIMILARITY_THRESHOLD


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are expected to be L2-normalized already
    return float(np.dot(a, b.T))


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray  # shape (dim,)
    cluster_id: Optional[int]
    result: List[Dict]


class SemanticCache:
    """
    A simple in-memory semantic cache keyed by embedding similarity.

    Instead of exact text matches, we:
    - embed each incoming query,
    - compare it with cached query embeddings within the same dominant cluster,
    - treat it as a cache hit if cosine similarity exceeds a configurable threshold.
    """

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold: float = similarity_threshold
        self._entries: List[CacheEntry] = []
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _iter_candidates(
        self, cluster_id: Optional[int]
    ) -> List[Tuple[int, CacheEntry]]:
        if cluster_id is None:
            return list(enumerate(self._entries))
        return [
            (i, e)
            for i, e in enumerate(self._entries)
            if e.cluster_id == cluster_id
        ]

    def lookup(
        self, query_embedding: np.ndarray, cluster_id: Optional[int]
    ) -> Tuple[bool, Optional[CacheEntry], Optional[float]]:
        """
        Try to find a semantically similar cached query within the same cluster.

        Returns:
            (hit, entry, similarity_score)
        """
        best_sim = -1.0
        best_entry: Optional[CacheEntry] = None

        for _, entry in self._iter_candidates(cluster_id):
            sim = cosine_similarity(query_embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.similarity_threshold:
            self._hit_count += 1
            return True, best_entry, best_sim

        self._miss_count += 1
        return False, None, None

    def add(
        self,
        query: str,
        embedding: np.ndarray,
        cluster_id: Optional[int],
        result: List[Dict],
    ) -> None:
        self._entries.append(
            CacheEntry(
                query=query,
                embedding=embedding.copy(),
                cluster_id=cluster_id,
                result=result,
            )
        )

    def clear(self) -> None:
        self._entries.clear()
        self._hit_count = 0
        self._miss_count = 0

    def get_stats(self) -> Dict[str, float]:
        total_entries = len(self._entries)
        total_lookups = self._hit_count + self._miss_count
        hit_rate = (
            float(self._hit_count) / float(total_lookups)
            if total_lookups > 0
            else 0.0
        )
        return {
            "total_entries": total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
        }

