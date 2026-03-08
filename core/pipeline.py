from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .clustering import FuzzyClusterModel
from .config import TOP_K_RESULTS
from .embeddings import EmbeddingIndex
from .semantic_cache import SemanticCache


@dataclass
class SemanticSearchPipeline:
    """
    High-level orchestration of:
    - embedding model + FAISS index
    - fuzzy clustering membership
    - semantic cache
    """

    index: EmbeddingIndex
    clustering: FuzzyClusterModel
    cache: SemanticCache

    def ensure_ready(self) -> None:
        self.index.ensure_ready()
        self.clustering.ensure_ready()

    def _dominant_cluster_for_doc(self, doc_index: int) -> int:
        membership = self.clustering.get_membership_for_doc(doc_index)
        return int(np.argmax(membership))

    def search(
        self, query: str
    ) -> Tuple[bool, Optional[str], Optional[float], List[Dict], Optional[int]]:
        """
        Full semantic search with cache involvement.

        Returns:
            cache_hit, matched_query, similarity_score, results, dominant_cluster
        """
        self.ensure_ready()

        query_emb = self.index.encode_query(query)  # shape (1, dim)
        query_vec = query_emb[0]

        # Approximate query cluster via nearest neighbor membership.
        distances, indices = self.index.search(query_emb, top_k=1)
        nn_index = int(indices[0])
        dominant_cluster = self._dominant_cluster_for_doc(nn_index)

        # 1) Try semantic cache within that cluster
        hit, entry, sim = self.cache.lookup(query_vec, dominant_cluster)
        if hit and entry is not None:
            return True, entry.query, sim, entry.result, dominant_cluster

        # 2) FAISS search for fresh results
        distances, indices = self.index.search(query_emb, top_k=TOP_K_RESULTS)

        results: List[Dict] = []
        for score, idx in zip(distances, indices):
            doc = self.index.get_document(int(idx))
            snippet = doc.cleaned_text[:400] + ("..." if len(doc.cleaned_text) > 400 else "")
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "score": float(score),
                    "text_snippet": snippet,
                    "category": doc.category,
                }
            )

        # 3) Add to cache for future semantically similar queries
        self.cache.add(query=query, embedding=query_vec, cluster_id=dominant_cluster, result=results)

        return False, None, None, results, dominant_cluster


_GLOBAL_PIPELINE: Optional[SemanticSearchPipeline] = None


def get_global_pipeline() -> SemanticSearchPipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = SemanticSearchPipeline(
            index=EmbeddingIndex(),
            clustering=FuzzyClusterModel(),
            cache=SemanticCache(),
        )
    return _GLOBAL_PIPELINE

