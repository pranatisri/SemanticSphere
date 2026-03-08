from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score

from .config import (
    CLUSTERING_LABELS_PATH,
    CLUSTERING_K_REPORT_PATH,
    CLUSTERING_MODEL_PATH,
    CLUSTERING_K_SWEEP,
    EMBEDDINGS_MATRIX_PATH,
    FUZZINESS_M,
    N_CLUSTERS,
)


class FuzzyClusterModel:
    """
    Fuzzy C-means clustering over document embeddings.

    Each document obtains a soft membership distribution over K clusters,
    which is critical for capturing overlapping semantic structure
    (e.g., a document that is both 'politics' and 'law').
    """

    def __init__(self) -> None:
        self.membership: np.ndarray | None = None  # shape (K, N)
        self.cluster_labels: List[str] | None = None

    def _load_embeddings(self) -> np.ndarray:
        if not EMBEDDINGS_MATRIX_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {EMBEDDINGS_MATRIX_PATH}. "
                "Ensure the embedding index has been built."
            )
        return np.load(EMBEDDINGS_MATRIX_PATH)

    def _load_from_disk(self) -> bool:
        if not CLUSTERING_MODEL_PATH.exists():
            return False
        self.membership = np.load(CLUSTERING_MODEL_PATH)

        if CLUSTERING_LABELS_PATH.exists():
            with CLUSTERING_LABELS_PATH.open("r", encoding="utf-8") as f:
                self.cluster_labels = json.load(f)
        return True

    def _build_from_scratch(self) -> None:
        embeddings = self._load_embeddings()

        # Transpose for skfuzzy.cmeans: data shape (features, samples)
        data = embeddings.T

        # ---- K selection justification (saved to disk) ----
        # The assignment asks you to choose a number of clusters and justify it.
        # We do a lightweight "K sweep" using silhouette score on hard labels
        # (argmax over fuzzy memberships). This isn't perfect for fuzzy clustering,
        # but it's a standard, simple diagnostic that provides quantitative support.
        k_candidates: List[int] = []
        for part in CLUSTERING_K_SWEEP.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                k_candidates.append(int(part))
            except ValueError:
                continue

        k_report = {"metric": "silhouette_cosine_on_argmax_membership", "scores": {}}
        for k in sorted(set([k for k in k_candidates if k >= 2])):
            try:
                _, u_k, _, _, _, _, _ = fuzz.cluster.cmeans(
                    data=data,
                    c=k,
                    m=FUZZINESS_M,
                    error=1e-5,
                    maxiter=500,
                    init=None,
                )
                hard_labels_k = np.argmax(u_k, axis=0)
                score_k = silhouette_score(embeddings, hard_labels_k, metric="cosine")
                k_report["scores"][str(k)] = float(score_k)
            except Exception:
                # Some K choices can fail to converge; we just omit them.
                continue

        CLUSTERING_K_REPORT_PATH.write_text(
            json.dumps(k_report, indent=2), encoding="utf-8"
        )

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data=data,
            c=N_CLUSTERS,
            m=FUZZINESS_M,
            error=1e-5,
            maxiter=1000,
            init=None,
        )

        # u has shape (K, N)
        self.membership = u
        np.save(CLUSTERING_MODEL_PATH, self.membership)

        # Optional: derive simple labels such as "Cluster 0", etc.
        self.cluster_labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]
        with CLUSTERING_LABELS_PATH.open("w", encoding="utf-8") as f:
            json.dump(self.cluster_labels, f)

        # Silhouette score justification (using hard labels from argmax).
        hard_labels = np.argmax(u, axis=0)
        sil_score = silhouette_score(embeddings, hard_labels, metric="cosine")
        # We simply log this value in a text file for inspection and justification.
        report_path: Path = CLUSTERING_MODEL_PATH.with_suffix(".txt")
        with report_path.open("w", encoding="utf-8") as f:
            f.write(
                "Fuzzy C-means clustering report\n"
                f"Number of clusters: {N_CLUSTERS}\n"
                f"Fuzziness (m): {FUZZINESS_M}\n"
                f"Silhouette score (cosine): {sil_score:.4f}\n"
            )

    def ensure_ready(self) -> None:
        if self.membership is not None:
            return
        if not self._load_from_disk():
            self._build_from_scratch()

    def get_membership_for_doc(self, doc_index: int) -> np.ndarray:
        assert self.membership is not None, "Clustering model not initialized."
        return self.membership[:, doc_index]

    def dominant_cluster_for_embedding(self, emb: np.ndarray) -> int:
        """
        Estimate the dominant cluster for a new query embedding.

        For simplicity, we approximate it by finding the nearest document
        in the embedding space and reusing its cluster distribution.
        In a production system you could project the embedding using the
        learned cluster centers directly.
        """
        # Caller is expected to map this via nearest-neighbor search.
        raise NotImplementedError(
            "dominant_cluster_for_embedding should be implemented via pipeline, "
            "using nearest neighbor membership as a proxy."
        )

