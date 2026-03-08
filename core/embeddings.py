from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from .config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_MATRIX_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from .data_loader import Document, load_20newsgroups


class EmbeddingIndex:
    """
    Wrapper around a sentence-transformer model and a FAISS index.
    """

    def __init__(self) -> None:
        # We intentionally keep this as "object" to avoid importing sentence-transformers
        # at module import time (which can trigger optional TF/Flax imports in Transformers).
        self.model: object | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.documents: List[Document] = []

    def _ensure_model(self):
        if self.model is None:
            # Avoid importing TensorFlow/Flax through Transformers integrations.
            # This prevents common Windows env issues where TF is installed but incompatible
            # with NumPy, breaking the import chain.
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

            from sentence_transformers import SentenceTransformer  # local import by design

            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.model

    def _build_from_scratch(self) -> None:
        """
        Build embeddings and FAISS index from raw data.
        """
        self.documents = load_20newsgroups(cache=True)

        model = self._ensure_model()
        texts = [d.cleaned_text for d in self.documents]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        np.save(EMBEDDINGS_MATRIX_PATH, embeddings)

        with METADATA_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "doc_ids": [d.doc_id for d in self.documents],
                    "categories": [d.category for d in self.documents],
                },
                f,
            )

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def _load_from_disk(self) -> bool:
        if (
            not EMBEDDINGS_MATRIX_PATH.exists()
            or not FAISS_INDEX_PATH.exists()
            or not METADATA_PATH.exists()
        ):
            return False

        embeddings = np.load(EMBEDDINGS_MATRIX_PATH)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        from .data_loader import load_20newsgroups

        # We still need the documents for text snippets.
        self.documents = load_20newsgroups(cache=True)
        return True

    def ensure_ready(self) -> None:
        """
        Ensure embeddings, FAISS index, and documents are loaded.
        """
        if self.index is not None and self.documents:
            return

        if not self._load_from_disk():
            self._build_from_scratch()

    def encode_query(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb.astype("float32")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None, "FAISS index is not initialized."
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

    def get_document(self, idx: int) -> Document:
        return self.documents[idx]

