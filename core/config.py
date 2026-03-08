from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Directories for artifacts
DATA_DIR: Final[Path] = BASE_DIR / "data"
EMBEDDINGS_DIR: Final[Path] = BASE_DIR / "embeddings"
CLUSTERING_DIR: Final[Path] = BASE_DIR / "clustering"

DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
CLUSTERING_DIR.mkdir(exist_ok=True)

# Embedding model configuration
EMBEDDING_MODEL_NAME: Final[str] = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS index configuration
FAISS_INDEX_PATH: Final[Path] = EMBEDDINGS_DIR / "faiss_index.bin"
EMBEDDINGS_MATRIX_PATH: Final[Path] = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_PATH: Final[Path] = EMBEDDINGS_DIR / "metadata.json"

# Fuzzy clustering configuration
N_CLUSTERS: Final[int] = int(os.getenv("N_CLUSTERS", "20"))
FUZZINESS_M: Final[float] = float(os.getenv("FUZZINESS_M", "2.0"))
CLUSTERING_MODEL_PATH: Final[Path] = CLUSTERING_DIR / "fcm_membership.npy"
CLUSTERING_LABELS_PATH: Final[Path] = CLUSTERING_DIR / "cluster_labels.json"
CLUSTERING_K_SWEEP: Final[str] = os.getenv("CLUSTERING_K_SWEEP", "10,15,20,25")

# Where we store K-sweep silhouette results (used to justify chosen K).
CLUSTERING_K_REPORT_PATH: Final[Path] = CLUSTERING_DIR / "k_sweep_report.json"

# Semantic cache configuration
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = float(
    os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.85")
)
TOP_K_RESULTS: Final[int] = int(os.getenv("TOP_K_RESULTS", "5"))

