from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.datasets import fetch_20newsgroups

from .config import DATA_DIR


HEADER_PATTERN = re.compile(
    r"^(From|Subject|Lines|Organization|Reply-To|Nntp-Posting-Host|Path|Xref|Newsgroups|Date):.*$",
    re.MULTILINE,
)
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
SIGNATURE_PATTERN = re.compile(r"(--+|__+|\*\*+).*", re.MULTILINE)
QUOTED_LINE_PATTERN = re.compile(r"^(>+).*?$", re.MULTILINE)


@dataclass
class Document:
    doc_id: int
    text: str
    cleaned_text: str
    category: str


def _clean_text(raw: str) -> str:
    """
    Clean a single 20 Newsgroups post.

    We clean aggressively because 20 Newsgroups posts are noisy (email formatting,
    headers, quotes, signatures). Those parts add author/transport artifacts that
    can dominate embeddings and hurt semantic search quality.

    We:
    - strip typical email headers like 'From:', 'Subject:' because they are boilerplate
      and would dominate the semantics without adding topic information.
    - drop email addresses to avoid user-specific noise.
    - remove quoted reply lines starting with '>' to reduce repetition and thread noise.
    - heuristically remove signatures which are often separated by '--' or similar markers.
    - normalize to lowercase and keep only alphabetic characters to reduce sparsity
      while preserving the core topical content.
    - collapse multiple whitespaces so the text is model-friendly.
    """
    text = HEADER_PATTERN.sub(" ", raw)
    text = EMAIL_PATTERN.sub(" ", text)
    text = QUOTED_LINE_PATTERN.sub(" ", text)
    text = SIGNATURE_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = text.lower()
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def load_20newsgroups(cache: bool = True) -> List[Document]:
    """
    Download (if necessary) and return the 20 Newsgroups dataset with cleaned text.

    Results are cached as a simple JSONL-style file for faster subsequent startups.
    """
    cache_path: Path = DATA_DIR / "20newsgroups_cleaned.jsonl"

    if cache and cache_path.exists():
        documents: List[Document] = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                documents.append(
                    Document(
                        doc_id=obj["doc_id"],
                        text=obj["text"],
                        cleaned_text=obj["cleaned_text"],
                        category=obj["category"],
                    )
                )
        return documents

    dataset = fetch_20newsgroups(subset="all", remove=())
    documents = []
    for idx, (text, target) in enumerate(zip(dataset.data, dataset.target)):
        cleaned = _clean_text(text)
        documents.append(
            Document(
                doc_id=idx,
                text=text,
                cleaned_text=cleaned,
                category=dataset.target_names[target],
            )
        )

    if cache:
        with cache_path.open("w", encoding="utf-8") as f:
            for d in documents:
                f.write(
                    json.dumps(
                        {
                            "doc_id": d.doc_id,
                            "text": d.text,
                            "cleaned_text": d.cleaned_text,
                            "category": d.category,
                        }
                    )
                    + "\n"
                )

    return documents

