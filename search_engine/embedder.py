"""Text embedding backends.

`auto` uses sentence-transformers (neural embeddings) when installed, otherwise
falls back to LSA (TF-IDF + truncated SVD), which needs only scikit-learn.
Plug in any model by subclassing `Embedder`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .bm25 import tokenize


class Embedder(ABC):
    name: str = "base"
    min_similarity: float = 0.25  # cosine below this (with no keyword hit) is treated as noise

    @abstractmethod
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Embed the corpus (fitting if needed). Rows are L2-normalised."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Embed new texts such as queries. Rows are L2-normalised."""


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = f"sbert:{model_name.split('/')[-1]}"
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts):
        return self.encode(texts)

    def encode(self, texts):
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        )


class LSAEmbedder(Embedder):
    """Latent Semantic Analysis: lightweight, offline, no model download."""

    name = "lsa"
    min_similarity = 0.30

    def __init__(self, dims: int = 64):
        self.dims = dims
        self.vec = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self.svd = None

    def fit_transform(self, texts):
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.dims, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        return normalize(self.svd.fit_transform(X))

    def encode(self, texts):
        assert self.svd is not None, "call fit_transform first"
        return normalize(self.svd.transform(self.vec.transform(list(texts))))


def get_embedder(kind: str = "auto") -> Embedder:
    if kind in ("auto", "sbert"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if kind == "sbert":
                raise
    return LSAEmbedder()
