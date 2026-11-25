"""Hybrid semantic product search.

parse query -> filter -> dense (embedding) + sparse (BM25) scores
            -> normalise -> weighted fusion -> light rating boost -> top-k
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .bm25 import BM25, tokenize
from .embedder import Embedder, get_embedder
from .query_parser import parse_query


@dataclass
class SearchResult:
    id: int
    title: str
    description: str
    category: str
    brand: str
    price: float
    rating: float
    score: float
    semantic_score: float
    keyword_score: float
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _maxnorm(x: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1] by the best candidate (keeps a lone match at 1.0)."""
    x = np.clip(x, 0, None)
    hi = x.max() if x.size else 0.0
    return x / hi if hi > 1e-12 else np.zeros_like(x)


class ProductSearchEngine:
    def __init__(self, csv_path: str, embedder: Union[str, Embedder] = "auto", alpha: float = 0.65):
        """alpha = weight of the semantic score; (1 - alpha) goes to BM25."""
        self.df = pd.read_csv(csv_path)
        self.alpha = alpha
        self.embedder = get_embedder(embedder) if isinstance(embedder, str) else embedder
        self.corpus = (
            self.df["title"] + ". " + self.df["category"] + ". "
            + self.df["brand"] + ". " + self.df["description"]
        ).tolist()
        self.vectors = self.embedder.fit_transform(self.corpus)
        self.bm25 = BM25(self.corpus)

    @property
    def categories(self) -> List[str]:
        return sorted(self.df["category"].unique())

    def search(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> List[SearchResult]:
        alpha = self.alpha if alpha is None else alpha
        parsed = parse_query(query)
        # explicit arguments win over constraints parsed from the text
        min_price = min_price if min_price is not None else parsed.min_price
        max_price = max_price if max_price is not None else parsed.max_price
        min_rating = min_rating if min_rating is not None else parsed.min_rating

        mask = np.ones(len(self.df), dtype=bool)
        if category:
            mask &= (self.df["category"].str.lower() == category.lower()).to_numpy()
        if min_price is not None:
            mask &= (self.df["price"] >= min_price).to_numpy()
        if max_price is not None:
            mask &= (self.df["price"] <= max_price).to_numpy()
        if min_rating is not None:
            mask &= (self.df["rating"] >= min_rating).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0 or not parsed.text.strip():
            return []

        q_vec = self.embedder.encode([parsed.text])[0]
        dense = self.vectors[idx] @ q_vec
        sparse = self.bm25.scores(parsed.text)[idx]

        # Relevance floor on the *raw* scores, so the normalisation below
        # can't promote noise when nothing truly matches the query.
        keep = (sparse > 0) | (dense >= self.embedder.min_similarity)
        if not keep.any():
            return []
        idx, dense, sparse = idx[keep], dense[keep], sparse[keep]

        fused = alpha * _maxnorm(dense) + (1 - alpha) * _maxnorm(sparse)
        fused = fused + 0.04 * (self.df["rating"].to_numpy()[idx] / 5.0)  # tie-breaker

        q_terms = set(tokenize(parsed.text))
        results = []
        for j in np.argsort(-fused)[:k]:
            row = self.df.iloc[idx[j]]
            results.append(
                SearchResult(
                    id=int(row["id"]), title=row["title"], description=row["description"],
                    category=row["category"], brand=row["brand"], price=float(row["price"]),
                    rating=float(row["rating"]), score=round(float(fused[j]), 4),
                    semantic_score=round(float(dense[j]), 4),
                    keyword_score=round(float(sparse[j]), 4),
                    matched_terms=sorted(q_terms & set(tokenize(self.corpus[idx[j]]))),
                )
            )
        return results
