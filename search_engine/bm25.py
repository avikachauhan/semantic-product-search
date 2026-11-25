"""Minimal Okapi BM25 for the lexical half of hybrid search."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import numpy as np

STOPWORDS = {
    "a", "an", "and", "the", "for", "of", "to", "in", "on", "with", "or", "is",
    "are", "my", "me", "i", "that", "this", "it", "at", "by", "from", "as", "be",
}


def stem(t: str) -> str:
    """Very light plural stemming: toddlers -> toddler, batteries -> battery."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in tokens]
        self.dl = np.array([len(t) for t in tokens], dtype=float)
        self.avgdl = float(self.dl.mean()) if len(docs) else 1.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for w in set(tokenize(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out
