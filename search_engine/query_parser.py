"""Pull structured constraints (price, rating) out of a natural-language query.

"wireless headphones under $100 rated 4 stars"
  -> text="wireless headphones", max_price=100, min_rating=4
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

NUM = r"\$?\s?(\d+(?:[.,]\d+)?)"


@dataclass
class ParsedQuery:
    text: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_query(query: str) -> ParsedQuery:
    q = query
    p = ParsedQuery(text=query)

    m = re.search(rf"between\s+{NUM}\s+(?:and|to|-)\s+{NUM}", q, re.I)
    if m:
        p.min_price, p.max_price = _num(m.group(1)), _num(m.group(2))
        q = q.replace(m.group(0), " ")

    m = re.search(rf"(?:under|below|less than|cheaper than|up to|max(?:imum)?|within)\s+{NUM}", q, re.I)
    if m:
        p.max_price = _num(m.group(1))
        q = q.replace(m.group(0), " ")

    m = re.search(r"(\d(?:\.\d)?)\s*\+?\s*stars?(?:\s*(?:and|or)\s*(?:up|above|more))?", q, re.I)
    if m:
        p.min_rating = float(m.group(1))
        q = q.replace(m.group(0), " ")
        q = re.sub(r"\brated\b", " ", q, flags=re.I)
    else:
        m = re.search(r"(?:top[- ]rated|highly rated|best rated)", q, re.I)
        if m:
            p.min_rating = 4.5
            q = q.replace(m.group(0), " ")

    m = re.search(rf"(?:over|above|more than|at least|min(?:imum)?)\s+{NUM}", q, re.I)
    if m and p.min_price is None:
        p.min_price = _num(m.group(1))
        q = q.replace(m.group(0), " ")

    q = re.sub(r"\s+", " ", q).strip(" ,.-")
    p.text = q or query
    return p
