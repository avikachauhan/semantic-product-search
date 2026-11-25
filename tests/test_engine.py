import unittest
from pathlib import Path

from search_engine import ProductSearchEngine
from search_engine.query_parser import parse_query

CSV = Path(__file__).parent.parent / "data" / "products.csv"


class QueryParserTests(unittest.TestCase):
    def test_price_and_rating(self):
        p = parse_query("wireless headphones under $100 rated 4 stars")
        self.assertEqual(p.max_price, 100)
        self.assertEqual(p.min_rating, 4)
        self.assertEqual(p.text, "wireless headphones")

    def test_between(self):
        p = parse_query("laptop between 500 and 1000")
        self.assertEqual((p.min_price, p.max_price), (500, 1000))

    def test_no_constraints(self):
        self.assertEqual(parse_query("yoga mat").text, "yoga mat")


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ProductSearchEngine(str(CSV), embedder="lsa")

    def titles(self, q, k=3, **kw):
        return [r.title for r in self.engine.search(q, k=k, **kw)]

    def test_keyword_match(self):
        self.assertIn("ZenFlow Yoga Mat", self.titles("yoga mat"))

    def test_semantic_intent(self):
        self.assertIn("FrostKeep Insulated Water Bottle", self.titles("keep drinks cold on a hike", k=3))

    def test_price_filter_from_text(self):
        res = self.engine.search("headphones under $100", k=10)
        self.assertTrue(res and all(r.price <= 100 for r in res))

    def test_category_filter(self):
        res = self.engine.search("fun for kids", k=10, category="Toys")
        self.assertTrue(res and all(r.category == "Toys" for r in res))

    def test_empty_when_filters_exclude_all(self):
        self.assertEqual(self.engine.search("laptop", max_price=5), [])

    def test_scores_sorted(self):
        scores = [r.score for r in self.engine.search("coffee", k=10)]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
