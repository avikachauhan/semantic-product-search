"""Interactive command-line search:  python cli.py            (REPL)
                                     python cli.py "query"    (one-shot)"""
import sys
from pathlib import Path

from search_engine import ProductSearchEngine

CSV = Path(__file__).parent / "data" / "products.csv"


def show(results):
    if not results:
        print("  (no matches)")
    for r in results:
        print(f"  {r.score:.3f}  {r.title}  [{r.category}]  ${r.price:,.2f}  ★{r.rating}")


def main():
    engine = ProductSearchEngine(str(CSV))
    print(f"Index ready: {len(engine.df)} products, embedder = {engine.embedder.name}")
    if len(sys.argv) > 1:
        show(engine.search(" ".join(sys.argv[1:]), k=5))
        return
    while True:
        try:
            q = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            show(engine.search(q, k=5))


if __name__ == "__main__":
    main()
