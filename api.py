"""REST API + web UI.   uvicorn api:app --reload   ->  http://localhost:8000"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from search_engine import ProductSearchEngine

ROOT = Path(__file__).parent
engine = ProductSearchEngine(str(ROOT / "data" / "products.csv"))
app = FastAPI(title="Smart Product Search", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "products": len(engine.df), "embedder": engine.embedder.name}


@app.get("/categories")
def categories():
    return engine.categories


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=50),
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    alpha: Optional[float] = Query(None, ge=0, le=1),
):
    results = engine.search(q, k, category, min_price, max_price, min_rating, alpha)
    return {"query": q, "count": len(results), "results": [r.to_dict() for r in results]}
