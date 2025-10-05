
import asyncio
import time
import hashlib
import json
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from credrefactor import load_bundle, score_items, CredItem, rule_score

# --- New deps for search + fetch + extraction
import httpx
from duckduckgo_search import DDGS
import trafilatura

# -------------------------------- Config --------------------------------
import os
BUNDLE_PATH = os.getenv("BUNDLE_PATH", "models/bundle.joblib")
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "8.0"))
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "1") == "1"
CACHE_MAX = int(os.getenv("CACHE_MAX", "5000"))
FETCH_TIMEOUT_SEC = float(os.getenv("FETCH_TIMEOUT_SEC", "4.0"))
MAX_FETCH_CONCURRENCY = int(os.getenv("MAX_FETCH_CONCURRENCY", "5"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))  # cap extracted text length

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("credibility_search_api")

# -------------------------------- Model ---------------------------------
try:
    bundle = load_bundle(BUNDLE_PATH)
    log.info(f"Loaded model bundle from {BUNDLE_PATH}")
except Exception as e:
    log.error(f"Failed to load bundle: {e}")
    bundle = None

# -------------------------------- App -----------------------------------
app = FastAPI(title="Credibility Scoring + Search API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------- Schemas --------------------------------
class ScoreRequest(BaseModel):
    id: str
    text: str
    url: Optional[HttpUrl] = None

class BatchRequest(BaseModel):
    items: List[ScoreRequest]

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchScoreResult(BaseModel):
    rank: int
    title: str
    url: HttpUrl
    snippet: str
    score: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None
    errors: List[str] = []

# -------------------------------- Cache ----------------------------------
from functools import lru_cache

@lru_cache(maxsize=CACHE_MAX)
def _cached_single(item_key: str) -> Dict:
    payload = json.loads(item_key)
    item = CredItem(**payload)
    return score_items([item], bundle)

# -------------------------------- Fallback -------------------------------
def fallback_score_item(item: ScoreRequest) -> Dict:
    rules = rule_score(item.text, base=60.0)
    return {
        "results": [{
            "id": item.id,
            "score": float(rules),
            "label": "Low" if rules < 40 else ("Medium" if rules < 70 else "High"),
            "explanation": {
                "blended_raw": rules,
                "blended_calibrated": rules,
                "ml_score": None,
                "rules_score": rules,
                "domain_nudge": 0.0,
                "thresholds": (40.0, 70.0),
                "top_positive": [],
                "top_negative": []
            },
            "model_version": "fallback_rules_v1",
            "errors": ["fallback"]
        }]
    }

# -------------------------------- Health ---------------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": getattr(bundle, "ridge", None).__class__.__name__ if bundle else "none"
    }

# -------------- Middleware: request timing -------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        logging.info(f"{request.method} {request.url.path} {dur_ms}ms {response.status_code if 'response' in locals() else 'ERR'}")

# ------------------------------ Endpoints --------------------------------
@app.post("/score")
async def score_endpoint(payload: ScoreRequest):
    if bundle is None and not ENABLE_FALLBACK:
        raise HTTPException(status_code=503, detail="Model not loaded")
    async def work():
        if bundle is None:
            return fallback_score_item(payload)
        key = json.dumps(payload.dict(), sort_keys=True)
        try:
            return _cached_single(key)
        except Exception as e:
            logging.error(f"score error: {e}")
            if ENABLE_FALLBACK:
                return fallback_score_item(payload)
            raise HTTPException(status_code=400, detail=str(e))
    try:
        return await asyncio.wait_for(work(), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logging.warning("score timeout")
        if ENABLE_FALLBACK:
            return fallback_score_item(payload)
        raise HTTPException(status_code=504, detail="Timeout")

@app.post("/score_batch")
async def score_batch_endpoint(payload: BatchRequest):
    if bundle is None and not ENABLE_FALLBACK:
        raise HTTPException(status_code=503, detail="Model not loaded")
    async def process_item(item: ScoreRequest):
        try:
            return await asyncio.wait_for(score_endpoint(item), timeout=REQUEST_TIMEOUT_SEC)
        except Exception as e:
            logging.error(f"batch item failed: {e}")
            return fallback_score_item(item) if ENABLE_FALLBACK else {"results":[{"id": item.id, "score": None, "label":"Error","errors":[str(e)]}]}
    results = await asyncio.gather(*(process_item(i) for i in payload.items))
    flat = []
    for r in results:
        flat.extend(r.get("results", []))
    return {"results": flat}

# ------------------------- Search Utilities ------------------------------
async def ddg_search(query: str, k: int) -> List[Dict[str, str]]:
    """Use DuckDuckGo to fetch top-k results (title, href, body/snippet)."""
    # DDGS is sync; wrap in thread to avoid blocking event loop
    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=k))
    return await asyncio.to_thread(_search)

async def fetch_url(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """Fetch URL and return raw HTML, with timeout and basic error handling."""
    try:
        r = await client.get(url, timeout=FETCH_TIMEOUT_SEC, follow_redirects=True)
        if r.status_code >= 200 and r.status_code < 300:
            return r.text
    except Exception as e:
        logging.warning(f"fetch failed for {url}: {e}")
    return None

def extract_text(html: Optional[str]) -> Optional[str]:
    """Extract main article text from HTML; returns None on failure."""
    if not html:
        return None
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False)
        return text
    except Exception as e:
        logging.warning(f"extract failed: {e}")
        return None

# --------------------------- Search + Score ------------------------------
@app.post("/search_and_score")
async def search_and_score(payload: SearchRequest):
    """Search the web, fetch pages concurrently, extract text, score each result."""
    if bundle is None and not ENABLE_FALLBACK:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def work():
        # 1) Search
        results = await ddg_search(payload.query, payload.k)
        # Normalize shape: [{'title','href','body'}]
        normalized = [{
            "rank": idx+1,
            "title": r.get("title") or "",
            "url": r.get("href") or r.get("url") or "",
            "snippet": r.get("body") or r.get("snippet") or ""
        } for idx, r in enumerate(results) if (r.get("href") or r.get("url"))]

        # 2) Fetch concurrently with a semaphore
        sem = asyncio.Semaphore(MAX_FETCH_CONCURRENCY)
        async with httpx.AsyncClient(headers={"User-Agent":"CredibilityBot/1.0"}) as client:
            async def fetch_and_extract(u: str) -> str:
                async with sem:
                    html = await fetch_url(client, u)
                    text = extract_text(html)
                    if not text:
                        return ""
                    return text[:MAX_TEXT_CHARS]

            texts = await asyncio.gather(*(fetch_and_extract(r["url"]) for r in normalized))

        # 3) Score (fall back to snippet if no extracted text)
        items = []
        for r, ex in zip(normalized, texts):
            text_for_scoring = ex if ex.strip() else r["snippet"]
            items.append(CredItem(id=str(r["rank"]), text=text_for_scoring or "", url=r["url"]))

        try:
            scored = score_items(items, bundle) if bundle is not None else None
        except Exception as e:
            logging.error(f"scoring error on batch: {e}")
            scored = None

        out = []
        for r, it in zip(normalized, items):
            entry = {
                "rank": r["rank"],
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "score": None,
                "label": None,
                "explanation": None,
                "errors": []
            }
            if scored is not None:
                # match by order; items length equals results length
                sr = scored["results"][r["rank"]-1]
                entry["score"] = sr.get("score")
                entry["label"] = sr.get("label")
                entry["explanation"] = sr.get("explanation")
                entry["errors"] = sr.get("errors") or []
            else:
                if ENABLE_FALLBACK:
                    fb = fallback_score_item(ScoreRequest(id=str(r["rank"]), text=it.text, url=it.url))
                    sr = fb["results"][0]
                    entry["score"] = sr.get("score")
                    entry["label"] = sr.get("label")
                    entry["explanation"] = sr.get("explanation")
                    entry["errors"] = ["fallback"]
            out.append(entry)

        return {"results": out}

    try:
        return await asyncio.wait_for(work(), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        if ENABLE_FALLBACK:
            # Gracefully degrade: return search results with heuristic-only scores on snippets
            results = await ddg_search(payload.query, payload.k)
            normalized = [{
                "rank": idx+1,
                "title": r.get("title") or "",
                "url": r.get("href") or r.get("url") or "",
                "snippet": r.get("body") or r.get("snippet") or ""
            } for idx, r in enumerate(results) if (r.get("href") or r.get("url"))]
            out = []
            for r in normalized:
                fb = fallback_score_item(ScoreRequest(id=str(r["rank"]), text=r["snippet"], url=r["url"]))
                sr = fb["results"][0]
                out.append({
                    "rank": r["rank"],
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["snippet"],
                    "score": sr.get("score"),
                    "label": sr.get("label"),
                    "explanation": sr.get("explanation"),
                    "errors": ["fallback","timeout"]
                })
            return {"results": out}
        raise HTTPException(status_code=504, detail="Timeout")
