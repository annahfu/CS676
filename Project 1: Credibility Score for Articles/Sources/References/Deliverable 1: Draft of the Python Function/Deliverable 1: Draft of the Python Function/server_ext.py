# server_ext.py — FastAPI app with robust search+score and chat UI
import asyncio
import json
import logging
import os
import re
import html as htmllib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

# Core scoring logic (your code shipped as credrefactor_commented.py -> copied to credrefactor.py in Dockerfile)
from credrefactor import load_bundle, score_items, CredItem, rule_score

# Web search + fetch + extraction
import httpx
from duckduckgo_search import DDGS
import trafilatura

# ------------------------------- Config -----------------------------------
BUNDLE_PATH = os.getenv("BUNDLE_PATH", "models/bundle.joblib")
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "8.0"))
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "1") == "1"
CACHE_MAX = int(os.getenv("CACHE_MAX", "5000"))

FETCH_TIMEOUT_SEC = float(os.getenv("FETCH_TIMEOUT_SEC", "4.0"))
MAX_FETCH_CONCURRENCY = int(os.getenv("MAX_FETCH_CONCURRENCY", "5"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))  # cap extracted length

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("credibility_search_api")

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")


# ------------------------------ Load model --------------------------------
try:
    bundle = load_bundle(BUNDLE_PATH)
    log.info(f"Loaded model bundle from {BUNDLE_PATH}")
except Exception as e:
    log.warning(f"Model bundle not loaded ({e}); will use fallback if enabled.")
    bundle = None

# --------------------------------- App ------------------------------------
app = FastAPI(title="Credibility Scoring + Search API", version="1.2.0")

# CORS (safe defaults for demos; tighten if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static chat UI at /web/* and mount landing page at /
app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/", include_in_schema=False)
def home_chat():
    html_path = Path("web/chat.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    # Fallback: link to docs if chat page is missing
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/docs"><a href="/docs">Open API docs</a>'
    )

# ------------------------------ Schemas -----------------------------------
class ScoreRequest(BaseModel):
    id: str
    text: str
    url: Optional[HttpUrl] = None

class BatchRequest(BaseModel):
    items: List[ScoreRequest]

class SearchRequest(BaseModel):
    query: str
    k: int = 5

# ------------------------------ Caching -----------------------------------
@lru_cache(maxsize=CACHE_MAX)
def _cached_single(item_key: str) -> Dict:
    payload = json.loads(item_key)
    item = CredItem(**payload)
    return score_items([item], bundle)

# ----------------------------- Fallback -----------------------------------
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

# ------------------------------ Middleware --------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        code = response.status_code if "response" in locals() else "ERR"
        log.info(f"{request.method} {request.url.path} {dur_ms}ms {code}")

# ------------------------------ Health ------------------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": getattr(bundle, "ridge", None).__class__.__name__ if bundle else "none"
    }

# ----------------------------- Endpoints ----------------------------------
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
            log.error(f"score error: {e}")
            if ENABLE_FALLBACK:
                return fallback_score_item(payload)
            raise HTTPException(status_code=400, detail=str(e))
    try:
        return await asyncio.wait_for(work(), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        log.warning("score timeout")
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
            log.error(f"batch item failed: {e}")
            return fallback_score_item(item) if ENABLE_FALLBACK else {
                "results":[{"id": item.id, "score": None, "label":"Error","errors":[str(e)]}]
            }
    results = await asyncio.gather(*(process_item(i) for i in payload.items))
    flat = []
    for r in results:
        flat.extend(r.get("results", []))
    return {"results": flat}

# ------------------------- Robust Search Helpers --------------------------

# ----------------------- Brave + DDG unified search -----------------------
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

async def brave_search(query: str, k: int) -> List[Dict[str, str]]:
    """
    Query Brave Search API. Returns normalized list of {title, href, body}.
    Requires BRAVE_API_KEY env var.
    """
    if not BRAVE_API_KEY:
        return []
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
        "User-Agent": "CredibilityBot/1.0"
    }
    # Brave supports 'count' up to 20 per request
    params = {
        "q": query,
        "count": min(max(k, 5), 20),     # ask for a few extra to ensure we have k after filtering
        "safesearch": "moderate",
        "country": "us",
        "search_lang": "en"
    }
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            r = await client.get(BRAVE_ENDPOINT, params=params, timeout=FETCH_TIMEOUT_SEC)
            if r.status_code == 429:
                # rate-limited: return empty to allow fallback
                return []
            r.raise_for_status()
            data = r.json() or {}
            items = (data.get("web", {}) or {}).get("results", []) or []
            out: List[Dict[str, str]] = []
            for it in items:
                href = it.get("url") or it.get("link")
                if not href:
                    continue
                out.append({
                    "title": it.get("title") or "",
                    "href": href,
                    "body": it.get("description") or it.get("snippet") or ""
                })
                if len(out) >= k:
                    break
            return out
    except Exception as e:
        log.warning(f"Brave search failed: {e}")
        return []

# Keep your robust DDG search (text → news → HTML/lite fallback)

# (Use the DDG function you currently have working)

async def search_web(query: str, k: int) -> List[Dict[str, str]]:
    """
    Prefer Brave (if API key is present). Fall back to DDG strategies otherwise.
    """
    # 1) Brave (if configured)
    if BRAVE_API_KEY:
        res = await brave_search(query, k)
        if res:
            return res
        log.info("Brave returned no items; falling back to DDG.")

    # 2) Fallback to DDG
    return await ddg_search(query, k)

import re, html as htmllib

def _normalize_results(items, k: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for it in items or []:
        href = it.get("href") or it.get("url")
        if not href:
            continue
        out.append({
            "title": it.get("title") or "",
            "href": href,
            "body": it.get("body") or it.get("snippet") or ""
        })
        if len(out) >= k:
            break
    return out

async def ddg_search(query: str, k: int) -> List[Dict[str, str]]:
    """
    Multi-strategy DuckDuckGo search with reliable fallbacks:
      A) ddgs.text
      B) ddgs.news
      C) HTML fallback (/html then /lite) with redirect-follow and simple parsing
    Returns: list of {title, href, body}
    """
    # A) ddgs.text
    try:
        def _text():
            with DDGS() as ddgs:
                # region & safesearch reduce empty returns in some locales
                return list(ddgs.text(query, max_results=max(k, 8), region="wt-wt", safesearch="moderate"))
        items = await asyncio.to_thread(_text)
        norm = _normalize_results(items, k)
        if norm:
            return norm
    except Exception as e:
        log.warning(f"DDG text search failed: {e}")

    # B) ddgs.news
    try:
        def _news():
            with DDGS() as ddgs:
                return list(ddgs.news(query, max_results=max(k, 8), region="wt-wt", safesearch="moderate"))
        items = await asyncio.to_thread(_news)
        norm = _normalize_results(items, k)
        if norm:
            return norm
    except Exception as e:
        log.warning(f"DDG news search failed: {e}")

    # C) HTML fallback (best effort): try /html then /lite (which is even lighter)
    async def _html_try(url: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        headers = {"User-Agent": "Mozilla/5.0 (CredibilityBot)"}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            r = await client.get(url, timeout=FETCH_TIMEOUT_SEC)
            if r.status_code != 200:
                return out

            # Try to parse <a class="result__a" href="...">Title</a>
            links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                               r.text, flags=re.I | re.S)
            # Snippets sometimes appear as <a class="result__snippet">...</a> or <div class="result__snippet">...</div>
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</',
                                  r.text, flags=re.I | re.S)

            if not links:
                # Alternative markup on /lite: <a class="result-link">Title</a>
                links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                                   r.text, flags=re.I | re.S)

            for i, (href, title_html) in enumerate(links[:k]):
                title = htmllib.unescape(re.sub("<.*?>", "", title_html)).strip()
                snippet = ""
                if i < len(snippets):
                    snippet = htmllib.unescape(re.sub("<.*?>", "", snippets[i])).strip()
                out.append({"title": title, "href": href, "body": snippet})
        return out

    # Try /html
    try:
        qp = httpx.QueryParams({"q": query})
        url_html = f"https://duckduckgo.com/html/?{qp}"
        res = await _html_try(url_html)
        if res:
            return res
    except Exception as e:
        log.warning(f"DDG HTML (/html) fallback failed: {e}")

    # Try /lite
    try:
        qp = httpx.QueryParams({"q": query})
        url_lite = f"https://duckduckgo.com/lite/?{qp}"
        res = await _html_try(url_lite)
        if res:
            return res
    except Exception as e:
        log.warning(f"DDG HTML (/lite) fallback failed: {e}")

    # No luck
    return []

# -------------------------- Fetch & Extraction ----------------------------
async def fetch_url(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, timeout=FETCH_TIMEOUT_SEC, follow_redirects=True)
        if 200 <= r.status_code < 300:
            return r.text
    except Exception as e:
        log.debug(f"fetch failed for {url}: {e}")
    return None

def extract_text(html: Optional[str]) -> Optional[str]:
    if not html:
        return None
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )
        return text
    except Exception as e:
        log.debug(f"extract failed: {e}")
        return None

# --------------------------- Search + Score --------------------------------
@app.post("/search_and_score")
async def search_and_score(payload: SearchRequest):
    if bundle is None and not ENABLE_FALLBACK:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def work():
        # 1) Search with robust fallbacks
        results = await search_web(payload.query, payload.k)

        normalized = [{
            "rank": idx + 1,
            "title": r.get("title") or "",
            "url": r.get("href") or r.get("url") or "",
            "snippet": r.get("body") or r.get("snippet") or ""
        } for idx, r in enumerate(results) if (r.get("href") or r.get("url"))]

        # No results: return empty list (UI will show 'No results')
        if not normalized:
            return {"results": []}

        # 2) Fetch & extract concurrently (best effort)
        sem = asyncio.Semaphore(MAX_FETCH_CONCURRENCY)
        async with httpx.AsyncClient(headers={"User-Agent":"CredibilityBot/1.0"}) as client:
            async def fetch_and_extract(u: str) -> str:
                async with sem:
                    html_doc = await fetch_url(client, u)
                    text = extract_text(html_doc)
                    if not text:
                        return ""
                    return text[:MAX_TEXT_CHARS]

            texts = await asyncio.gather(*(fetch_and_extract(r["url"]) for r in normalized))

        # 3) Build items for scoring (fallback to snippet when extraction empty)
        items = []
        for r, ex in zip(normalized, texts):
            text_for_scoring = ex if ex.strip() else r["snippet"]
            items.append(CredItem(id=str(r["rank"]), text=text_for_scoring or "", url=r["url"]))

        # 4) Score (fallback to rules if needed)
        try:
            scored = score_items(items, bundle) if bundle is not None else None
        except Exception as e:
            log.error(f"scoring error on batch: {e}")
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
                sr = scored["results"][r["rank"] - 1]  # same order
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
            # Return search results (if any) with heuristic-only scores from snippets
            results = await ddg_search(payload.query, payload.k)
            normalized = [{
                "rank": idx + 1,
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
                    "errors": ["fallback", "timeout"]
                })
            return {"results": out}
        raise HTTPException(status_code=504, detail="Timeout")
