
import asyncio
import time
import hashlib
import json
import logging
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from credrefactor import load_bundle, score_items, CredItem, rule_score, CredConfig

import os
BUNDLE_PATH = os.getenv("BUNDLE_PATH", "models/bundle.joblib")
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "4.0"))
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "1") == "1"
CACHE_MAX = int(os.getenv("CACHE_MAX", "5000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("credibility_api")

try:
    bundle = load_bundle(BUNDLE_PATH)
    log.info(f"Loaded model bundle from {BUNDLE_PATH}")
except Exception as e:
    log.error(f"Failed to load bundle: {e}")
    bundle = None

app = FastAPI(title="Credibility Scoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoreRequest(BaseModel):
    id: str
    text: str
    url: Optional[HttpUrl] = None

class BatchRequest(BaseModel):
    items: List[ScoreRequest]

from functools import lru_cache

def _hash_item(d: Dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

@lru_cache(maxsize=CACHE_MAX)
def _cached_single(item_key: str) -> Dict:
    payload = json.loads(item_key)
    item = CredItem(**payload)
    return score_items([item], bundle)

def fallback_score(item: ScoreRequest) -> Dict:
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

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": getattr(bundle, "ridge", None).__class__.__name__ if bundle else "none"
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        log.info(f"{request.method} {request.url.path} {dur_ms}ms {response.status_code if 'response' in locals() else 'ERR'}")

@app.post("/score")
async def score_endpoint(payload: ScoreRequest):
    if bundle is None and not ENABLE_FALLBACK:
        raise HTTPException(status_code=503, detail="Model not loaded")
    async def work():
        if bundle is None:
            return fallback_score(payload) if ENABLE_FALLBACK else HTTPException(status_code=503, detail="Model not loaded")
        key = json.dumps(payload.dict(), sort_keys=True)
        try:
            return _cached_single(key)
        except Exception as e:
            log.error(f"score error: {e}")
            if ENABLE_FALLBACK:
                return fallback_score(payload)
            raise HTTPException(status_code=400, detail=str(e))
    try:
        return await asyncio.wait_for(work(), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        log.warning("score timeout")
        if ENABLE_FALLBACK:
            return fallback_score(payload)
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
            return fallback_score(item) if ENABLE_FALLBACK else {"results":[{"id": item.id, "score": None, "label":"Error","errors":[str(e)]}]}
    results = await asyncio.gather(*(process_item(i) for i in payload.items))
    flat = []
    for r in results:
        flat.extend(r.get("results", []))
    return {"results": flat}
