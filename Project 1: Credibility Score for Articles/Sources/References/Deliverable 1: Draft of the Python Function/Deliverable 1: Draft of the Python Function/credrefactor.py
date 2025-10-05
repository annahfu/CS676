
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.isotonic import IsotonicRegression
from pydantic import BaseModel, Field, HttpUrl

import joblib  # NEW: save/load

class CredConfig(BaseModel):
    max_features: int = Field(10000, ge=1000)
    min_df: int = Field(2, ge=1)
    ngram_max: int = Field(2, ge=1)
    ridge_alpha_grid: List[float] = [0.5, 1.0, 2.0, 5.0]
    blend_w: float = Field(0.7, ge=0.0, le=1.0)
    base_score: float = Field(60.0, ge=0.0, le=100.0)
    enable_domain_nudge: bool = True
    domain_nudge_scale: float = 8.0
    nudge_max_abs: float = 10.0
    threshold_grid_step: float = 1.0
    random_state: int = 42

POSITIVE_CUES = [
    "peer reviewed", "dataset", "evidence", "statistically significant",
    "replication", "confidence interval", "methodology", "appendix",
    "supplementary", "doi", "pre-registered", "randomized", "controlled",
]
NEGATIVE_CUES = [
    "shocking", "miracle", "cure", "you won't believe", "rumor",
    "unverified", "clickbait", "hoax", "exposed!!!", "secret", "banned",
]

class SimpleSignals(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = [
            "len_chars","exclaim_count","allcaps_count","digit_count",
            "pos_cues","neg_cues","question_count","comma_count","period_count"
        ]
    def fit(self, X, y=None): return self
    def transform(self, X: List[str]):
        rows = []
        for t in X:
            s = t if isinstance(t, str) else ""
            len_chars = float(len(s))
            exclaim = float(s.count("!"))
            allcaps = float(sum(1 for c in s if c.isalpha() and c.isupper()))
            digits = float(sum(c.isdigit() for c in s))
            pos = float(sum(1 for cue in POSITIVE_CUES if cue in s.lower()))
            neg = float(sum(1 for cue in NEGATIVE_CUES if cue in s.lower()))
            qmark = float(s.count("?"))
            comma = float(s.count(","))
            period = float(s.count("."))
            rows.append([len_chars, exclaim, allcaps, digits, pos, neg, qmark, comma, period])
        arr = np.asarray(rows, dtype=float)
        return csr_matrix(arr)

def rule_score(text: str, base: float = 60.0) -> float:
    if not isinstance(text, str) or not text.strip():
        return base
    t = text.strip()
    score = base
    score += 5.0 * sum(1 for cue in POSITIVE_CUES if cue in t.lower())
    score -= 6.0 * sum(1 for cue in NEGATIVE_CUES if cue in t.lower())
    score -= 2.0 * max(0, t.count("!") - 1)
    score -= 1.5 * max(0, t.count("?") - 1)
    caps = sum(1 for c in t if c.isalpha() and c.isupper())
    score -= min(8.0, 0.02 * caps)
    n_words = len(t.split())
    if n_words < 40:
        score -= (40 - n_words) * 0.2
    elif 80 <= n_words <= 300:
        score += 2.0
    return float(np.clip(score, 0.0, 100.0))

_RE_TLD = re.compile(r"https?://([^/]+)/?", re.I)
REPUTABLE_TLDS = (".gov", ".edu")
BLOG_HINTS = ("blogspot.", ".substack.com", "medium.com/@", "wordpress.com", ".blog/")

def domain_nudge(url: Optional[str], scale: float, max_abs: float) -> float:
    if not url or not isinstance(url, str):
        return 0.0
    m = _RE_TLD.match(url.strip())
    if not m:
        return 0.0
    host = m.group(1).lower()
    boost = 0.0
    if any(host.endswith(tld) for tld in REPUTABLE_TLDS):
        boost += scale
    if any(h in host for h in BLOG_HINTS):
        boost -= 0.5 * scale
    hyphens = host.count("-")
    if hyphens >= 3:
        boost -= 0.25 * scale
    return float(np.clip(boost, -abs(max_abs), abs(max_abs)))

from dataclasses import dataclass
@dataclass
class ModelBundle:
    union: FeatureUnion
    ridge: Ridge
    iso: Optional[IsotonicRegression]
    thresholds: Tuple[float, float]
    config: 'CredConfig'

def build_gridsearched_model(texts: List[str], y: np.ndarray, cfg: 'CredConfig') -> Tuple[FeatureUnion, Ridge]:
    union = FeatureUnion([
        ("tfidf", TfidfVectorizer(ngram_range=(1,cfg.ngram_max), max_features=cfg.max_features, min_df=cfg.min_df)),
        ("signals", SimpleSignals())
    ])
    best_ridge = None
    best_union = None
    best_rmse = float("inf")
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    for alpha in [0.5, 1.0, 2.0, 5.0]:
        ridge = Ridge(alpha=alpha, random_state=cfg.random_state)
        rmses = []
        for tr, va in kf.split(texts):
            X_tr = union.fit_transform([texts[i] for i in tr])
            X_va = union.transform([texts[i] for i in va])
            ridge.fit(X_tr, y[tr])
            p = ridge.predict(X_va)
            rmses.append(mean_squared_error(y[va], p, squared=False))
        rmse = float(np.mean(rmses))
        if rmse < best_rmse:
            best_rmse = rmse
            best_ridge = Ridge(alpha=alpha, random_state=cfg.random_state)
            best_union = union

    X_all = best_union.fit_transform(texts)
    best_ridge.fit(X_all, y)
    return best_union, best_ridge

def optimize_thresholds(scores: np.ndarray, y_true: np.ndarray, step: float = 1.0) -> Tuple[float,float]:
    def to_labels(vals, t1, t2):
        labs = np.zeros_like(vals, dtype=int)
        labs[vals > t1] = 1
        labs[vals > t2] = 2
        return labs
    g1, g2 = np.quantile(y_true, [1/3, 2/3])
    y_lab = to_labels(y_true, g1, g2)
    grid = np.arange(0, 100+1e-9, step)
    best, best_f1 = (33.0, 66.0), -1.0
    for t1 in grid:
        for t2 in grid:
            if t2 <= t1: continue
            pred_lab = to_labels(scores, t1, t2)
            f1 = f1_score(y_lab, pred_lab, average="macro")
            if f1 > best_f1:
                best_f1, best = f1, (float(t1), float(t2))
    return best

def fit_isotonic(blended: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True, out_of_bounds="clip")
    iso.fit(blended, y_true)
    return iso

def get_feature_names(union: FeatureUnion) -> List[str]:
    names = []
    tfidf: TfidfVectorizer = union.transformer_list[0][1]
    signals: SimpleSignals = union.transformer_list[1][1]
    names.extend(list(tfidf.get_feature_names_out()))
    names.extend(signals.feature_names_)
    return names

def top_contributors(union: FeatureUnion, ridge: Ridge, text: str, k: int = 8):
    X = union.transform([text])
    coefs = ridge.coef_.ravel()
    contrib = X.toarray().ravel() * coefs
    names = np.array(get_feature_names(union))
    order = np.argsort(contrib)
    top_pos = [(names[i], float(contrib[i])) for i in order[-k:][::-1] if abs(contrib[i]) > 0]
    top_neg = [(names[i], float(contrib[i])) for i in order[:k] if abs(contrib[i]) > 0]
    return {"top_positive": top_pos, "top_negative": top_neg}

class CredItem(BaseModel):
    id: str
    text: str
    url: Optional[HttpUrl] = None

class CredExplanation(BaseModel):
    blended_raw: float
    blended_calibrated: float
    ml_score: float
    rules_score: float
    domain_nudge: float
    thresholds: Tuple[float, float]
    top_positive: List[Tuple[str, float]] = []
    top_negative: List[Tuple[str, float]] = []

class CredResult(BaseModel):
    id: str
    score: float
    label: str
    explanation: CredExplanation
    model_version: str = "credrefactor_v1"
    errors: List[str] = []

@dataclass
class ModelBundle:
    union: FeatureUnion
    ridge: Ridge
    iso: Optional[IsotonicRegression]
    thresholds: Tuple[float, float]
    config: CredConfig

def score_items(items: List[CredItem], bundle: ModelBundle) -> Dict[str, List[Dict]]:
    out = {"results": []}
    for it in items:
        try:
            ml = float(bundle.ridge.predict(bundle.union.transform([it.text]))[0])
            rules = rule_score(it.text, base=bundle.config.base_score)
            dn = domain_nudge(it.url, bundle.config.domain_nudge_scale, bundle.config.nudge_max_abs) if bundle.config.enable_domain_nudge else 0.0
            blended_raw = bundle.config.blend_w * ml + (1.0 - bundle.config.blend_w) * rules
            blended_raw = float(np.clip(blended_raw + dn, 0.0, 100.0))
            calibrated = float(bundle.iso.transform([blended_raw])[0]) if bundle.iso is not None else blended_raw
            t1, t2 = bundle.thresholds
            label = "Low"
            if calibrated > t1: label = "Medium"
            if calibrated > t2: label = "High"
            contribs = top_contributors(bundle.union, bundle.ridge, it.text, k=8)
            res = CredResult(
                id=it.id,
                score=calibrated,
                label=label,
                explanation=CredExplanation(
                    blended_raw=blended_raw,
                    blended_calibrated=calibrated,
                    ml_score=ml,
                    rules_score=rules,
                    domain_nudge=dn,
                    thresholds=bundle.thresholds,
                    top_positive=contribs["top_positive"],
                    top_negative=contribs["top_negative"]
                )
            )
            out["results"].append(json.loads(res.json()))
        except Exception as e:
            out["results"].append({
                "id": it.id,
                "score": None,
                "label": "Error",
                "explanation": {},
                "model_version": "credrefactor_v1",
                "errors": [str(e)],
            })
    return out

def train_bundle(texts: List[str], y: np.ndarray, cfg: CredConfig) -> ModelBundle:
    X_tr, X_va, y_tr, y_va = train_test_split(texts, y, test_size=0.2, random_state=cfg.random_state)
    union, ridge = build_gridsearched_model(X_tr, y_tr, cfg)
    ml_va = ridge.predict(union.transform(X_va))
    rules_va = np.array([rule_score(t, base=cfg.base_score) for t in X_va])
    blended_va = cfg.blend_w * ml_va + (1.0 - cfg.blend_w) * rules_va
    blended_va = np.clip(blended_va, 0.0, 100.0)
    t1, t2 = optimize_thresholds(blended_va, y_va, step=cfg.threshold_grid_step)
    iso = fit_isotonic(blended_va, y_va)
    union, ridge = build_gridsearched_model(texts, y, cfg)
    return ModelBundle(union=union, ridge=ridge, iso=iso, thresholds=(t1, t2), config=cfg)

def save_bundle(bundle: ModelBundle, path: str) -> None:
    joblib.dump({
        "union": bundle.union,
        "ridge": bundle.ridge,
        "iso": bundle.iso,
        "thresholds": bundle.thresholds,
        "config": bundle.config.dict(),
    }, path)

def load_bundle(path: str) -> ModelBundle:
    data = joblib.load(path)
    cfg = CredConfig(**data["config"])
    return ModelBundle(
        union=data["union"],
        ridge=data["ridge"],
        iso=data["iso"],
        thresholds=tuple(data["thresholds"]),
        config=cfg
    )
