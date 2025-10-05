"""
High-level architecture
-----------------------
- Feature extraction uses a FeatureUnion:
  * TF-IDF (1-2 grams) captures token/phrase statistics
  * SimpleSignals adds lightweight lexical features (caps, punctuation, cues)
- Model is a linear Ridge regressor trained on the union features to predict a 0..100 score
- A transparent rule-based score is computed independently (heuristics)
- Final "blended" score = w * ML + (1-w) * rules (+ bounded domain nudge)
- Isotonic regression calibrates blended scores to better match ground truth
- Thresholds (Low/Medium/High) are optimized on validation to maximize macro-F1
- Save/load via joblib; pydantic schemas ensure clean I/O; explainability returns top features
"""

from __future__ import annotations

# ---- Standard library imports
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# ---- Scientific stack
import numpy as np
from scipy.sparse import csr_matrix

# ---- Scikit-learn pieces
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.isotonic import IsotonicRegression

# ---- Validation / serialization
from pydantic import BaseModel, Field, HttpUrl
import joblib  # for saving/loading the trained bundle

# =============================================================================================
# Configuration (CredConfig)
# =============================================================================================
class CredConfig(BaseModel):
    """
    Central, strongly-typed configuration for the model.

    Why we use this:
    - Keeps experiments reproducible (values are captured in the saved artifact)
    - Lets you tune behavior without editing code (you can instantiate with different values)
    - Pydantic validates types/ranges so bad configs fail fast
    """
    # TF-IDF vocabulary size (trade-off: bigger is more expressive but heavier)
    max_features: int = Field(10000, ge=1000)
    # Ignore very rare tokens (min_df = 2 means drop tokens that appear only once overall)
    min_df: int = Field(2, ge=1)
    # Use both unigrams and bigrams (1..2)
    ngram_max: int = Field(2, ge=1)

    # Ridge regularization search grid; we do a simple CV to pick a good alpha
    ridge_alpha_grid: List[float] = [0.5, 1.0, 2.0, 5.0]

    # Blend weight for combining ML and rule scores (w close to 1 favors ML)
    blend_w: float = Field(0.7, ge=0.0, le=1.0)

    # Starting point for rule score before adding/subtracting cues and penalties
    base_score: float = Field(60.0, ge=0.0, le=100.0)

    # Domain nudge controls (bounded for abuse resistance)
    enable_domain_nudge: bool = True
    domain_nudge_scale: float = 8.0
    nudge_max_abs: float = 10.0

    # Step size for scanning thresholds in 0..100 (smaller = finer but slower)
    threshold_grid_step: float = 1.0

    # Seed to keep CV splits and shuffles reproducible
    random_state: int = 42

# =============================================================================================
# Lexical signal extraction (SimpleSignals) and cue lists
# =============================================================================================

# Positive and negative cues.
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
    """
    A tiny, fast transformer that converts raw text into a small numeric feature vector.

    Outputs a SciPy CSR sparse matrix with these columns:
      0: len_chars        -> total characters in text
      1: exclaim_count    -> number of '!' characters (excess suggests sensationalism)
      2: allcaps_count    -> number of uppercase alphabetic characters
      3: digit_count      -> number of digits (proxies for quantitative evidence)
      4: pos_cues         -> count of positive credibility phrases present
      5: neg_cues         -> count of negative / clickbait phrases present
      6: question_count   -> number of '?' (excess suggests uncertainty/clickbait)
      7: comma_count      -> stylistic signal
      8: period_count     -> sentence boundary signal

    These features are intentionally simple (low cost, interpretable) and complement TF-IDF.
    """
    def __init__(self):
        # Expose names for explainability
        self.feature_names_ = [
            "len_chars","exclaim_count","allcaps_count","digit_count",
            "pos_cues","neg_cues","question_count","comma_count","period_count"
        ]

    def fit(self, X, y=None):
        # No training required; return self to be compatible with sklearn pipeline
        return self

    def transform(self, X: List[str]):
        rows = []
        for t in X:
            s = t if isinstance(t, str) else ""
            len_chars = float(len(s))
            exclaim = float(s.count("!"))
            allcaps = float(sum(1 for c in s if c.isalpha() and c.isupper()))
            digits = float(sum(c.isdigit() for c in s))

            # Count how many cue phrases occur in lowercase text (simple substring match)
            pos = float(sum(1 for cue in POSITIVE_CUES if cue in s.lower()))
            neg = float(sum(1 for cue in NEGATIVE_CUES if cue in s.lower()))

            # More punctuation signals
            qmark = float(s.count("?"))
            comma = float(s.count(","))
            period = float(s.count("."))

            rows.append([len_chars, exclaim, allcaps, digits, pos, neg, qmark, comma, period])

        # Convert to a sparse matrix because downstream estimators expect numeric arrays
        arr = np.asarray(rows, dtype=float)
        return csr_matrix(arr)

# =============================================================================================
# Rule-based score (transparent heuristics)
# =============================================================================================
def rule_score(text: str, base: float = 60.0) -> float:
    """
    Compute a human-understandable heuristic credibility score in 0..100.

    Start from a base (optimistic neutrality), then:
    - Add for positive cues (peer reviewed, methodology, etc.)
    - Subtract for negative cues (shocking, miracle, hoax...)
    - Penalize excess punctuation and ALL CAPS
    - Penalize very short texts; mildly reward medium-length texts
    """
    if not isinstance(text, str) or not text.strip():
        return base

    t = text.strip()
    score = base

    # Positive cues add points (bounded by count)
    score += 5.0 * sum(1 for cue in POSITIVE_CUES if cue in t.lower())

    # Negative cues subtract points (slightly stronger than positive)
    score -= 6.0 * sum(1 for cue in NEGATIVE_CUES if cue in t.lower())

    # Excess '!' and '?' beyond the first can signal sensationalism
    score -= 2.0 * max(0, t.count("!") - 1)
    score -= 1.5 * max(0, t.count("?") - 1)

    # ALL CAPS penalty; small but bounded
    caps = sum(1 for c in t if c.isalpha() and c.isupper())
    score -= min(8.0, 0.02 * caps)

    # Length heuristic: very short texts lack context; medium length gets a tiny boost
    n_words = len(t.split())
    if n_words < 40:
        score -= (40 - n_words) * 0.2
    elif 80 <= n_words <= 300:
        score += 2.0

    # Clip to [0,100] to keep everything in the same scale
    return float(np.clip(score, 0.0, 100.0))

# =============================================================================================
# Domain-based nudges (bounded)
# =============================================================================================

# Precompiled regex to grab the host from a URL (http[s]://HOST/...)
_RE_TLD = re.compile(r"https?://([^/]+)/?", re.I)

# Coarse TLD priors: helpful but not decisive
REPUTABLE_TLDS = (".gov", ".edu")
BLOG_HINTS = ("blogspot.", ".substack.com", "medium.com/@", "wordpress.com", ".blog/")

def domain_nudge(url: Optional[str], scale: float, max_abs: float) -> float:
    """
    Provide a small, bounded adjustment based on the URL host.
    - Boost .gov/.edu (often more institutional)
    - Lightly penalize blog-like hosts and very hyphenated subdomains
    - Always clip to [-max_abs, +max_abs] to prevent runaway effects
    """
    if not url or not isinstance(url, str):
        return 0.0

    m = _RE_TLD.match(url.strip())
    if not m:
        return 0.0

    host = m.group(1).lower()
    boost = 0.0

    # Boost institutional TLDs
    if any(host.endswith(tld) for tld in REPUTABLE_TLDS):
        boost += scale

    # Penalize “bloggy” patterns a little
    if any(h in host for h in BLOG_HINTS):
        boost -= 0.5 * scale

    # Very hyphenated subdomains are often low quality
    hyphens = host.count("-")
    if hyphens >= 3:
        boost -= 0.25 * scale

    # Clip to a safe range
    return float(np.clip(boost, -abs(max_abs), abs(max_abs)))

# =============================================================================================
# ModelBundle: trained objects packaged together
# =============================================================================================
@dataclass
class ModelBundle:
    """
    Container for everything needed at inference time:
    - union: the FeatureUnion (TF-IDF + SimpleSignals) with fitted vocabulary
    - ridge: the trained Ridge regressor
    - iso:   isotonic calibration model (can be None if you skip calibration)
    - thresholds: (t1, t2) cut-points for Low/Medium/High
    - config: CredConfig used for this model (for reproducibility)
    """
    union: FeatureUnion
    ridge: Ridge
    iso: Optional[IsotonicRegression]
    thresholds: Tuple[float, float]
    config: CredConfig

# =============================================================================================
# Training helpers: CV over Ridge alpha, isotonic calibration, and thresholds
# =============================================================================================
def build_gridsearched_model(texts: List[str], y: np.ndarray, cfg: CredConfig) -> Tuple[FeatureUnion, Ridge]:
    """
    Build features + pick a good Ridge alpha via 5-fold CV (simple, fast, effective).

    Returns:
      - Fitted FeatureUnion on ALL training data (after picking alpha)
      - Fitted Ridge model on ALL training data
    """
    # Combine sparse TF-IDF features with our small numeric signals
    union = FeatureUnion([
        ("tfidf", TfidfVectorizer(ngram_range=(1, cfg.ngram_max),
                                  max_features=cfg.max_features,
                                  min_df=cfg.min_df)),
        ("signals", SimpleSignals())
    ])

    best_ridge = None
    best_union = None
    best_rmse = float("inf")

    # 5-fold CV to choose alpha (strength of L2 regularization)
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    for alpha in cfg.ridge_alpha_grid:
        ridge = Ridge(alpha=alpha, random_state=cfg.random_state)
        rmses = []

        for tr, va in kf.split(texts):
            # Fit union on the fold's training split to avoid leakage
            X_tr = union.fit_transform([texts[i] for i in tr])
            X_va = union.transform([texts[i] for i in va])

            # Train and evaluate on the fold
            ridge.fit(X_tr, y[tr])
            p = ridge.predict(X_va)
            rmses.append(mean_squared_error(y[va], p, squared=False))

        rmse = float(np.mean(rmses))
        if rmse < best_rmse:
            best_rmse = rmse
            best_ridge = Ridge(alpha=alpha, random_state=cfg.random_state)
            best_union = union  # keep the last fitted union (will refit below)

    # Refit best union on ALL data and then train ridge on full features
    X_all = best_union.fit_transform(texts)
    best_ridge.fit(X_all, y)
    return best_union, best_ridge

def optimize_thresholds(scores: np.ndarray, y_true: np.ndarray, step: float = 1.0) -> Tuple[float, float]:
    """
    Choose two cut-points t1 < t2 in [0,100] that maximize macro-F1 for 3 classes.
    First convert the ground-truth continuous scores into three bins using tertiles.
    """
    def to_labels(vals, t1, t2):
        labs = np.zeros_like(vals, dtype=int)
        labs[vals > t1] = 1
        labs[vals > t2] = 2
        return labs

    # Define ground truth classes by tertiles of y_true
    g1, g2 = np.quantile(y_true, [1/3, 2/3])
    y_lab = to_labels(y_true, g1, g2)

    # Brute-force search over thresholds with a configurable step
    grid = np.arange(0, 100 + 1e-9, step)
    best, best_f1 = (33.0, 66.0), -1.0

    for t1 in grid:
        for t2 in grid:
            if t2 <= t1:
                continue
            pred_lab = to_labels(scores, t1, t2)
            f1 = f1_score(y_lab, pred_lab, average="macro")
            if f1 > best_f1:
                best_f1, best = f1, (float(t1), float(t2))
    return best

def fit_isotonic(blended: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """
    Fit a monotonic calibration function mapping blended_raw -> calibrated score
    Keeps ordering but improves numeric alignment with observed targets.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True, out_of_bounds="clip")
    iso.fit(blended, y_true)
    return iso

# =============================================================================================
# Explainability helpers
# =============================================================================================
def get_feature_names(union: FeatureUnion) -> List[str]:
    """
    Return the combined feature names from TF-IDF and SimpleSignals.
    These names correspond to the columns used by the linear model.
    """
    names = []
    tfidf: TfidfVectorizer = union.transformer_list[0][1]
    signals: SimpleSignals = union.transformer_list[1][1]
    names.extend(list(tfidf.get_feature_names_out()))
    names.extend(signals.feature_names_)
    return names

def top_contributors(union: FeatureUnion, ridge: Ridge, text: str, k: int = 8) -> Dict[str, List[Tuple[str, float]]]:
    """
    For a single text, compute linear contribution = feature_value * coefficient.
    Return the top positive and top negative contributors for transparency.
    """
    X = union.transform([text])          # shape (1, n_features)
    coefs = ridge.coef_.ravel()          # linear weights
    contrib = X.toarray().ravel() * coefs
    names = np.array(get_feature_names(union))

    order = np.argsort(contrib)
    top_pos = [(names[i], float(contrib[i])) for i in order[-k:][::-1] if abs(contrib[i]) > 0]
    top_neg = [(names[i], float(contrib[i])) for i in order[:k] if abs(contrib[i]) > 0]
    return {"top_positive": top_pos, "top_negative": top_neg}

# =============================================================================================
# Pydantic schemas for clean I/O
# =============================================================================================
class CredItem(BaseModel):
    """
    Input item for scoring.
    """
    id: str
    text: str
    url: Optional[HttpUrl] = None

class CredExplanation(BaseModel):
    """
    Rich explanation returned with each result to audit and debug.
    """
    blended_raw: float                 # pre-calibration blended score
    blended_calibrated: float         # post-calibration score (what we label on)
    ml_score: float                   # pure ML prediction (Ridge on features)
    rules_score: float                # pure heuristic score
    domain_nudge: float               # bounded URL-based adjustment
    thresholds: Tuple[float, float]   # (t1, t2) boundaries for L/M/H
    top_positive: List[Tuple[str, float]] = []   # top positive contributors
    top_negative: List[Tuple[str, float]] = []   # top negative contributors

class CredResult(BaseModel):
    """
    Output record for each input. Suitable for JSON serialization.
    """
    id: str
    score: float                      # final calibrated score in 0..100
    label: str                        # "Low" | "Medium" | "High"
    explanation: CredExplanation
    model_version: str = "credrefactor_v1"
    errors: List[str] = []            # non-empty if something went wrong

# =============================================================================================
# Public scoring API
# =============================================================================================
def score_items(items: List[CredItem], bundle: ModelBundle) -> Dict[str, List[Dict]]:
    """
    Main inference function. For each item:
      1) Predict ML score (Ridge over union features)
      2) Compute rule-based score and domain nudge
      3) Blend and calibrate
      4) Map to label using thresholds
      5) Return explainability details
    """
    out = {"results": []}
    for it in items:
        try:
            # 1) ML prediction
            ml = float(bundle.ridge.predict(bundle.union.transform([it.text]))[0])

            # 2) Rule score + (optional) domain nudge
            rules = rule_score(it.text, base=bundle.config.base_score)
            dn = domain_nudge(
                it.url, bundle.config.domain_nudge_scale, bundle.config.nudge_max_abs
            ) if bundle.config.enable_domain_nudge else 0.0

            # 3) Blend + clip; then calibrate with isotonic regression (if present)
            blended_raw = bundle.config.blend_w * ml + (1.0 - bundle.config.blend_w) * rules
            blended_raw = float(np.clip(blended_raw + dn, 0.0, 100.0))
            calibrated = float(bundle.iso.transform([blended_raw])[0]) if bundle.iso is not None else blended_raw

            # 4) Label via optimized thresholds
            t1, t2 = bundle.thresholds
            label = "Low"
            if calibrated > t1: label = "Medium"
            if calibrated > t2: label = "High"

            # 5) Explainability (top contributing features)
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
            # If anything fails for an item, return a structured error payload
            out["results"].append({
                "id": it.id,
                "score": None,
                "label": "Error",
                "explanation": {},
                "model_version": "credrefactor_v1",
                "errors": [str(e)],
            })
    return out

# =============================================================================================
# Training: end-to-end bundle builder (CV + thresholds + calibration)
# =============================================================================================
def train_bundle(texts: List[str], y: np.ndarray, cfg: CredConfig) -> ModelBundle:
    """
    Trains the full pipeline on your labeled data.
      - Splits off validation to pick thresholds & fit calibration
      - Re-fits final model on all data to maximize training signal
    """
    # Hold out a validation slice for choosing thresholds and fitting isotonic calibration
    X_tr, X_va, y_tr, y_va = train_test_split(texts, y, test_size=0.2, random_state=cfg.random_state)

    # Train union + ridge with CV on the TRAIN slice
    union, ridge = build_gridsearched_model(X_tr, y_tr, cfg)

    # Compute blended scores on VALIDATION slice
    ml_va = ridge.predict(union.transform(X_va))
    rules_va = np.array([rule_score(t, base=cfg.base_score) for t in X_va])
    blended_va = cfg.blend_w * ml_va + (1.0 - cfg.blend_w) * rules_va
    blended_va = np.clip(blended_va, 0.0, 100.0)

    # Choose best (t1, t2) by maximizing macro-F1 against validation labels (tertiles)
    t1, t2 = optimize_thresholds(blended_va, y_va, step=cfg.threshold_grid_step)

    # Fit isotonic mapping blended_raw -> calibrated using the same validation slice
    iso = fit_isotonic(blended_va, y_va)

    # Finally, refit the union+ridge on ALL data to get the best possible model
    union, ridge = build_gridsearched_model(texts, y, cfg)

    return ModelBundle(union=union, ridge=ridge, iso=iso, thresholds=(t1, t2), config=cfg)

# =============================================================================================
# Persistence: save/load the bundle for production
# =============================================================================================
def save_bundle(bundle: ModelBundle, path: str) -> None:
    """
    Serialize everything needed for inference:
      - union (with fitted vocabulary)
      - ridge (with learned coefficients)
      - iso (calibration mapping)
      - thresholds (t1, t2)
      - config (so you know how it was trained)
    """
    joblib.dump({
        "union": bundle.union,
        "ridge": bundle.ridge,
        "iso": bundle.iso,
        "thresholds": bundle.thresholds,
        "config": bundle.config.dict(),
    }, path)

def load_bundle(path: str) -> ModelBundle:
    """
    Load a previously saved bundle and reconstruct the ModelBundle object.
    """
    data = joblib.load(path)
    cfg = CredConfig(**data["config"])
    return ModelBundle(
        union=data["union"],
        ridge=data["ridge"],
        iso=data["iso"],
        thresholds=tuple(data["thresholds"]),
        config=cfg
    )
