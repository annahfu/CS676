from __future__ import annotations
import re
# app.py
# Planning-based Agentic Retail Item Classifier (Agentic-lite)
#
# Adds Tier-3 agent loop features:
# - Explicit plan (step list) + executor
# - Pause/resume follow-up question (state machine)
# - Tool use (product_db.csv) + persistent memory (corrections.csv)
# - Execution trace for auditability (JSON in UI)
#
# Run:
#   pip install -U pandas numpy scikit-learn gradio
#   python app.py

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =========================
# Configuration
# =========================
@dataclass
class AgentConfig:
    product_db_path: str = "product_db.csv"
    corrections_path: str = "corrections.csv"
    low_conf: float = 0.60
    very_low_conf: float = 0.45

CFG = AgentConfig()


# =========================
# Core ML Classifier
# =========================
class RetailHierarchyClassifier:
    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.retail_hierarchy = self._build_retail_hierarchy()

    def _build_retail_hierarchy(self) -> Dict[str, List[str]]:
        # Keep small for demo; expand freely.
        return {
            "Produce": ["bananas","apples","spinach","tomatoes","onions","potatoes"],
            "Meat & Seafood": ["chicken breast","ground beef","salmon","shrimp"],
            "Dairy": ["milk","cheese","yogurt","butter"],
            "Bakery": ["bread","bagels","muffins"],
            "Frozen": ["frozen pizza","ice cream","frozen vegetables"],
            "Snacks": ["chips","pretzels","popcorn"],
            "Beverages": ["soda","juice","coffee","energy drink"],
            "Household & Cleaning": ["detergent","paper towels","bleach"],
            "Personal Care": ["shampoo","soap","toothpaste"],
            "Pet Care": ["dog food","cat food"],
            "Health & Wellness": ["vitamin c","pain reliever"]
        }

    def _build_training_data(self) -> Tuple[List[str], List[str]]:
        items, labels = [], []
        for section, kws in self.retail_hierarchy.items():
            for kw in kws:
                items.append(kw)
                labels.append(section)
        return items, labels

    def train(self) -> None:
        items, labels = self._build_training_data()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
        X = self.vectorizer.fit_transform(items)
        y = np.array(labels)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(Xtr, ytr)

        print("Classification report (tiny demo set):")
        print(classification_report(yte, self.model.predict(Xte)))

    def predict(self, item: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not trained; call train() first.")
        X = self.vectorizer.transform([item])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        idx = np.argsort(proba)[::-1]
        return (
            str(classes[idx[0]]),
            float(proba[idx[0]]),
            [(str(classes[i]), float(proba[i])) for i in idx[:3]],
        )


# =========================
# Tools: Product DB + Memory
# =========================
def normalize_item_text(s: str) -> str:
    """Light normalization to improve memory/DB matching."""
    if not s:
        return ""
    s = str(s).lower().strip()
    # remove sizes & pack counts
    s = re.sub(r"\b\d+\s?(oz|ct|count|pk|pack|ml|l|lb|g)\b", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_product_db() -> pd.DataFrame:
    p = Path(CFG.product_db_path)
    if not p.exists():
        return pd.DataFrame(columns=["item", "section", "item_norm"])
    df = pd.read_csv(p)
    if "item" not in df.columns:
        df["item"] = ""
    if "section" not in df.columns:
        df["section"] = ""
    df["item_norm"] = df["item"].astype(str).apply(normalize_item_text)
    return df


PRODUCT_DB = load_product_db()


def db_lookup(item: str) -> Optional[str]:
    if PRODUCT_DB.empty:
        return None
    q = normalize_item_text(item)
    hit = PRODUCT_DB[PRODUCT_DB["item_norm"] == q]
    if hit.empty:
        return None
    return str(hit.iloc[0]["section"]).strip() or None


def load_corrections() -> pd.DataFrame:
    p = Path(CFG.corrections_path)
    if not p.exists():
        return pd.DataFrame(columns=["item", "correct_section", "item_norm"])
    df = pd.read_csv(p)
    if "item" not in df.columns:
        df["item"] = ""
    if "correct_section" not in df.columns:
        df["correct_section"] = ""
    df["item_norm"] = df["item"].astype(str).apply(normalize_item_text)
    return df


CORRECTIONS = load_corrections()


def correction_override(item: str) -> Optional[str]:
    if CORRECTIONS.empty:
        return None
    q = normalize_item_text(item)
    hit = CORRECTIONS[CORRECTIONS["item_norm"] == q]
    if hit.empty:
        return None
    return str(hit.iloc[-1]["correct_section"]).strip() or None


def save_correction(item: str, section: str) -> None:
    """Append a correction to corrections.csv (persistent memory)."""
    global CORRECTIONS
    item = (item or "").strip()
    section = (section or "").strip()
    if not item or not section:
        return
    row = {
        "item": item,
        "correct_section": section,
        "item_norm": normalize_item_text(item),
    }
    CORRECTIONS = pd.concat([CORRECTIONS, pd.DataFrame([row])], ignore_index=True)
    CORRECTIONS.to_csv(CFG.corrections_path, index=False)


# =========================
# Agent (Tier 3): Plan + Executor + Trace
# =========================
class StepType(str, Enum):
    NORMALIZE = "normalize"
    CHECK_MEMORY = "check_memory"
    QUERY_DB = "query_product_db"
    PREDICT_ML = "predict_ml"
    MAYBE_FOLLOWUP = "maybe_followup"
    FINALIZE = "finalize"


@dataclass
class StepResult:
    step: StepType
    ok: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    goal: str
    steps: List[StepType]


@dataclass
class AgentState:
    pending_item: str = ""
    awaiting_followup: bool = False
    last_top3: List[Tuple[str, float]] = field(default_factory=list)
    trace: List[StepResult] = field(default_factory=list)


AGENT_STATE = AgentState()

FOLLOW_UP_Q = "Which bucket best fits this item?"
FOLLOW_UP_CHOICES = ["Food", "Beverage", "Household", "Personal Care", "Pet", "Health/Wellness", "Not sure"]

# Coarse bucket constraints for follow-up resolution
BUCKET_TO_ALLOWED = {
    "Food": ["Produce","Meat & Seafood","Dairy","Bakery","Frozen","Snacks"],
    "Beverage": ["Beverages"],
    "Household": ["Household & Cleaning"],
    "Personal Care": ["Personal Care"],
    "Pet": ["Pet Care"],
    "Health/Wellness": ["Health & Wellness"],
}


def needs_follow_up(confidence: float, item_name: str) -> bool:
    name = (item_name or "").strip()
    if confidence < CFG.low_conf:
        return True
    # very short queries tend to be ambiguous
    if len(name.split()) <= 1 and confidence < 0.75:
        return True
    return False


def build_plan() -> Plan:
    return Plan(
        goal="Classify retail item into store section with high reliability",
        steps=[
            StepType.NORMALIZE,
            StepType.CHECK_MEMORY,
            StepType.QUERY_DB,
            StepType.PREDICT_ML,
            StepType.MAYBE_FOLLOWUP,
            StepType.FINALIZE,
        ],
    )


def execute_plan(item_name: str, clf: RetailHierarchyClassifier, state: AgentState) -> Dict[str, Any]:
    """Runs the plan until it finalizes or needs a follow-up."""
    plan = build_plan()
    state.trace = []

    ctx: Dict[str, Any] = {
        "raw": item_name,
        "normalized": None,
        "category": None,
        "confidence": None,
        "source": None,  # memory | db | ml | followup
        "top3": [],
    }

    try:
        for step in plan.steps:
            if step == StepType.NORMALIZE:
                ctx["normalized"] = normalize_item_text(ctx["raw"])
                state.trace.append(StepResult(step, True, {"normalized": ctx["normalized"]}))

            elif step == StepType.CHECK_MEMORY:
                ov = correction_override(ctx["normalized"]) or correction_override(ctx["raw"])
                state.trace.append(StepResult(step, True, {"override": ov}))
                if ov:
                    ctx["category"], ctx["confidence"], ctx["source"] = ov, 0.99, "memory"
                    break

            elif step == StepType.QUERY_DB:
                db_cat = db_lookup(ctx["normalized"]) or db_lookup(ctx["raw"])
                state.trace.append(StepResult(step, True, {"db_match": db_cat}))
                if db_cat:
                    ctx["category"], ctx["confidence"], ctx["source"] = db_cat, 0.95, "db"
                    break

            elif step == StepType.PREDICT_ML:
                cat, conf, top3 = clf.predict(ctx["raw"])
                ctx["category"], ctx["confidence"], ctx["source"] = cat, float(conf), "ml"
                ctx["top3"] = top3
                state.last_top3 = top3
                state.trace.append(StepResult(step, True, {"ml_cat": cat, "conf": conf, "top3": top3}))

            elif step == StepType.MAYBE_FOLLOWUP:
                conf = float(ctx["confidence"] or 0.0)
                ask = needs_follow_up(conf, ctx["raw"])
                state.trace.append(StepResult(step, True, {"ask_followup": ask, "conf": conf}))
                if ask:
                    state.pending_item = ctx["raw"]
                    state.awaiting_followup = True
                    return {
                        "status": "need_followup",
                        "category": ctx["category"],
                        "confidence": conf,
                        "message": "Low confidence — needs clarification",
                        "trace": [sr.__dict__ for sr in state.trace],
                    }

            elif step == StepType.FINALIZE:
                state.trace.append(StepResult(step, True, {"final": True}))

        state.awaiting_followup = False
        return {
            "status": "final",
            "category": ctx["category"] or "Unknown",
            "confidence": float(ctx["confidence"] or 0.0),
            "message": f"Finalized via {ctx['source'] or 'unknown'}",
            "trace": [sr.__dict__ for sr in state.trace],
        }

    except Exception as e:
        state.trace.append(StepResult(StepType.FINALIZE, False, {"error": str(e)}))
        return {
            "status": "error",
            "category": "Error",
            "confidence": 0.0,
            "message": str(e),
            "trace": [sr.__dict__ for sr in state.trace],
        }


def resume_with_followup(answer: str, clf: RetailHierarchyClassifier, state: AgentState) -> Dict[str, Any]:
    """Resumes the agent loop after the user answers the follow-up question."""
    if not state.awaiting_followup or not state.pending_item:
        return {"status": "error", "category": "Error", "confidence": 0.0, "message": "No pending follow-up.", "trace": []}

    item = state.pending_item
    bucket = (answer or "").strip()

    # Re-run ML, then constrain decision by user's bucket using top-3
    cat, conf, top3 = clf.predict(item)
    chosen = cat
    chosen_conf = float(conf)

    allowed = BUCKET_TO_ALLOWED.get(bucket, [])
    if allowed:
        for sec, p in top3:
            if sec in allowed:
                chosen = sec
                chosen_conf = min(0.95, float(p) + 0.20)
                break

    state.trace.append(StepResult(StepType.MAYBE_FOLLOWUP, True, {"followup_answer": bucket, "chosen": chosen}))
    state.trace.append(StepResult(StepType.FINALIZE, True, {"final": True, "source": "followup"}))

    state.awaiting_followup = False
    state.pending_item = ""

    return {
        "status": "final",
        "category": chosen,
        "confidence": chosen_conf,
        "message": "Finalized via follow-up",
        "trace": [sr.__dict__ for sr in state.trace],
    }


# =========================
# App (Gradio)
# =========================
clf = RetailHierarchyClassifier()
clf.train()

SECTIONS = list(clf.retail_hierarchy.keys())

with gr.Blocks() as demo:
    gr.Markdown("# Agentic Retail Item Classifier (Planning-based)")

    item_in = gr.Textbox(label="Item name", placeholder="e.g., Celsius energy drink 12 pack")
    classify_btn = gr.Button("Classify", variant="primary")

    out_cat = gr.Textbox(label="Result")
    out_conf = gr.Textbox(label="Confidence / Status")
    out_trace = gr.JSON(label="Agent Trace (Plan Execution)", visible=False)

    followup_md = gr.Markdown(visible=False)
    followup_choice = gr.Dropdown(choices=FOLLOW_UP_CHOICES, label="Follow-up Answer", visible=False)
    resolve_btn = gr.Button("Resolve", visible=False)

    gr.Markdown("### Feedback (self-correction memory)")
    correct_section = gr.Dropdown(choices=SECTIONS, label="If wrong, pick correct section")
    save_btn = gr.Button("Save correction")
    save_msg = gr.Textbox(label="Feedback status", interactive=False)

    def ui_classify(item_name: str):
        if not item_name or not str(item_name).strip():
            return (
                "Please enter an item name",
                "—",
                [],
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        res = execute_plan(str(item_name), clf, AGENT_STATE)
        show_followup = res["status"] == "need_followup"

        msg = f"{res['message']} ({res['confidence']:.2f})"
        if res["status"] == "need_followup":
            msg = f"{res['message']} ({res['confidence']:.2f}) — {FOLLOW_UP_Q}"

        return (
            res["category"],
            msg,
            res["trace"],
            gr.update(value=f"**Follow-up:** {FOLLOW_UP_Q}", visible=show_followup),
            gr.update(visible=show_followup, value=None),
            gr.update(visible=show_followup),
        )

    def ui_resolve(answer: str):
        res = resume_with_followup(answer, clf, AGENT_STATE)
        return (
            res["category"],
            f"{res['message']} ({res['confidence']:.2f})",
            res["trace"],
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def ui_save(item_name: str, correct: str):
        if not item_name or not str(item_name).strip():
            return "Type an item name first."
        if not correct or not str(correct).strip():
            return "Pick a correct section first."
        save_correction(str(item_name), str(correct))
        return f"Saved: '{item_name}' -> {correct}. Future queries will use this correction."

    classify_btn.click(
        fn=ui_classify,
        inputs=item_in,
        outputs=[out_cat, out_conf, out_trace, followup_md, followup_choice, resolve_btn],
    )
    resolve_btn.click(
        fn=ui_resolve,
        inputs=followup_choice,
        outputs=[out_cat, out_conf, out_trace, followup_md, followup_choice, resolve_btn],
    )
    save_btn.click(fn=ui_save, inputs=[item_in, correct_section], outputs=save_msg)

demo.launch()
