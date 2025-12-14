---
title: Retail Item Classifier Agentic AI
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
---
# Agentic Retail Item Classifier (Agentic-lite)

This project demonstrates how a traditional supervised machine learning
classifier can be extended into an **agentic system** by layering
decision policies, tool usage, and memory on top of the model. Hugging Face site: https://huggingface.co/spaces/annahfu1/Retail_Item_Classifier_Agentic_AI

## Files
- `app.py` — main application and agent policy
- `product_db.csv` — optional local product database (tool)
- `corrections.csv` — persistent self-correction memory

## Configuration
All configurable behavior is defined in the `AgentConfig` class at the top
of `app.py`.

Parameters:
- `product_db_path`: path to local product database CSV
- `corrections_path`: path to user correction memory CSV
- `low_conf`: confidence threshold that triggers a follow-up question
- `very_low_conf`: reserved for stronger uncertainty messaging

## How It Works
At inference time, the system follows this priority order:
1. Apply learned corrections from prior user feedback (memory)
2. Query the local product database if available (tool usage)
3. Fall back to ML classification
4. If confidence is low, request clarification before finalizing

## Why This Is Agentic
This system is agentic because it:
- **Operates over multiple steps** instead of single-pass prediction
- **Uses tools** (product database lookup) to improve decisions
- **Maintains memory** of past interactions and corrections
- **Adapts behavior** based on uncertainty thresholds

The underlying ML model is not autonomous, but the surrounding
decision policy forms an agent loop that decides *what to do next*
to achieve the goal of accurate classification.

## Run
```bash
pip install -U pandas numpy scikit-learn gradio
python app.py
```
