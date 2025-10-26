
---
title: Multi-Agent Simulator (Gradio)
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
---

# Multi-Agent Simulator (Gradio)

- Uses `OPENAI_API_KEY` from **Settings → Repository secrets** (optional; falls back to rule-based output if not set).
- Personas selectable for Interviewer, Interviewee, and Judge.
- Judge provides a short evaluation and a **formal natural-language decision** to stop/continue (no numeric metrics shown).
