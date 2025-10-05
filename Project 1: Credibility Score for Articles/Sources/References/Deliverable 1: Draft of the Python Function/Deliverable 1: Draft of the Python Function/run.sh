#!/usr/bin/env bash
set -euo pipefail
uvicorn server:app --host 0.0.0.0 --port 7860
