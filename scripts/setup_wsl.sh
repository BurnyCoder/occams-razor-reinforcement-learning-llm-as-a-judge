#!/usr/bin/env bash
# Setup is intentionally separate from model execution; run only after source publication.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/d/AI/occam-rl/uv-cache}"
export HF_HOME="${HF_HOME:-/mnt/d/AI/occam-rl/hf}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
uv venv --python 3.12 .venv
uv sync --frozen --all-extras
if [[ ! -f .env ]]; then cp .env.example .env; fi
docker build -t occam-verifier:1 docker
