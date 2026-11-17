#!/usr/bin/env bash
# Build only the published llama.cpp commit; caches live on the approved data drive.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
revision=$(uv run --frozen python -c 'import json; print(json.load(open("manifests/revisions.json"))["llama_cpp_commit"])')
target=/mnt/d/AI/occam-rl/llama.cpp
if [[ ! -d "$target/.git" ]]; then git clone https://github.com/ggml-org/llama.cpp.git "$target"; fi
git -C "$target" fetch origin "$revision"
git -C "$target" checkout --detach "$revision"
cmake -S "$target" -B "$target/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
cmake --build "$target/build" --config Release -j 6 --target llama-server
