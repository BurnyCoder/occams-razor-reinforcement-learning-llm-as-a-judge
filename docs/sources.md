# Source and license notes

The project implements its own rubric and orchestration and reuses maintained libraries through their public APIs. No unlicensed research repository code is copied. Paper descriptions and equations are cited; downloaded papers remain outside the repository.

| Component | Official reference | Use |
|---|---|---|
| uv | https://docs.astral.sh/uv/ | Environment and lockfile |
| TRL | https://huggingface.co/docs/trl/grpo_trainer | GRPO, Dr. GRPO loss, custom reward interface |
| Unsloth | https://unsloth.ai/docs/models/qwen3.5/fine-tune | Local Qwen3.5 optimization |
| Transformers | https://huggingface.co/docs/transformers/model_doc/qwen3_5 | Qwen3.5 text inference |
| PEFT | https://huggingface.co/docs/peft/index | LoRA and disabled-adapter reference |
| bitsandbytes | https://huggingface.co/docs/transformers/quantization/bitsandbytes | Frozen judge quantization |
| LM Format Enforcer | https://github.com/noamgat/lm-format-enforcer/tree/817f944fcc8851917c40300a47f62d1defc3ffc3 | MIT; imported JSON-schema decoder and Transformers integration |
| llama.cpp | https://github.com/ggml-org/llama.cpp | CUDA Gemma GGUF inference |
| Docker | https://docs.docker.com/engine/containers/run/ | Isolated code verification |
| Hugging Face Hub | https://huggingface.co/docs/huggingface_hub/guides/cli | Artifact publication and revisions |
| GitHub Pages | https://docs.github.com/en/pages | Static result explorer |

Dependencies retain their upstream licenses; the environment lock provides the exact versions. Dataset licenses are listed separately in [data.md](data.md). Gemma and Qwen cards are authoritative for their own model licenses; do not substitute a repository's source-code license for a dataset or model license.
