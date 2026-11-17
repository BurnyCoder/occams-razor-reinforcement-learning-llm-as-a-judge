# State of the art and gap

Update before model results: [ConciseRL](https://aclanthology.org/2025.findings-emnlp.927/) already combines judged semantic conciseness with correctness and discourages missing reasoning. Treat this project as a controlled extension and evaluation study; direct replication is additional roadmap work.

2026-11-17. This is a scoped review of efficient reasoning and local implementation, not a leaderboard claiming coverage of every publication.

Existing work establishes several relevant approaches: sampled optimal length ([ShorterBetter](https://arxiv.org/abs/2504.21370)), length harmonization ([O1-Pruner](https://arxiv.org/abs/2501.12570)), step-level pruning ([StepPruner](https://arxiv.org/abs/2510.03805)), and changes to normalization in the RL objective ([Dr. GRPO](https://arxiv.org/abs/2503.20783)). Their training budgets, models, prompts, and metrics differ, so cross-paper score rankings would be misleading.

The proposed experiment asks a narrower unresolved question: **does a qualified local judge's semantic parsimony signal help beyond matched length signals, when both correctness and essential reasoning are required?** No claim of first-ever semantic simplicity optimization is made. The best comparison for causality is retraining all arms from the same pinned model under the same data, decoding, updates, adequacy gate, and evaluation. Prompt-only controls determine whether asking is sufficient.

| Choice | Current official support | Consequence |
|---|---|---|
| Qwen3.5 small dense models | [Model cards](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_5) | Start small and profile, retain larger-model roadmap. |
| Local GRPO / BF16 LoRA | [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [TRL](https://huggingface.co/docs/trl/grpo_trainer) | Reuse supported implementations; avoid assuming four-bit training is interchangeable with BF16. |
| Newest suitable Gemma | [Google release history](https://ai.google.dev/gemma/docs/releases), [Gemma 4 documentation](https://ai.google.dev/gemma/docs/core) | Use Gemma 4 rather than Phi; choose E4B or E2B by measured fit before evaluation. An embedding release is not a generative judge. |
| Structural formal proofs | [Lean expressions](https://lean-lang.org/doc/api/Lean/Expr.html), [proof validation](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) | Proof-term and dependency measurements are future construct-validation work, not an implemented current reward. |

Main risks are weak small-model judges, reward exploitation, zero-variance rollout groups, insufficient statistical power, model-family dependence, public-data contamination, quantization effects, and compute limits. Each requires measurement; none is resolved merely by using Occam's razor as a label. See the frozen [analysis plan](preregistration.md).
