# Adequate Simplicity under Local GRPO

Libor “Burny” Burian

2026-11-17. Living preregistered research report. Artifact regeneration: {{GENERATED_AT}}.

## Abstract

We study whether a local language-model judge can encourage simpler solutions without rewarding missing reasoning. The proposed reward adds a semantic simplicity bonus only when a deterministic verifier accepts the answer and a frozen judge finds the reasoning adequate. Mathematics, introductory physics, and Python functions share a small Qwen3.5 policy. Comparisons include correctness-only reinforcement learning, token and formal-length rewards, shuffled simplicity, and three unchanged-model prompt-only treatments. An independent Gemma 4 judge measures held-out semantic quality. The registered test requires both quality superiority over length baselines and accuracy noninferiority. Current analysis status: {{ANALYSIS_STATUS}}. Pending cells are missing experiments, not zero performance.

## Context, motivation, and introduction

Occam-style economy applies when competing explanations meet the same explanatory obligations. The Stanford Encyclopedia distinguishes ontological parsimony from syntactic elegance [1,2]. An output can be short because it is economical or because it omits the argument. We therefore separate verified correctness, reasoning sufficiency, and semantic simplicity. This operational definition is fallible and context-dependent; it is not a computable universal notion of simplicity.

## Background and related work

DeepSeekMath introduced GRPO with grouped reward estimates rather than a separate learned critic [3]. Dr. GRPO analyzes length and reward-standard-deviation normalization and motivates a constant-denominator objective [4]. ShorterBetter optimizes sample-optimal reasoning length [5]; O1-Pruner and StepPruner study related efficiency objectives [6,7]. LLM judging has documented position, verbosity, and self-preference biases [8]. Our proposed distinction is the controlled adequacy-gated semantic comparison, not the invention of concise reasoning. Prompt-only controls test whether instructions provide a sufficient alternative to training.

## Method

ConciseRL already combines semantic conciseness judging, correctness gating and instructions against answer-only traces [15]. Our claim is limited to evaluating this explicit gate and rubric under the registered controls.

The reward is R=C(1+0.2 A S): binary verified correctness C, a binary adequacy judgment A, and semantic simplicity S in [0,1]. S averages assumption economy, conceptual economy, reasoning economy, and clarity, each scored from zero to four. The judge reads all generated thinking and final text. A correct final answer with essential omissions receives no simplicity bonus. Necessary conditions, units, boundary cases, and useful checks must remain.

The token and formal-length arms use the same adequacy gate and S=k/(k+L). Length normalizers are fixed by calibration after the first Occam run and before length-arm training. Correctness-only uses C. The shuffled control permutes eligible simplicity scores within each question group. Prompt-only controls leave model weights unchanged and use ordinary, concise-but-complete, or full Occam instructions. Decoding, task IDs, and output budgets are held constant.

The implementation reuses TRL, Unsloth, PEFT, Transformers, and local llama.cpp [9,10]. Models run sequentially on the local GPU. BF16 LoRA policy training and four-bit frozen judging reduce resource requirements; measured headroom, full traces, failures, environment versions, and resumable states are recorded. The exact configuration and model revisions reside in published manifests, not an unstated default.

## Data and experimental design

Procedural tasks combine exact references, analytic physics laws and hidden Python function tests. Public sources are GSM8K, MBPP, eligible numerical UGPhysics, and MBPP+ [11-14]. Source licenses remain in force; UGPhysics is evaluation-only and noncommercial/share-alike. Split IDs, normalized question hashes, source revisions and exclusions are frozen before retained training. Family-OOD tasks are distinguished from new parameters of a seen family. Pretraining contamination of public benchmarks remains unknown.

Judge qualification uses constructed dominance and omission examples, including fixed-answer omissions, unnecessary assumptions, redundant reasoning, obfuscation and injection. It is a specification audit, not independent human validation. Failed audits and prompt revisions must be disclosed. Confirmatory evaluation uses a different model family from the training judge.

## Results

### Prerequisite checks

{{READINESS}}

These are constructed specification checks, not policy benchmark results. Under amendment 002, the user authorized exploratory training despite residual training-judge quality failures after mechanical corrections. The failed audit remains visible, and current model/prompt identity, GPU fit and verifier checks remain required. Independent Gemma evaluation still requires qualification. Individual reports retain labels and Wilson intervals; small constructed sets cannot establish general validity.

### Training diagnostics

{{TRAINING}}

These are sampled training traces, not independent benchmark observations. Reward variation within a group is required for a nonzero GRPO task-reward advantage. Completed optimizer steps alone do not demonstrate learning; the saved-run parameter comparison checks whether trainable tensors changed. Missing final-state measurements remain missing. Pilot runs are excluded from retained policy comparisons.

The completed initial pilot produced no verified correct trace and no changed trainable tensors. Its outputs exhausted the completion cap before finishing. Consequently the training judge was never called on an eligible solution: this pilot cannot test the proposed semantic reward. Qwen documents thinking-loop susceptibility in this policy and defaults it to non-thinking mode (https://huggingface.co/Qwen/Qwen3.5-0.8B#thinking-mode). This is a candidate explanation requiring a controlled follow-up, not a proven cause. Full traces and the failed startup preceding this completed attempt are retained in the runtime artifacts.

### Policy comparisons

{{RESULTS}}

No claim should be inferred from an unfilled cell. Complete comparisons, when available, are computed from raw results with paired family-cluster uncertainty, corrected primary tests, domain-specific accuracy intervals, and seed sensitivity. A one-seed milestone is exploratory. Training rollout metrics describe the training distribution only.

## Claims, implications, and discussion

The primary claim under test is higher independently judged C*A*S than each matched length baseline while retaining verified accuracy within the registered margin. Samplewise reward separation does not guarantee aggregate accuracy retention. If the intervals are broad or the effect absent, the registered claim is not established. Shorter outputs alone do not establish greater explanatory power or better generalization.

## Limitations

Small local judges can miss gaps, prefer superficial style, or be exploited. Agreement on constructed examples does not establish reliable judgment on all model outputs. The semantic metric is not independent of its own rubric. Quantization, model-family dependence, procedural simplicity, limited family counts, imperfect code tests, public-data contamination and few training seeds limit interpretation. The selected numerical UGPhysics and JSON-compatible code subsets are not the full benchmarks. GPU sampling may miss transient memory peaks; allocator measurements and failure logs supplement it.

## Conclusion

This report separates a testable adequacy-gated simplicity hypothesis from evidence that has not yet been collected. All positive, negative, and inconclusive experiments remain part of the record. Larger models and programmatic Lean proof-term measurements are preregistered roadmap items, not completed results.

## Appendix: reproducibility and analysis

The repository preregistration specifies the primary metric, falsification criteria, paired bootstrap, multiplicity adjustment, accuracy margin, seeds, and execution order. Every change is pushed before execution. Full checkpoints preserve optimizer, scheduler, random-number and trainer state. One report command regenerates tables, figures, examples, this PDF and the static results explorer from artifacts. Experiment reports contain failures and investigator notes. See reports/preregistration.md, reports/plan.md and docs/usage.md.

## References

[1] Stanford Encyclopedia of Philosophy, William of Ockham. https://plato.stanford.edu/entries/ockham/

[2] Stanford Encyclopedia of Philosophy, Simplicity. https://plato.stanford.edu/entries/simplicity/

[3] Shao et al., DeepSeekMath. https://arxiv.org/abs/2402.03300

[4] Liu et al., Understanding R1-Zero-Like Training. https://arxiv.org/abs/2503.20783

[5] Yi et al., ShorterBetter. https://arxiv.org/abs/2504.21370

[6] O1-Pruner. https://arxiv.org/abs/2501.12570

[7] StepPruner. https://arxiv.org/abs/2510.03805

[8] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685

[9] Hugging Face TRL GRPO documentation. https://huggingface.co/docs/trl/grpo_trainer

[10] Unsloth Qwen3.5 fine-tuning guide. https://unsloth.ai/docs/models/qwen3.5/fine-tune

[11] GSM8K. https://huggingface.co/datasets/openai/gsm8k

[12] MBPP. https://huggingface.co/datasets/google-research-datasets/mbpp

[13] UGPhysics. https://huggingface.co/datasets/UGPhysics/ugphysics

[14] EvalPlus. https://github.com/evalplus/evalplus

[15] Dumitru et al., ConciseRL. https://aclanthology.org/2025.findings-emnlp.927/

## AI statement

Codex wrote this codebase and paper from the exact specification provided by a human. Human cleanup and rewriting are in progress.
