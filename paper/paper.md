# Adequate Simplicity under Local GRPO

Libor “Burny” Burian

2026-11-19. Living research report with a preserved preregistration and dated amendments. Artifact regeneration: 2026-11-19T09:34:03.415+00:00.

## Abstract

We study whether a local language-model judge can encourage simpler solutions without rewarding missing reasoning. The proposed reward adds a semantic simplicity bonus only when a deterministic verifier accepts the answer and a frozen judge finds the reasoning adequate. Mathematics, introductory physics, and Python functions share a small Qwen3.5 policy. Planned comparisons include correctness-only reinforcement learning, token and formal-length rewards, shuffled simplicity, and three unchanged-model prompt-only treatments. Independent Gemma 4 qualification and held-out evaluation remain pending. The completed exploratory pilot yielded no verifier-accepted trace or recorded trainable-tensor change. The registered test requires both quality superiority over length baselines and accuracy noninferiority. Current analysis status: pending. Pending cells are missing experiments, not zero performance.

## Context, motivation, and introduction

Occam-style economy applies when competing explanations meet the same explanatory obligations. The Stanford Encyclopedia distinguishes ontological parsimony from syntactic elegance [1,2]. An output can be short because it is economical or because it omits the argument. We therefore separate verified correctness, reasoning sufficiency, and semantic simplicity. This operational definition is fallible and context-dependent; it is not a computable universal notion of simplicity.

## Background and related work

DeepSeekMath introduced GRPO with grouped reward estimates rather than a separate learned critic [3]. Dr. GRPO analyzes length and reward-standard-deviation normalization and motivates a constant-denominator objective [4]. ShorterBetter optimizes sample-optimal reasoning length [5]; O1-Pruner and StepPruner study related efficiency objectives [6,7]. MT-Bench demonstrates position and verbosity biases in its tested judges; its self-enhancement investigation is inconclusive [8, §3.3]. Our proposed distinction is the controlled adequacy-gated semantic comparison, not the invention of concise reasoning. Prompt-only controls test whether instructions provide a sufficient alternative to training.

## Method

ConciseRL already combines semantic conciseness judging, correctness gating and instructions against answer-only traces [15]. Our claim is limited to evaluating this explicit gate and rubric under the registered controls.

The reward is R=C(1+0.2 A S): binary verified correctness C, a binary adequacy judgment A, and semantic simplicity S in [0,1]. S averages assumption economy, conceptual economy, reasoning economy, and clarity, each scored from zero to four. The judge reads all generated thinking and final text. A correct final answer with essential omissions receives no simplicity bonus. Necessary conditions, units, boundary cases, and useful checks must remain.

The token and formal-length arms use the same adequacy gate and S=k/(k+L). Length normalizers are fixed by calibration after the first Occam run and before length-arm training. Correctness-only uses C. The shuffled control permutes eligible simplicity scores within each question group. Prompt-only controls leave model weights unchanged and use ordinary, concise-but-complete, or full Occam instructions. Decoding, task IDs, and output budgets are held constant.

The implementation reuses TRL, Unsloth, PEFT, Transformers, and local llama.cpp [9,10]. Models run sequentially on the local GPU. BF16 LoRA policy training and four-bit frozen judging reduce resource requirements; measured headroom, full traces, failures, environment versions, and resumable states are recorded. The exact configuration and model revisions reside in published manifests, not an unstated default.

## Data and experimental design

Procedural tasks combine exact references, analytic physics laws and finite Python function tests withheld from policy prompts. Public sources are GSM8K, MBPP, eligible numerical UGPhysics, and MBPP+ [11-14]. Source licenses remain in force; UGPhysics is evaluation-only and noncommercial/share-alike. Split IDs and normalized question hashes are in manifests/data.json; revisions are in manifests/revisions.json; exclusion reasons are in preparation logs. These artifacts remain frozen. Family-OOD tasks are distinguished from new parameters of a seen family. Public pretraining contamination remains unknown. Expected MBPP assertions execute with candidate code inside the container; passing finite tests is not a proof of program correctness or resistance to harness manipulation [16].

Judge qualification uses constructed dominance and omission examples, including fixed-answer omissions, unnecessary assumptions, redundant reasoning, obfuscation and injection. It is a specification audit, not independent human validation. Failed audits and prompt revisions must be disclosed. Confirmatory evaluation uses a different model family from the training judge.

## Results

### Prerequisite checks

| Judge | Dominance | Omission rejection | Invalid judgments | Qualified |
|---|---:|---:|---:|---|
| Qwen3.5-4B (v2) | 0/12 | 5/10 | 18 | No |
| Qwen3.5-4B (v3) | 4/12 | 8/10 | 9 | No |
| Qwen3.5-2B | 1/12 | 7/10 | 16 | No |

GPU/state prerequisite measurements (synthetic inputs; not learning results):

| Check | Recorded result |
|---|---|
| Backward minimum free VRAM (MiB) | 3860 |
| Generation minimum free VRAM (MiB) | 4141 |
| Role-switch trainable tensors unchanged | True |
| Role-switch CUDA RNG unchanged | True |
| Optimizer state-entry count preserved after reload | True |
| Profile passed | True |


These are constructed specification checks, not policy benchmark results. Under amendment 002, the user authorized exploratory training despite residual training-judge quality failures after mechanical corrections. The failed audit remains visible, and current model/prompt identity, GPU fit and verifier checks remain required. Independent Gemma evaluation still requires qualification. Individual reports retain labels and Wilson intervals; small constructed sets cannot establish general validity.

### Training diagnostics

| Run | Updates | Correct / traces | Truncated | Groups with reward spread | Changed tensors | State |
|---|---:|---:|---:|---:|---:|---|
| occam-seed17-pilot | 4 | 0 / 16 | 16 | 0 / 4 | 0 | complete pilot |

occam-seed17-pilot: 16 recorded rollouts across 4 distinct tasks. Repeated rollouts are not independent tasks. Source: https://huggingface.co/datasets/BurnyCoder/occam-rl-artifacts/blob/main/runs/occam-seed17-pilot/events.jsonl

| Verifier rejection reason | Recorded traces |
|---|---:|
| Expected a numeric final answer | 2 |
| Expected exactly one code submission | 4 |
| Expected exactly one final answer | 10 |

These are repeated completions of sampled training tasks, not independent benchmark observations. Reward variation within a group is required for a nonzero centered task-reward advantage; a KL gradient can in principle exist independently. Completed optimizer steps alone do not demonstrate learning. The runtime compares trainable tensors; initial tensors were not separately persisted for an independent reconstruction. Pilot runs are excluded from retained comparisons. Rollout plots index completed updates before sampling, while loss plots index completed updates [16].

The completed initial pilot produced no verifier-accepted trace and recorded no changed trainable tensors. Its outputs exhausted the completion cap. Missing or unparsable submissions caused rejection; this does not establish that every intermediate mathematical statement was false. The training judge was never called on an eligible solution, so this pilot cannot test the semantic reward. Qwen documents thinking-loop susceptibility and defaults this policy to non-thinking mode (https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md). This motivates a controlled follow-up, not a proven causal diagnosis. Full traces and the startup failure remain in runtime artifacts [16].

### Policy comparisons

| Run | Split | N | Accuracy | Adequacy | C × A × S | Tokens | Status |
|---|---|---:|---:|---:|---:|---:|---|
| occam | pending | — | — | — | — | — | not evaluated |
| accuracy | pending | — | — | — | — | — | not evaluated |
| token | pending | — | — | — | — | — | not evaluated |
| formal | pending | — | — | — | — | — | not evaluated |
| shuffled | pending | — | — | — | — | — | not evaluated |
| prompt_ordinary | pending | — | — | — | — | — | not evaluated |
| prompt_concise | pending | — | — | — | — | — | not evaluated |
| prompt_occam | pending | — | — | — | — | — | not evaluated |

No claim should be inferred from an unfilled cell. Complete comparisons, when available, are computed from raw results with paired family-cluster uncertainty, corrected primary tests, domain-specific accuracy intervals, and seed sensitivity. A one-seed milestone is exploratory. Training rollout metrics describe the training distribution only.

## Claims, implications, and discussion

The primary claim under test is higher independently judged C*A*S than each matched length baseline while retaining verified accuracy within the registered margin. Samplewise reward separation does not guarantee aggregate accuracy retention. If the intervals are broad or the effect absent, the registered claim is not established. Shorter outputs alone do not establish greater explanatory power or better generalization.

## Limitations

Small local judges can miss gaps, prefer superficial style, or be exploited. Agreement on constructed examples does not establish reliable judgment on all model outputs. The semantic metric is not independent of its own rubric. Quantization, model-family dependence, procedural simplicity, limited family counts, imperfect code tests, public-data contamination and few training seeds limit interpretation. The selected numerical UGPhysics and JSON-compatible code subsets are not the full benchmarks. GPU sampling may miss transient memory peaks; allocator measurements and failure logs supplement it.

## Conclusion

This report separates an adequacy-gated simplicity hypothesis from evidence that has not yet been collected. Recorded positive, negative, and inconclusive checks remain visible. Larger models and Lean proof-term measurements are roadmap items in reports/plan.md, not completed results or experiments specified in the original preregistration.

## Appendix: reproducibility and analysis

The original preregistration specifies the primary metric, falsification criteria, paired bootstrap, multiplicity adjustment, accuracy margin, seeds, and execution order. Push-before-execution is an enforced current workflow; historical commit dates alone do not prove historical push timing. Checkpoint files contain optimizer, scheduler, random-number and trainer state; successful interrupted-training resumption remains untested. The synthetic profile compares trainable tensors, current CUDA RNG and optimizer state-entry counts after reload, not all state tensors. It does not establish worst-case judge-context fit. One report command regenerates measured tables, figures, examples, this PDF, model-card metrics and explorer data. Amendment 003 requires exact evaluation task membership, domain coverage and consistent recorded provenance before complete/confirmatory labels. Explicit failed judgments retain the preregistered zero score [16]. Wilson intervals are descriptive score intervals on constructed audits; the paired family bootstrap conditions on recorded seeds, with seed resampling only as a sensitivity analysis [17-19].

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

[16] Independent evidence audit, source map, raw-record checks and dated amendment. https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/blob/main/reports/independent-fact-check.md

[17] Wilson (1927), probable inference and the score interval. https://doi.org/10.1080/01621459.1927.10502953

[18] Holm (1979), a simple sequentially rejective multiple test procedure. https://www.jstor.org/stable/4615733

[19] Efron and Tibshirani (1986), bootstrap methods. The project's domain/family/seed hierarchy is its own registered choice. https://doi.org/10.1214/ss/1177013815

## AI statement

Codex wrote this codebase and paper from the exact specification provided by a human. Human cleanup and rewriting are in progress.
