# Preregistration: adequate simplicity under local GRPO

Author: Libor “Burny” Burian. Date: 2026-11-17. Version: 1.0, before any model results. Git history records amendments; this file must not be silently rewritten after observing outcomes.

## Question and hypotheses

H1: Occam-trained Qwen3.5 achieves higher held-out macro-average `C*A*S` than each matched token-length and formal-length arm. H2: its verified accuracy is noninferior to correctness-only training within 0.02 absolute accuracy, both overall and in each domain. A method is successful only if both pass. Failure of either is negative or inconclusive according to the intervals, not success redefined after the fact.

H3 (secondary): Occam training improves the same metric relative to three unchanged-model prompt-only controls: ordinary, concise-but-complete, and the full Occam rubric. This explicitly tests whether asking for the desired behavior suffices. The policy weights and sampling settings are identical for these controls; only instructions differ. These comparisons are reported without promoting them to H1 after results.

## Arms, units, and estimands

Domains: mathematics, introductory mechanics/electricity, and pure Python functions. One shared multitask policy, equal domain sampling. Unit of pairing is task ID within seed; procedural family is the clustering unit. Public problems are singleton clusters unless a known common source/template provides a broader cluster. The principal estimand is an equal-domain-weight mean across eligible tasks, then equal weight across training seeds.

Arms: Occam, correctness only, token length, formal length, shuffled simplicity, ordinary prompt, concise prompt, Occam prompt. Trained arms share base revision, data order for a given seed, generation parameters, update count, optimizer, LoRA configuration, KL reference, and evaluation. The shuffled control permutes `S` only among correct, adequate responses within the same question group; singleton groups are explicitly logged as unchanged. Correctness-only computes no semantic reward; diagnostic auditing must not affect its update.

`R = C (1 + lambda A S)`, with `lambda = 0.2`, `C in {0,1}`, `A in {0,1}`, and `S in [0,1]`. Occam `S` is the mean of the four rubric dimensions divided by four. Length arms use `S = k/(k+L)` with the **same adequacy judge and gate**. This guarantees samplewise reward separation between wrong and right answers, not aggregate accuracy preservation. Correct but incomplete solutions receive reward 1; wrong solutions receive 0.

Token `L` includes all generated tokens, thinking and final response. Formal `L` counts math lexical tokens in explicitly delimited mathematical expressions plus non-whitespace, non-comment Python characters in code regions throughout the trace. Missing regions are reported; omission does not earn a semantic adequacy pass automatically. AST nodes, nesting, expression structure, and formatting sensitivity are secondary diagnostics, not hidden reward components. These lexical counts are representation-dependent controls, not universal complexity measures.

`k` is the within-domain median positive length on a distinct calibration split from ordinary untrained prompting, using the same generation settings. If none are positive, use 1 and report the failure. Calibration happens **after** first Occam training to honor the execution order, before any length-baseline training; its algorithm is fixed now and it never uses evaluation outcomes. Occam does not depend on `k`.

## Models and resource rules

Policy: Qwen/Qwen3.5-0.8B, BF16 LoRA, thinking enabled. Training judge: Qwen/Qwen3.5-2B, four-bit inference, thinking disabled, deterministic structured output. Held-out judge: official Gemma 4 E4B Q4_0 GGUF, with official E2B Q4_0 fallback only if E4B cannot satisfy the memory criterion. Frozen judge prompts and model revisions are recorded before confirmatory evaluation; no test-set tuning.

All model computation is local GPU computation. Roles serialize GPU use; RAM holds inactive weights. Profile longest allowed generation, backward, and judge inputs; require 750 MiB of free GPU memory throughout measured peaks. No automatic CPU offload, hidden context truncation, or silent model substitution. If both Gemmas fail memory or qualification, report the blockage and revise explicitly before confirmatory experiments.

GRPO uses TRL's Dr. GRPO constant denominator and no group-standard-deviation reward scaling, following [the normalization analysis](https://arxiv.org/abs/2503.20783). Keep KL coefficient 0.02 against the frozen initial policy by disabling adapters. This nonzero KL is a deliberate deviation from that paper. Initial learning rate 1e-5, LoRA rank 16/alpha 32/dropout 0, clipping 0.2, gradient checkpointing, rollout group 4, microbatch 1, accumulation 4. Maximum completion 2048 tokens, prompt limit 1536. Sampling: temperature 0.6, top_p 0.95, top_k 20, min_p 0, repetition penalty 1; identical across policy arms. Report truncation and loops, including incorrect outcomes; do not silently disable thinking.

Seeds: 17, 29, 43. A separate exploratory Occam pilot selects the largest common update count in {64,128,256} projected to fit eight training hours, using measured pilot wall time. Pilot is excluded from confirmatory tests; reset from the same original weights for the first retained run. If no option fits, amend before retained training. Save full optimizer/scheduler/RNG/trainer state every ten updates and at clean interruption boundaries. Resume the same run only with matching data/config/model hashes.

## Data, splits, and leakage

Training draws equal domain counts from procedural tasks plus eligible GSM8K and MBPP training records. Physics training is procedural. No eval problem or solution enters model or judge training. Main evaluation target: 900 tasks, 300/domain, half procedural and half public. Public evaluation: GSM8K test, eligible English numerical text-only mechanics/electricity UGPhysics, and disjoint MBPP+ tests. Eligibility is determined by schema, answer parsability, license, and prompt budget before performance, with exclusion reasons retained. If a stratum lacks the target count, report actual counts and an amendment rather than replace tasks based on scores.

Freeze train/dev/calibration/test/OOD task IDs, generation seeds, family identifiers, public dataset revisions, normalized question hashes, and deduplication checks. In-distribution procedural evaluation uses new parameters; separate OOD evaluation holds out template families. Do not mislabel fresh numbers from a seen family as family-OOD. Public pretraining contamination is unknown and cannot be ruled out by local deduplication.

Development and judge audit examples are separate from test tasks. Constructed judge qualification includes omission pairs with the same correct answer, unnecessary assumptions, redundant paths, code obfuscation, formatting changes, and injection instructions. Before training, the training judge must achieve at least 80% expected strict-dominance agreement and 90% rejection of clearly insufficient reasoning; the held-out judge must qualify before confirmatory evaluation. Report counts and Wilson intervals. No repeated tuning against the same locked audit: a revision creates a fresh audit and records previous failures. These labels are constructed specification checks, not human population preference ground truth.

## Analysis and falsification

Primary quality score is `C*A*S` using the independent Gemma judge, with failures scored zero and exclusions never based on output quality. Evaluate one sampled solution per task/seed under the fixed decoding settings. Preserve every response. Report separate C, A, and each S dimension so a composite cannot conceal failure.

For H1, paired clustered bootstrap resampling within domain over task families, then equal domain weighting, yields confidence intervals for Occam minus each length baseline. Use 10,000 deterministic replicates. Two primary superiority comparisons use Holm correction at familywise alpha .05, with centered bootstrap null tests; report unadjusted and adjusted p-values and paired effect intervals. For H2, the lower endpoint of the paired 95% interval must exceed -0.02 for correctness-only comparison overall **and in every domain**. This is an intersection-union success requirement. If H1 does not exclude zero or H2 cannot exclude a loss exceeding two points, success is not established. An interval entirely below zero for H1 or below -0.02 for H2 supports failure. Otherwise label inconclusive.

Report seed-wise effects and dispersion separately. Three seeds provide weak information about training-seed uncertainty; task bootstrap intervals must not be presented as precise inference across all possible training runs. Include a hierarchical seed-and-task sensitivity bootstrap and the one-seed milestone as explicitly exploratory. Pairing and clustering are preserved; missing arm/task records block paired inference instead of being dropped to favor an arm. Public/procedural and domain-specific effects are secondary. Length-matched and jointly-correct comparisons are descriptive selection-conditioned analyses, not replacements for the primary estimand.

Secondary measures: accuracy, sufficiency, rubric dimensions, token/formal length, AST structure, latency, GPU memory, truncation, invalid judge JSON, verifier errors, zero-variance groups, gradient norm, KL, and reward. Show raw points or counts behind plots and explain intervals. No accuracy-preservation claim based on a nonsignificant difference. The target sample may be underpowered for a two-point margin; report that outcome honestly.

## Execution and stopping

Publish this document and README before results. Publish all code/tests/configuration/lockfile before any test or pipeline execution. Publish every change before rerunning. Publish generated manifests before training. First retained run is Occam seed 17; save and publish it before standalone baseline inference or baseline training. Then execute prompt-only controls/calibration, publish frozen normalizers, then correctness/token/formal training. Repeat Occam first for additional seeds, shuffled control last. Save/publish each model before proceeding.

Initial milestone budget: approximately fourteen hours including useful evaluation; total training window one week. Budget is a resource target, not a positive-result stopping rule. Stop a failing pilot to diagnose numerical errors, no informative groups, broken verification, or unusable judge behavior. Amend, publish, and restart as exploratory work when necessary; never erase failures or treat a tuned pilot as confirmatory. Final reports include negative results, limitations, and departures from this plan.
