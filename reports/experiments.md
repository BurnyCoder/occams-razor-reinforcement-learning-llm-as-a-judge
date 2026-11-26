# Experiment summary

Generated from run artifacts at 2026-11-26T12:47:27.100+00:00.

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

## Training diagnostics

| Run | Updates | Correct / traces | Truncated | Groups with reward spread | Changed tensors | State |
|---|---:|---:|---:|---:|---:|---|
| occam-seed17-pilot | 4 | 0 / 16 | 16 | 0 / 4 | 0 | complete pilot |

occam-seed17-pilot: 16 recorded rollouts across 4 distinct tasks. Repeated rollouts are not independent tasks. Source: https://huggingface.co/datasets/BurnyCoder/occam-rl-artifacts/blob/main/runs/occam-seed17-pilot/events.jsonl

| Verifier rejection reason | Recorded traces |
|---|---:|
| Expected a numeric final answer | 2 |
| Expected exactly one code submission | 4 |
| Expected exactly one final answer | 10 |


Accuracy is the fraction verified correct. Adequacy asks whether necessary reasoning is present. Simplicity summarizes four judge dimensions. C × A × S is zero for wrong or inadequate solutions. Training rollout accuracy is not held-out accuracy.

Primary analysis status: `pending`. See results/summary.json for paired intervals, multiplicity adjustments, and seed sensitivity when available.

The frozen MBPP+ prompts contain assertions in 150/150 tasks. Examples share scored input values in 149 tasks, including 146 matches with the same argument types. The physics reference review flags 39/150 public targets for adjudication. Choosing the candidate with fewer characters selects the designated preference on 36/36 constructed pairs. Qualification passed in 0/3 completed sets. H1 is untested. H2 is untested. H3 is untested. [Independent scientific audit](reviews/2026-11-26-scientific-audit.md)

## Prerequisite audits

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


See [readiness evidence](readiness.md) for data counts and individual audit reports.
