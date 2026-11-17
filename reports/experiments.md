# Experiment summary

Generated from run artifacts at 2026-11-17T06:07:25.289+00:00.

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

Accuracy is the fraction verified correct. Adequacy asks whether necessary reasoning is present. Simplicity summarizes four judge dimensions. C × A × S is zero for wrong or inadequate solutions. Training rollout accuracy is not held-out accuracy.

Primary analysis status: `pending`. See results/summary.json for paired intervals, multiplicity adjustments, and seed sensitivity when available.

## Prerequisite audits

| Judge | Dominance | Omission rejection | Invalid JSON | Qualified |
|---|---:|---:|---:|---|
| Qwen3.5-4B (v2) | 0/12 | 5/10 | 18 | No |
| Qwen3.5-4B (v3) | 4/12 | 8/10 | 9 | No |
| Qwen3.5-2B | 1/12 | 7/10 | 16 | No |

GPU/state prerequisite measurements (synthetic inputs; not learning results):

| Check | Recorded result |
|---|---|
| Backward minimum free VRAM (MiB) | 3860 |
| Generation minimum free VRAM (MiB) | 4141 |
| Role-switch weights unchanged | True |
| Role-switch RNG unchanged | True |
| Optimizer serialization round trip | True |
| Profile passed | True |


See [readiness evidence](readiness.md) for data counts and individual audit reports.
