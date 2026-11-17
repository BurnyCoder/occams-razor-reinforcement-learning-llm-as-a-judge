# occam-seed17-pilot

{}

# Occam pilot investigator notes

2026-11-17. This exploratory pilot tests runtime, reward variation, parameter changes and timing; it is excluded from retained policy comparisons.

The initial attempt used the published current Qwen audit under the explicit training-only waiver. It loaded the policy but failed before any rollout or optimizer update because the JSONL reader split a Unicode separator inside the GSM8K question `gsm8k/train/2381`. Frozen dataset hashes still matched. The reader fix follows LF record boundaries and preserves every task byte. See `reports/runtime-integration.md` and the immutable startup log in `results/runtime/occam-pilot-startup.jsonl`.

Retry from the original seed is authorized only before the first model prompt or rollout. The original failure stays in this event log; a subsequent attempt appends its own source identity. No learning or checkpoint-resume success is inferred from startup recovery. Hyperparameters remain those in the preregistered configuration and are not selected from held-out performance.


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