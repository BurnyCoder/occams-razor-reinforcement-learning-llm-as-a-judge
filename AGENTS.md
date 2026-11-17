# Project working instructions

Follow the user's complete scientific and execution-order instructions supplied in the chat. This file records the operational rules needed by future contributors.

- Think like a scientist: preregister, cite primary sources, retain failures, separate exploratory from confirmatory results, and avoid unsupported claims.
- Use uv, Python 3.12, a project-local .venv, and ignored .env configuration. Work only in this project; D:\AI\occam-rl is approved for large artifacts.
- Write all phase code, tests, configs, lockfile, and README before execution. Commit, push, verify remote commit, then test/run. Push fixes before rerunning. Publish generated-data manifests before training. Do not bypass the publication guard.
- Train and publish Occam seed 17 before standalone baseline inference or training. Baselines include ordinary, concise, and full-rubric prompt-only unchanged-model controls.
- Use only local GPU models: Qwen3.5 policy/judge and newest suitable Gemma 4 independent judge. No hosted inference, implicit CPU model computation, hidden context truncation, or silent model swaps.
- Preserve necessary reasoning. A correct final answer alone does not qualify for the simplicity bonus.
- Use meaningful commits and feature PRs; merge authorized PRs. Never commit secrets or add assistant attribution to commits/README.
- Save resumable checkpoints for long training. Log complete timestamped prompts, outputs, scores, failures, and metrics.
- Regenerate measured numbers and plots from artifacts. Keep docs, reports, README, paper, model cards, and site consistent; no invented results.
- Keep larger models and programmatic Lean proof-structure evaluation on the roadmap.
