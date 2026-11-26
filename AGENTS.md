# Project working instructions

Follow the user's complete scientific and execution-order instructions supplied in the chat. This file records the operational rules needed by future contributors.

- Think like a scientist: preregister, cite primary sources, retain failures, separate exploratory from confirmatory results, and avoid unsupported claims.
- Use uv, Python 3.12, a project-local .venv, and ignored .env configuration. Work only in this project; D:\AI\occam-rl is approved for large artifacts.
- Write all phase code, tests, configs, lockfile, and README before execution. Commit, push, verify remote commit, then test/run. Push fixes before rerunning. Publish generated-data manifests before training. Do not bypass the publication guard.
- Train and publish Occam seed 17 before standalone baseline inference or training. Baselines include ordinary, concise, and full-rubric prompt-only unchanged-model controls.
- Use only local GPU models: Qwen3.5 policy/judge and newest suitable Gemma 4 independent judge. No hosted inference, implicit CPU model computation, hidden context truncation, or silent model swaps.
- Preserve necessary reasoning. A correct final answer alone does not qualify for the simplicity bonus.
- User amendment: after fixing mechanical judge-output issues, try exploratory training despite residual judge-quality failures. Preserve the failed current audit and identity checks; the training-only override does not waive runtime, verifier, or independent-evaluator checks. Reuse compatible upstream implementations when fixing remaining pipeline issues.
- Use meaningful commits and feature PRs; merge authorized PRs. Never commit secrets or add assistant attribution to commits/README.
- Save resumable checkpoints for long training. Log complete timestamped prompts, outputs, scores, failures, and metrics.
- Regenerate measured numbers and plots from artifacts. Keep docs, reports, README, paper, model cards, and site consistent; no invented results.
- Keep larger models and programmatic Lean proof-structure evaluation on the roadmap.
- Audit-specific user amendment (2026-11-19): the usage reserve below is overridden for this independent audit only. Use existing evidence; no new training or model inference. Preserve original evidence and preregistration. These instructions prescribe behavior, not proof that earlier execution/publication order was satisfied. See [amendment 003](reports/amendment-003-evidence-reporting.md) and the [claim register](reports/claim-register.json). Retained Occam seed 17, not the pilot, must precede baselines; Gemma suitability and qualification remain pending.
- Preserve the user's weekly Codex usage reserve. Read account usage regularly and before substantial new work; stop with a buffer above 10% remaining (begin saving a resume point at 12% remaining). Usage is account-wide and rounded, so do not wait for an exact 10% reading.
- The 2026-11-26 audit uses stored records and CPU execution only. Preserve the frozen preregistration, splits and original experiment logs. Prospective scoring corrections, unresolved physics references and judge construct-validity controls are recorded in [amendment 004](reports/amendment-004-scoring.md). The generated [review](reports/reviews/2026-11-26-scientific-audit.md) gives the status of each registered hypothesis.
