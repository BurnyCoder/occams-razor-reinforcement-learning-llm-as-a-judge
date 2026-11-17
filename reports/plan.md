# Implementation and experiment plan

Updated 2026-11-17. Preserve the preregistration; record deviations separately. The full original scope remains active.

## Phase order

1. Research and fact-check philosophical framing, efficient reasoning RL, judging, datasets, model support, licenses, and research methods. Write literature review, state-of-the-art review, rubric, data docs, preregistration, README. Publish before results.
2. Implement modular pipeline, tests first, complete configs and dependency lock, publication guards, task generation/verifiers, full trace logs, local judges, GRPO, checkpoints, evaluation/statistics, reporting/PDF, static explorer and live dashboard. Push **all existing publishable work before the first test**. Every fix is pushed before execution.
3. Run CPU correctness tests and sandbox tests. Freeze dataset/model revisions and generated-data manifests; publish. Qualify the training judge, measure GPU peaks and save resumability evidence. Qualify Gemma independently before held-out evaluation. Publish failures and amendments when needed.
4. Occam-only exploratory pilot; select common budget from the preregistered rule. Publish configuration. Train retained Occam seed 17 first, resume safely, save adapter and full checkpoint, generate/update reports and upload GitHub/Hugging Face artifacts.
5. Only now run unchanged-model ordinary, concise, and full-Occam-rubric **prompt-only baselines**. Freeze calibration normalizers and publish them before training the length arms. Evaluate using the identical held-out tasks and decoding budget.
6. Train correctness-only, token-length, and formal-length seed 17, publishing each result. Run paired evaluation and report the one-seed milestone as exploratory.
7. Repeat matched runs for seeds 29 and 43, Occam first within each seed. Run shuffled-simplicity control. If budget permits, prespecified ablations and sensitivity sweeps receive distinct exploratory labels and amendments before execution.
8. Generate all measured tables, figures, examples, reports, paper PDF, Hugging Face cards, and GitHub Pages data from raw artifacts in one command. Publish and check the site. Audit methodology once for substantial errors; fix and disclose errors without endless review loops.

## Implementation boundaries

- Use current checkout and project-local venv; D:\AI\occam-rl for caches and checkpoints. No unrelated project inspection.
- All models local on the GPU. Serialize roles, profile actual memory, enforce headroom, and never conceal CPU offload or context loss.
- Data and model revisions, environment, Git commits, full raw prompts/outputs, timestamps, judge errors, training metrics, and resumable states are artifacts.
- Do not execute generated code on the host: use constrained networkless containers with time/memory/process limits.
- Publish only nonsecret files. Tokens and .env remain ignored. Original dataset licenses remain in force.
- Papers credit Libor “Burny” Burian. The requested AI authorship statement appears in the paper only.

## Planned roadmap beyond the minimum experiment

Add a matched local replication of ConciseRL before making comparisons against semantic-reward prior work. The present matched length/prompt experiment cannot establish superiority over that paper's method.

**Bigger models:** profile larger Qwen3.5 policies and stronger local judges. Keep matched compute/data and report resource differences; larger models are not automatically feasible on this GPU. Domain-specialist policies can follow the shared-policy experiment.

**Programmatic Lean complexity:** pin Lean/mathlib and theorem statements, check proofs with the kernel, reject `sorry` and unapproved axioms, and collect elaborated proof-term tree/DAG nodes, depth, shared subterms, dependency graph size and axiom usage. A short tactic call may hide a large elaborated proof. First compare these diagnostics with judge and script-length scores on fixed statements; only then consider a structural reward. Relevant APIs: [Lean.Expr](https://lean-lang.org/doc/api/Lean/Expr.html), [validation](https://lean-lang.org/doc/reference/latest/ValidatingProofs/), [axioms](https://lean-lang.org/doc/reference/latest/Axioms/).

## Progress record

- Workspace initialized with uv; literature retrieved; hardware/account inventory complete.
- Research documents and preregistration published at commit 2994ef3167660b981c64fb18f55ae6df01c39e79.
- Initial modular implementation, counterexample tests, dependency lock, site, paper template and command guide written; implementation push precedes the first test.
- No tests, data generation, inference, or training results exist at this stage.

During runs, inspect progress according to expected run duration: five minutes for >10-minute runs, fifteen for >30-minute runs, thirty for >one-hour runs. Report meaningful changes and active implementation progress promptly. Never install an automation just to watch this task.
