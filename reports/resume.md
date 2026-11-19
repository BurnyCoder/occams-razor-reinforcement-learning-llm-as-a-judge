# Resume point

Original resume point: 2026-11-17; evidence correction: 2026-11-19. Model experiments remain stopped. The independently authorized audit uses existing records and overrides the usage reserve only for that audit. This is an incomplete research project, not a successful training result.

## Preserved state

The original thinking-mode Occam pilot completed and saved its adapter plus full training state. Its source identity is recorded in `results/runtime/occam-pilot-thinking-status.json`; the exact prompts, outputs, verification decisions and optimizer metrics are in `results/runtime/occam-pilot-thinking-events.jsonl`. Generated measurements and interpretation are in [the pilot report](experiment-occam-seed17-pilot.md), [the experiment summary](experiments.md) and `results/summary.json`. The paper and interactive site use those artifacts.

Local run: `D:/AI/occam-rl/runs/occam-seed17-pilot`. The `checkpoint-4` directory contains adapter weights, optimizer, scheduler, RNG and trainer state. It has not been tested by interrupting and resuming a real training run. Do not infer that property from serialization alone. The original startup failure remains in the appended event log.

Every pilot trace hit the cap and failed verification. The log records zero rewards and gradients; the runtime comparison records no changed trainable tensor, without separately saved initialization for reconstruction. Rejections include missing/unparsable submissions, not proof that every reasoning statement was false. The correctness gate never invoked the training judge. The experiment does not test the semantic reward's effectiveness or estimate general accuracy. See the [independent fact-check](independent-fact-check.md).

The current Qwen training-judge audit failed; its explicit exploratory waiver remains authorized. Do not erase or relabel that failure. Independent Gemma qualification is still required before held-out semantic evaluation. Data hashes and the current full GPU profile are frozen in `manifests/`.

## Next controlled experiment — not yet implemented

1. Recheck account usage and preserve the user's minimum reserve. Do not restart work automatically while the user has asked to stop.
2. Publish an amendment testing `enable_thinking=False` for the policy. The [official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B#thinking-mode) documents thinking loops and defaults this small model to non-thinking mode. This motivates a hypothesis, not a proven diagnosis. Keep the explicit necessary-reasoning prompt, original weights, seed, task order, sampler, completion cap and reward unchanged for the initial controlled comparison. Later matched arms must share the selected policy decoding mode.
3. Add an explicit policy-thinking configuration field and a distinct pilot identity, shared by CLI logging, training, publication and budget selection. Current code hardcodes policy thinking and the pilot name. Preserve the old run; do not delete it, overwrite its fingerprint, resume it with a changed configuration or bypass guards. Write tests first and push code, configuration, docs and README before testing.
4. Repeat the configuration-bound GPU profile for the new configuration, preserving the previous profile as an immutable artifact. Publish exact Git/Hugging Face source snapshots before the next model phase. The current preflight hashes the entire configuration; do not edit its hash by hand.
5. Run the new Occam pilot from the original model. Inspect termination, verified correctness, within-group reward variation, gradient norms and changed adapter tensors. If it still cannot learn, retain that outcome and investigate the sampler or a larger GPU-profiled policy with a separate amendment. No held-out tuning.
6. Freeze the common retained update budget only after a completed pilot with actual parameter changes satisfies the existing timing rule. Train and publish retained Occam seed 17 before standalone prompt baselines or baseline training. Then continue the ordered phases in [the plan](plan.md): independent evaluation, prompt controls, length/correctness baselines, shuffled control, matched seeds and any declared ablations.

## Existing commands and environment

Run from the Windows checkout through WSL Ubuntu. The project-local `.venv`, ignored `.env`, model cache and frozen data remain installed; do not delete the uv cache because the environment uses it. All model computation must remain local. No GPU training process is intentionally left running.

The existing completed pilot must not be used to select a retained budget: its no-parameter-change guard correctly rejects it. A new pilot needs the implementation described above; there is intentionally no misleading one-command restart here.

For already implemented reporting/publication, first commit, push, verify a clean checkout at the remote HEAD, then use:

```powershell
wsl -d Ubuntu -- .venv/bin/python -m occam_rl.cli report
# Commit/push regenerated reports before the next command.
wsl -d Ubuntu -- .venv/bin/python -m occam_rl.cli publish --run occam-seed17-pilot
# Commit/push the publication receipt before publishing the exact source snapshot.
wsl -d Ubuntu -- .venv/bin/python -m occam_rl.cli publish --source
```

Generated tables distinguish zero observations from missing evaluations. Keep GitHub, the single Hugging Face collection, the PDF and Pages site consistent. Larger models and programmatic Lean proof complexity remain roadmap items, not completed work.
