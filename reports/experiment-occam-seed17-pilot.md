# occam-seed17-pilot

{
  "name": "occam-seed17-pilot",
  "arm": "occam",
  "seed": 17,
  "pilot": true,
  "steps": 4,
  "judge_qualified": false,
  "exploratory_judge_override": true,
  "target_steps": 4,
  "complete": true,
  "seconds": 1148.1373695599978,
  "changed_parameter_tensors": 0,
  "git_commit": "825cc9b8c19fdc5b14341ca723267faa0515c1ee",
  "fingerprint": "1f9cf1fe8fb0c8985543d4fcdb21389bb5439a0c53bfceccfade46a10a7763df"
}

# Occam pilot investigator notes

2026-11-17. This exploratory pilot tests runtime, reward variation, parameter changes and timing; it is excluded from retained policy comparisons.

The initial attempt used the published current Qwen audit under the explicit training-only waiver. It loaded the policy but failed before any rollout or optimizer update because the JSONL reader split a Unicode separator inside the GSM8K question `gsm8k/train/2381`. Frozen dataset hashes still matched. The reader fix follows LF record boundaries and preserves every task byte. See `reports/runtime-integration.md` and the immutable startup log in `results/runtime/occam-pilot-startup.jsonl`.

Retry from the original seed is authorized only before the first model prompt or rollout. The original failure stays in this event log; a subsequent attempt appends its own source identity. No learning or checkpoint-resume success is inferred from startup recovery. Hyperparameters remain those in the preregistered configuration and are not selected from held-out performance.

## Completed thinking-mode pilot

The retry completed its planned optimizer steps and saved the adapter plus optimizer, scheduler, trainer and random-number-generator state. Every sampled trace reached the completion cap, failed verification and received zero reward. Every logged gradient norm was zero, and the final parameter comparison found no changed trainable tensors. The table below is regenerated from the event log and status artifact; no held-out accuracy is inferred.

The correctness gate never admitted a trace to the training judge. This run therefore does not test whether the semantic simplicity reward improves learning. It establishes a policy-generation failure for this short, thinking-enabled pilot. The small sampled task set spans all requested domains but cannot estimate per-domain or general model accuracy. Completion of checkpoint serialization also does not establish successful interrupted-training resumption.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B#thinking-mode), checked 2026-11-17, states that this model defaults to non-thinking mode and documents thinking loops that can prevent termination. The observed repeated arithmetic and unfinished derivations are consistent with that warning, but no causal ablation has yet been run. The next controlled experiment should disable thinking while preserving explicit necessary reasoning, the task stream, sampler, output cap, reward and original weights. Preserve this pilot and publish the amendment and code before the new run. Do not freeze the retained training budget from a pilot with no parameter changes.


## Training diagnostics

| Run | Updates | Correct / traces | Truncated | Groups with reward spread | Changed tensors | State |
|---|---:|---:|---:|---:|---:|---|
| occam-seed17-pilot | 4 | 0 / 16 | 16 | 0 / 4 | 0 | complete pilot |

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