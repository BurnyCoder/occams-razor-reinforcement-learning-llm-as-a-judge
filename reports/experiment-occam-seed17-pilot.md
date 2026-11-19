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

[Original dated investigator notes](https://huggingface.co/datasets/BurnyCoder/occam-rl-artifacts/blob/main/runs/occam-seed17-pilot/notes.md) are preserved as historical interpretation. The current evidence and corrections are in [the independent fact-check](independent-fact-check.md).

Verification failures include missing or unparsable submissions; they do not demonstrate that every intermediate mathematical statement is wrong. Parameter changes are the recorded runtime comparison; initial tensors were not separately persisted for reconstruction. No trace passed the correctness gate, so reward judging was not exercised on an eligible trace. The causal explanation remains untested.

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