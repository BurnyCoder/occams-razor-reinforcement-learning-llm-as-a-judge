# Scientific findings

The frozen MBPP+ prompts contain assertions in 150/150 tasks. Examples share scored input values in 149 tasks, including 146 matches with the same argument types. The physics reference review flags 39/150 public targets for adjudication. Choosing the candidate with fewer characters selects the designated preference on 36/36 constructed pairs. Qualification passed in 0/3 completed sets. H1 is untested. H2 is untested. H3 is untested.

| Run | Updates | Correct / traces | Truncated | Groups with reward spread | Changed tensors | State |
|---|---:|---:|---:|---:|---:|---|
| occam-seed17-pilot | 4 | 0 / 16 | 16 | 0 / 4 | 0 | complete pilot |

occam-seed17-pilot: 16 recorded rollouts across 4 distinct tasks. Repeated rollouts are not independent tasks. Source: https://huggingface.co/datasets/BurnyCoder/occam-rl-artifacts/blob/main/runs/occam-seed17-pilot/events.jsonl

| Verifier rejection reason | Recorded traces |
|---|---:|
| Expected a numeric final answer | 2 |
| Expected exactly one code submission | 4 |
| Expected exactly one final answer | 10 |

[Methods, scoring defects and qualification replay](reviews/2026-11-26-scientific-audit.md)
