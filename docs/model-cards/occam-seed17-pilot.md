---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
tags:
- reinforcement-learning
- grpo
- lora
---
# occam-seed17-pilot

Research author: Libor “Burny” Burian. This LoRA artifact has no demonstrated advantage over the base model or baselines.

Pilot: True. Training judge qualified: False.
Training-only judge waiver: True.
Completion denotes finished scheduled updates, not successful learning. Parameter changes below are the recorded runtime comparison, not an independent reconstruction of the initial tensors.

## Training evidence

| Run | Updates | Correct / traces | Truncated | Groups with reward spread | Changed tensors | State |
|---|---:|---:|---:|---:|---:|---|
| occam-seed17-pilot | 4 | 0 / 16 | 16 | 0 / 4 | 0 | complete pilot |

occam-seed17-pilot: 16 recorded rollouts across 4 distinct tasks. Repeated rollouts are not independent tasks. Source: https://huggingface.co/datasets/BurnyCoder/occam-rl-artifacts/blob/main/runs/occam-seed17-pilot/events.jsonl

| Verifier rejection reason | Recorded traces |
|---|---:|
| Expected a numeric final answer | 2 |
| Expected exactly one code submission | 4 |
| Expected exactly one final answer | 10 |

## Held-out evaluation

| Run | Split | N | Accuracy | Adequacy | C × A × S | Tokens | Status |
|---|---|---:|---:|---:|---:|---:|---|
| occam-seed17-pilot | pending | — | — | — | — | — | not evaluated |

Code, limitations, data rights, and the independent fact-check: https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/blob/main/reports/independent-fact-check.md

The base model and project code are Apache-2.0; dataset records retain their separate source licenses. Checkpoint files are available under training-state; successful interrupted-training resumption has not been demonstrated.

The frozen MBPP+ prompts contain assertions in 150/150 tasks. Examples share scored input values in 149 tasks, including 146 matches with the same argument types. The physics reference review flags 39/150 public targets for adjudication. Choosing the candidate with fewer characters selects the designated preference on 36/36 constructed pairs. Qualification passed in 0/3 completed sets. H1 is untested. H2 is untested. H3 is untested. [Scientific audit](https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/blob/main/reports/reviews/2026-11-26-scientific-audit.md)
