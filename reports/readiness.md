# Readiness evidence

| Judge | Dominance | Omission rejection | Invalid JSON | Qualified |
|---|---:|---:|---:|---|
| Qwen3.5-2B | 1/12 | 7/10 | 16 | No |

Constructed specification checks are not held-out policy accuracy or human preference validation. Invalid or contradictory JSON does not count as a successful omission rejection.

| Split | Domain | Source | Tasks |
|---|---|---|---:|
| calibration | code | procedural-v1 | 50 |
| calibration | math | procedural-v1 | 50 |
| calibration | physics | procedural-v1 | 50 |
| dev | code | procedural-v1 | 30 |
| dev | math | procedural-v1 | 30 |
| dev | physics | procedural-v1 | 30 |
| ood | code | procedural-v1 | 60 |
| ood | math | procedural-v1 | 60 |
| ood | physics | procedural-v1 | 60 |
| test | code | evalplus/mbppplus-v0.2.0 | 150 |
| test | code | procedural-v1 | 150 |
| test | math | openai/gsm8k | 150 |
| test | math | procedural-v1 | 150 |
| test | physics | UGPhysics/ugphysics | 150 |
| test | physics | procedural-v1 | 150 |
| train | code | google-research-datasets/mbpp | 256 |
| train | code | procedural-v1 | 256 |
| train | math | openai/gsm8k | 256 |
| train | math | procedural-v1 | 243 |
| train | math | reasoning-gym/simple_equations | 13 |
| train | physics | procedural-v1 | 512 |
