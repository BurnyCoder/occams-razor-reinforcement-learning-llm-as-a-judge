# Readiness evidence

| Judge | Dominance | Omission rejection | Invalid judgments | Qualified |
|---|---:|---:|---:|---|
| Qwen3.5-4B (v2) | 0/12 | 5/10 | 18 | No |
| Qwen3.5-4B (v3) | 4/12 | 8/10 | 9 | No |
| Qwen3.5-2B | 1/12 | 7/10 | 16 | No |

GPU/state prerequisite measurements (synthetic inputs; not learning results):

| Check | Recorded result |
|---|---|
| Backward minimum free VRAM (MiB) | 3860 |
| Generation minimum free VRAM (MiB) | 4141 |
| Role-switch trainable tensors unchanged | True |
| Role-switch CUDA RNG unchanged | True |
| Optimizer state-entry count preserved after reload | True |
| Profile passed | True |


Constructed specification checks are not held-out policy accuracy or human preference validation. Invalid judgments include JSON syntax errors, schema violations and contradictory adequacy/missing-reasoning fields; they do not count as successful omission rejections.

Profile evidence: [recorded measurements](../manifests/preflight.json). The optimizer check compares state-entry counts after loading; it does not compare every tensor or test interrupted training. Role-switch equality covers trainable tensors and the current CUDA RNG, not every model buffer or every RNG.

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
