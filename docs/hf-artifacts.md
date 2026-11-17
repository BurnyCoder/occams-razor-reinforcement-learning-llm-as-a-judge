---
license: other
license_name: source-specific-research-artifacts
license_link: https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/blob/main/docs/sources.md
task_categories:
- text-generation
language:
- en
tags:
- reinforcement-learning
- reproducibility
---
# Occam RL research artifacts

Research author: Libor “Burny” Burian. This dataset repository mirrors published source, frozen task splits, complete model outputs and evaluation records for the [research repository](https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge). Uploads do not imply that training or evaluation succeeded. Inspect run status and raw records.

The implementation and original procedural generators are Apache-2.0. Public task records retain their original licenses and attribution: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) MIT, [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) CC-BY-4.0, and [UGPhysics](https://huggingface.co/datasets/UGPhysics/ugphysics) CC-BY-NC-SA-4.0. UGPhysics is used only for evaluation. Records and logs containing those tasks retain the source's conditions; this mixed collection is not covered by a blanket Apache license. See each task's source/license fields and the linked source notes before reuse.

`manifests/data.json` records the exact bytes, split IDs and provenance of `frozen-data/`. `runs/` contains complete timestamped records, including unsuccessful or partial attempts. Prompt-only baselines have unchanged policy weights and therefore no adapter repository. Trained adapters and resumable optimizer states are separate model items in the same project collection. The README, paper and result explorer are regenerated from the records in the linked GitHub repository.
