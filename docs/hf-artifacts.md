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

`manifests/data.json` records file hashes/counts and task metadata for `frozen-data/`; revisions, seeds and exclusion records live separately as described in [the data contract](https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/blob/main/docs/data.md). `runs/` contains timestamped records, including failures. The published pilot finished scheduled updates with no verifier-accepted trace or recorded tensor change. Its checkpoint exists, but interrupted-training resumption remains untested. Baseline and held-out policy evaluations are pending. No adapter is required for the planned unchanged-model prompt controls. Measured tables, paper, card metrics and explorer data share the reporting generator; see the [independent evidence audit](https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/blob/main/reports/independent-fact-check.md).
