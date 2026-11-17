# Data design and licenses

| Source | Role | License / source |
|---|---|---|
| Original procedural math, mechanics, circuits and pure functions | Train/dev/calibration/test with separate seeds; separate held-out families | Project Apache-2.0; analytic references and hidden tests |
| Reasoning Gym | Reused procedural components and reference design | [Repository](https://github.com/open-thought/reasoning-gym), Apache-2.0 with component notices |
| GSM8K | Official train for training; official test for public evaluation | [Card](https://huggingface.co/datasets/openai/gsm8k), MIT |
| MBPP | Official train only for training | [Card](https://huggingface.co/datasets/google-research-datasets/mbpp), CC-BY-4.0 |
| MBPP+ / EvalPlus | Disjoint evaluation IDs with stronger executable tests | [Repository and notices](https://github.com/evalplus/evalplus); retain upstream task provenance |
| UGPhysics | Eligible English numerical text-only mechanics/electricity, evaluation only | [Card](https://huggingface.co/datasets/UGPhysics/ugphysics), CC-BY-NC-SA-4.0 |

Manifest fields include source/revision/license, task ID, normalized question hash, family, split, seed, answer type, expected unit, eligibility decision, exclusion reason, and generator version. Public problems are never silently reassigned from test to train. Duplicate normalized questions across splits are an error. Task text and full traces belong in appropriately licensed artifacts; Git source files contain IDs and provenance, not an unlicensed redistribution of a benchmark.

Use exact rational arithmetic for procedural reference values. Physics numeric tolerance and unit normalization are fixed per task; multiple answers and symbolic responses need explicit verifier support or pre-performance exclusion. Never accept an arbitrary number appearing anywhere in a response: parse the declared final answer region. Hidden code tests include boundary/property cases; run untrusted completions inside a networkless read-only container with no host directory mounts, limited memory/processes, and a wall-clock timeout. Unit tests run trusted reference examples as well as failure examples.

The first generated manifest must be committed before training. Public dataset/model revisions are resolved and frozen before use. A data-readiness audit checks source counts, split overlap, units, reference tests, and missing strata. Known test contamination during pretraining cannot be removed by local split hygiene; report public and procedural results separately.
