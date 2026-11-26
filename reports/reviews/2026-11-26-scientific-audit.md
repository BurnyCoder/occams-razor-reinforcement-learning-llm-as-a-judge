# Scientific audit of the registered comparisons

Libor “Burny” Burian · 2026-11-26

The frozen MBPP+ prompts contain assertions in 150/150 tasks. Examples share scored input values in 149 tasks, including 146 matches with the same argument types. The physics reference review flags 39/150 public targets for adjudication. Choosing the candidate with fewer characters selects the designated preference on 36/36 constructed pairs. Qualification passed in 0/3 completed sets. H1 is untested. H2 is untested. H3 is untested.

## What the pilot measured

The trace and token records reproduce 0/16 verifier acceptances on 4 distinct training tasks, with 0 semantic judgments. The observed reward gate prevented a test of the semantic bonus. The saved status reports the trainable-tensor comparison. A cause for the unsuccessful completions and a population performance effect remain open.

## Qualification replay

| Set | Valid outputs | Invalid outputs | Valid pairs | Strict wins | Simplicity ties among valid pairs | Valid omissions | Omissions rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| judge-audit.json | 18/34 | 16 | 2/12 | 1 | 1 | 8/10 | 7 |
| judge-audit-v2.json | 16/34 | 18 | 1/12 | 0 | 0 | 6/10 | 5 |
| judge-audit-v3.json | 25/34 | 9 | 6/12 | 4 | 1 | 8/10 | 8 |

Interrupted `2026-11-17T01-32-58.043+00-00-qualify.jsonl` recorded 0 prompts and 0 outputs (0 valid, 0 invalid). These attempts have no qualification verdict.

Interrupted `2026-11-17T01-45-19.407+00-00-qualify.jsonl` recorded 0 prompts and 0 outputs (0 valid, 0 invalid). These attempts have no qualification verdict.

Interrupted `2026-11-17T03-11-13.694+00-00-qualify.jsonl` recorded 3 prompts and 2 outputs (2 valid, 0 invalid). These attempts have no qualification verdict.

Math delimiters appear only in the preferred response on 24 pairs. These are deliberately constructed examples. Comparing sets confounds changed stimuli, prompts and runtime. Both semantic recognition and reliance on surface cues require counterbalanced evidence. The Wilson intervals in the JSON describe the constructed checks. The logs include interrupted attempts.

## Public task scoring

Across 16115 scored input occurrences, visible argument values match 149 base and 25 additional-suite occurrences. The remaining 15941 occurrences differ from matched examples. The native checker accepted 150/150 frozen canonical programs. Occurrences can repeat. In 3 tasks, examples use tuples where the scorer supplies lists with the same elements. Mbpp/87 supplies different dictionary arguments. Per-task records distinguish exact types from sequence values and retain any unparsed assertions. The original MBPP+ training/test ID exclusion still holds. Visible worked examples are part of the public task design.

The prospective scorer runs the pinned EvalPlus comparator on native candidate and reference values in a CPU container. Mbpp/111 permits set-equivalent reordering, Mbpp/257 requires the returned tuple, and Mbpp/581 accepts an alternative pyramid-height interpretation. A public example's tolerance can differ from the benchmark comparator. Mbpp/82 illustrates this distinction. The project still uses its own frozen subset, prompts, resource limits and three-domain aggregation. [Upstream evaluator](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/__init__.py)

The physics review covers all 150 public targets; 70 had empty unit fields. Each row links the frozen target to the original question and worked solution. Some empty units designate ratios. Others conceal physical dimensions. Source contradictions and missing context are flagged before evaluation. The stored population remains frozen. [Per-task decisions](../../configs/physics-review.psv) · [Raw audit](../../results/scientific-audit.json)

## Hypotheses, sources and limits

The primary quality estimand uses paired task differences within seed, equal domain weights, family resampling and two Holm-adjusted one-sided tests. Accuracy uses the registered lower confidence bound overall and in each domain. H3 has three secondary paired intervals. A concentrated cluster distribution, a degenerate bootstrap or few seeds limits population inference. A literal registered threshold alone cannot validate a flawed measurement.

The original and revised versions have distinct scoring identities. Public source fidelity and byte equality can be checked; independent correctness of every external solution, general judge validity, model pretraining contamination, and historical push times remain unresolved. The claim register and file inventory give evidence or limitations for the repository's substantive claims. [Registered tests](../preregistration.md) · [Amendment](../amendment-004-scoring.md) · [Claim register](../claim-register.json) · [Primary-source review](../source-verification.md).

## Prompt-only control context

Counts use the pinned tokenizer, chat template and complete frozen test questions. Over-limit IDs are in the raw audit. A context failure prevents the complete registered comparison.

| Control | Test prompts | Maximum tokens | Configured limit | Over limit |
|---|---:|---:|---:|---:|
| prompt_ordinary | 900 | 417 | 1536 | 0 |
| prompt_concise | 900 | 429 | 1536 | 0 |
| prompt_occam | 900 | 874 | 1536 | 0 |

## Frozen selection and generalization

The frozen files contain 2856 distinct task IDs. Each selected public record was reconciled with its upstream source. The native procedural answers were independently recomputed from the questions or input lists. Normalized questions do not repeat and the held-out family split shares no frozen family labels with training. This check cannot identify benchmark examples encountered in model pretraining or every pair of semantically similar tasks.

| Completed preparation · exclusion review | Events |
|---|---:|
| historical reference-execution failure; hardware timing and exception not independently reproduced | 6 |
| native non-JSON input confirmed | 37 |
| normalized duplicate confirmed | 37 |
| official training ID confirmed | 108 |
| source eligibility reproduced | 1048 |
