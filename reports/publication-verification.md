# Publication and presentation verification

2026-11-19. Implementation and regression tests were pushed before execution under the existing guard. The final CPU/container suite and UI tests passed; counts are parsed from [JUnit records](../results/tests/) into the [independent fact-check](independent-fact-check.md). Passing tests does not establish scientific validity.

The browser review exercised the actual local dashboard, live refresh, domain selection, empty evaluation/example states, a synthetic partial-domain fixture and escaped candidate text. Desktop and mobile views were inspected. Partial results remain marked unverified/exploratory; missing overall results do not become zero scores. A zero-only chart now has distinct displayed axis bounds. No model inference was run.

Every rendered PDF page was inspected, including result tables, references, the architecture diagram, training plots and the final AI statement. Plot indices distinguish pre-update rollouts from completed-update loss; planned evaluation is labeled. The paper's simplicity normalization was made explicit during this inspection.

The PDF annotation check also found closing sentence punctuation included in an inline source URL. The renderer now excludes that punctuation while retaining balanced URL parentheses; [the PDF regression](../tests/test_pdf.py) checks the actual annotation bytes. This correction changes no measured result.

Repeated `occam report` generation preserved summary measurements and the independently recomputed frozen-data, checkpoint, pilot and test records. Comparison removed only the declared generation timestamp; the file-coverage inventory expanded when generated files became tracked. The paper's explicitly described wording correction is separate from measured content. Both generations remain in Git history.

The [feature PR](https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/pull/9) is merged. [Pages deployment](https://github.com/BurnyCoder/occams-razor-reinforcement-learning-llm-as-a-judge/actions/runs/44499123117) succeeded, and the deployed dashboard was inspected in the browser. Source, paper, reports and dataset card were uploaded to the existing Hub artifact repository; the pilot model card and existing collection description were updated. No checkpoint was rewritten.

The [dated machine-readable receipt](../results/publication-verification.json) records the verified Git/Hub revisions, public URLs and SHA-256 identities, plus the repeated-generation comparison. All listed checks passed. [The verification command](../scripts/verify_publication.py) compares Pages bytes, source/card bytes, frozen-data hashes, checkpoint identities and existing collection membership. Large checkpoint objects are checked against the Hub's SHA-256 LFS identities; smaller objects are downloaded and hashed. This verifies currently published contents, not the historical time at which each original artifact first became public.

The archived receipt refers to its explicitly recorded revision, before this receipt was added to the source snapshot. Subsequent final-snapshot verification writes `.local/publication-verification.json`; keeping that live receipt outside Git avoids a circular source-commit identity.

The original pilot receipt identifies the adapter upload. The subsequent checkpoint upload is a separate Hub commit. Neither receipt alone proves successful interrupted-training resumption. Current model and dataset contents were checked without rewriting the original receipt or checkpoint.
## November 26 publication evidence

The dated audit receipt binds the merged Git revision to the matching Hub source revision, every tracked source object, the frozen split hashes, the rendered site, the pilot card and checkpoint identities. The Hub appends its binary storage rules to the source attributes file. The receipt is stored separately so its contents can identify the exact published source commit. The November 19 receipt describes the earlier revision. [Dated publication receipt](https://huggingface.co/datasets/BurnyCoder/occam-rl-artifacts/blob/main/verification/2026-11-26-publication.json) · [Verification program](../scripts/verify_publication.py)
