# Publication and presentation verification

2026-11-19. Implementation and regression tests were pushed before execution under the existing guard. The final CPU/container suite and UI tests passed; counts are parsed from [JUnit records](../results/tests/) into the [independent fact-check](independent-fact-check.md). Passing tests does not establish scientific validity.

The browser review exercised the actual local dashboard, live refresh, domain selection, empty evaluation/example states, a synthetic partial-domain fixture and escaped candidate text. Desktop and mobile views were inspected. Partial results remain marked unverified/exploratory; missing overall results do not become zero scores. A zero-only chart now has distinct displayed axis bounds. No model inference was run.

Generated measurement stability, every PDF page and final remote publication checks remain to be recorded after report generation. [The publication verification command](../scripts/verify_publication.py) checks Pages bytes, source/card bytes, frozen-data hashes, checkpoint identities and existing collection membership. Its live receipt is kept in ignored `.local/publication-verification.json` to avoid a circular source-commit identity.

The original pilot receipt identifies the adapter upload. The subsequent checkpoint upload is a separate Hub commit. Neither receipt alone proves successful interrupted-training resumption. Current model and dataset contents will be checked without rewriting the original receipt or checkpoint.
