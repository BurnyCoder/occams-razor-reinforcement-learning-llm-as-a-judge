# Code task contract corrections before freezing data

No training data or policy performance had been frozen or measured when these defects were identified. The first preparation process was interrupted while still importing its dependencies; its failure log is retained.

Procedural code questions explicitly forbid input mutation. The runner originally checked only return values, so sorting a list in place could pass despite violating that constraint. The verifier now snapshots JSON arguments and rejects changed inputs when the task requires preservation. A counterexample test has the right numeric output but illegally mutates the list.

MBPP natural-language statements do not always specify the exact callable name used by their tests. The adapter now exposes the tested reference signature, without its body, and preserves the dataset's test setup statements. Ambiguous contracts are excluded before model performance is observed. Selected MBPP reference solutions must pass the same container verifier before entering frozen training data. This makes eligibility follow the declared reference-check rule; it is not selection on trained-model outcomes. Source: [official MBPP dataset](https://huggingface.co/datasets/google-research-datasets/mbpp), [Python AST interface](https://docs.python.org/3.12/library/ast.html).
