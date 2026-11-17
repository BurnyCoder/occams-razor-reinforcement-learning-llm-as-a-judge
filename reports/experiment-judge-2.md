# Judge audit

| Judge | Dominance | Omission rejection | Invalid JSON | Qualified |
|---|---:|---:|---:|---|
| Qwen3.5-2B | 1/12 | 7/10 | 16 | No |

Evidence: [results/judge-audits/qwen-initial-2b.json](../results/judge-audits/qwen-initial-2b.json).

Dominance Wilson interval: [0.014865094404917095, 0.35387991114111694]. Omission Wilson interval: [0.39677814746114537, 0.8922087325936989]. These intervals do not establish population-wide validity from constructed examples.

The declared qualification decision is retained. The training-only override in amendment 002 permits exploratory rewards after a failed current audit; it does not qualify the judge or waive runtime checks. Independent evaluation still requires a passing audit.
