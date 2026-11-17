# Literature review

Libor “Burny” Burian · 2026-11-17 · Before model experiments

## Scope and synthesis

The question is whether semantic parsimony, conditioned on both correctness and adequate justification, improves small-model mathematical, physical, and coding solutions relative to length rewards. This is an empirical operationalization, not a formalization of the entire philosophical principle.

| Perspective | What it contributes | Boundary for this experiment |
|---|---|---|
| William of Ockham, as discussed in the [Stanford Encyclopedia](https://plato.stanford.edu/entries/ockham/) | Avoid positing entities without necessity; necessity is essential to the maxim. | Do not remove necessary steps or assumptions. The familiar maxim has a complicated attribution history. |
| [SEP: Simplicity](https://plato.stanford.edu/entries/simplicity/) | Distinguishes elegance/syntactic simplicity from parsimony/ontological simplicity; simplicity depends on representation and background commitments. | Count concepts and assumptions separately from words; hold the task and explanatory obligations fixed. |
| [Grünwald's MDL tutorial](https://arxiv.org/abs/math/0406077) | Trades model description against unexplained data through a coding framework. | Motivates economical explanation, but a judge score is not a description-length code or MDL estimator. |
| [Wikipedia's Occam overview](https://en.wikipedia.org/wiki/Occam%27s_razor) | Accessible overview and route to historical/statistical perspectives. | Secondary orientation; contested historical attribution and empirical claims must be checked against primary sources. |

Our synthesis: **prefer the least unnecessarily complicated solution among solutions that meet the same correctness and explanatory obligations**. “Fewest elements” means fewest necessary conceptual commitments, not fewest printed characters. In physics, an explicitly needed idealization is preferable to silently omitting it. In code, readable use of an established abstraction can be simpler than a short obfuscated expression. In mathematics, a valid standard lemma can shorten a proof without making a missing justification valid. These are proposed rubric choices, to be audited.

## Closest technical work

[ConciseRL](https://aclanthology.org/2025.findings-emnlp.927/) is particularly close prior art: it uses an LLM conciseness reward, correctness gating, and explicitly discourages answer-only traces. Our separately audited binary sufficiency gate and multidimensional protocol are experimental design differences, not claims to invent semantic conciseness or reasoning completeness. Its [repository](https://github.com/RazvanDu/ConciseRL) is Apache-2.0.

Reading its appendix also provides a concrete caution. Figure 11 reaches the correct final answer for the roots-of-unity problem but describes all roots of `z^4+z^2+1` as primitive sixth roots. Algebraically, a primitive cube root satisfies `z^3=1`, `z!=1`, and therefore `z^4+z^2+1=z+z^2+1=0`; its order is three. This is a deductive counterexample to that intermediate statement, not an estimate of the method's error rate. The final answer remains six because primitive sixth roots are also present. Answer verification alone cannot certify the reasoning. [Source, Figure 11](https://aclanthology.org/2025.findings-emnlp.927.pdf).

| Source | Method and evidence relevant here | Reuse / difference |
|---|---|---|
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO uses groups of outputs and relative rewards, avoiding a separate learned critic. | Reuse maintained GRPO machinery, with explicitly documented objective changes. Its original experiments do not validate our judges or GPU configuration. |
| [Understanding R1-Zero / Dr. GRPO](https://arxiv.org/abs/2503.20783) | Analyzes response-length and group-standard-deviation normalization; studies template effects, pretraining, ablations, and independent runs. | Use a constant token denominator and unscaled centered rewards. Keep a small KL term because our semantic reward is fallible; this differs from the paper's zero-KL experiments. |
| [ShorterBetter](https://arxiv.org/abs/2504.21370) | Defines the shortest correct sampled response as a length target, with fallback when no correct answer exists. Includes math/code evaluations, a collapse ablation, and reasoning-category analysis. | Strong prior art for concise RL. Our length arms are controlled matched-budget baselines, not exact reproductions of its SOL objective. |
| [O1-Pruner](https://arxiv.org/abs/2501.12570) | Studies length-harmonizing fine-tuning for efficient reasoning. | Motivates reporting accuracy and length jointly. Original code is [MIT](https://github.com/StarDewXXX/O1-Pruner). |
| [StepPruner](https://arxiv.org/abs/2510.03805) | Uses step-level reasoning structure and pruning incentives, including discussion of reward exploitation. | Counting paragraphs is not semantic simplicity; representation-changing attacks belong in our audit. [Code](https://github.com/Guncuke/StepPruner). |
| [LLM-as-a-judge / MT-Bench](https://arxiv.org/abs/2306.05685) | Documents position, verbosity, and self-preference limitations in LLM judgment. | Blind judges to arm names; use a different model family for evaluation, perturb formatting, and report audit failures. Agreement does not prove validity. |
| [G-Eval](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/03/2303.16634.pdf) | Separates criteria, generated evaluation steps and form-based scoring, with probability-weighted ratings. Its evaluations concern summarization/dialogue and show sensitivity to evaluator design. | Clarify rating direction and evidence fields; this is not a replication of its scoring rule or proof of reliability for small local judges. |
| [GFPO](https://arxiv.org/abs/2508.09726) | Further evidence that group-level filtering and efficient reasoning objectives are active research topics. | Novelty must be limited to this particular adequacy-gated semantic comparison, not length-aware RL generally. |

The argument “complex explanations overfit” does not itself establish better generalization here. Statistical model complexity, explanatory complexity, and output length are different constructs. We test held-out tasks and template families rather than treating a bias–variance analogy as evidence.

## Reusable implementation and data

Use [TRL](https://huggingface.co/docs/trl/grpo_trainer) rather than reimplementing PPO clipping; [Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune) for documented Qwen3.5 optimization; [PEFT](https://huggingface.co/docs/peft/index) for adapters; [Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_5) for text-only Qwen3.5; [bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) for judge quantization; [llama.cpp](https://github.com/ggml-org/llama.cpp) for local Gemma GGUF inference. Version and model revision pins must be recorded. Supporting a model in documentation is not proof that a specific Blackwell laptop configuration works.

[Reasoning Gym](https://github.com/open-thought/reasoning-gym) supplies procedural-task patterns and exact-answer generators. [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), [UGPhysics](https://huggingface.co/datasets/UGPhysics/ugphysics), and [EvalPlus](https://github.com/evalplus/evalplus) supply public comparison tasks. Test counts alone do not establish correctness; generated programs must execute in isolation, and physics answers require unit-aware checks. Public benchmark exposure during pretraining remains unknown.

## Research practice

[Lones](https://arxiv.org/abs/2108.02497) motivates controls against leakage, inadequate baselines, and evaluation mistakes. [Lipton and Steinhardt](https://arxiv.org/abs/1807.03341) motivate separating explanation from speculation and improvements from claims of understanding. The [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) motivates explicit uncertainty, compute accounting, reproducibility, and limitations. The user's requested [Nanda research-process essay](https://www.lesswrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand) is contextual advice; its browser response was unreadable, so no substantive claim is attributed to it. [Schulman's research guide](https://joschu.net/blog/opinionated-guide-ml-research.html) is an additional methodological reference, not empirical support for the hypothesis.

## Reading and fact-check record

Primary papers were retrieved as PDFs in a local, untracked literature cache. Technical decisions are checked against paper methods and current official documentation; sections not yet fully inspected remain on the implementation checklist. Numerical performance claims from prior papers are deliberately not transferred to this hardware or model family. The review is not an exhaustive novelty search and does not establish absence of similar work. Sources may be updated; resolved revisions belong in the experiment manifest.
