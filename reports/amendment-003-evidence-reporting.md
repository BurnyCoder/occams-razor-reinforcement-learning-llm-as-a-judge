# Amendment 003: evidence and reporting corrections

Libor “Burny” Burian · 2026-11-19. Post-pilot, before any held-out evaluation. The original [preregistration](preregistration.md), raw logs, frozen data, configurations and checkpoints are preserved. This amendment authorizes no model inference or training and changes no reward coefficient, hypothesis, sample target, bootstrap estimand or success threshold.

The user requested an independent audit and explicitly waived the usage reserve for that work. Earlier prose audits are not evidence. [The claim register](claim-register.json) classifies observations, deductions, hypotheses, implementation descriptions and instructions. [The generated audit](independent-fact-check.md) records raw-data checks and preservation hashes.

## Confirmed corrections

- The data manifest contains IDs, split/domain/family/source/license, normalized question hashes and file hashes/counts. Revisions, generator seeds, units and exclusion reasons are in separate artifacts; some originally described per-task fields were never stored. See [data.md](../docs/data.md) and [data.py](../src/occam_rl/data.py).
- MBPP test assertions execute in the same container process as candidate code. Procedural expected outputs remain on the host. Neither finite testing nor container isolation proves universal correctness or immunity to harness manipulation. See [verify.py](../src/occam_rl/verify.py).
- Pilot rollouts repeat tasks. Verifier rejection includes absent or unparsable submissions and is distinct from a demonstrated mathematical error. Zero tensor changes is the recorded runtime comparison; the original initial tensors were not separately persisted. The proposed thinking-loop explanation is a hypothesis, not a causal finding.
- Invalid judgments include syntax, schema and semantic-consistency failures. Explicit failed judgments still receive zero adequacy/simplicity under the original registered failure policy. They do not count as successful omission detection.
- Rollout metrics are sampled before an update; loss metrics follow completed updates. Architecture diagrams identify planned evaluation. Scheduled completion does not imply learning.
- GPU profile checks cover trainable tensors and current CUDA RNG. Optimizer reload compares state-entry counts. Worst-case judge-context fit and actual interrupted-training resumption remain unestablished.

## Evidence requirements

[evaluation_issues](../src/occam_rl/evidence.py) now requires unique result IDs equal to the declared population; unlimited evaluations must declare the entire frozen split. It checks domain/family metadata, arm/seed/source identities, recorded model/judge/config/data provenance, score ranges, judgment consistency and the composite equation. All registered domains must be present for the overall macro average. Partial domains remain visible, without an overall score. These checks validate recorded provenance consistency, not the authenticity of arbitrary user-supplied hashes.

Legacy records stay readable and publishable with explicit incomplete/unverified status. Missing or inconsistent evidence blocks primary analysis. Limited evaluations and incomplete seeds retain exploratory labels. Across seeds, a task's family/domain must agree. The first-Occam receipt must explicitly identify a completed, non-pilot seed-17 run before baselines may execute. [Regression tests](../tests/test_evidence.py) exercise these cases; [software test artifacts](../results/tests/) record outcomes independently of scientific validity.

Historical investigation notes are retained and dated as such. Their claims about push timing, browser inspection, profiling causes or environment versions require original records; a commit timestamp alone is insufficient. Current publication is checked against remote bytes and receipts in [publication verification](publication-verification.md). No claim of an exhaustive novelty search or guaranteed error-free research is made.
