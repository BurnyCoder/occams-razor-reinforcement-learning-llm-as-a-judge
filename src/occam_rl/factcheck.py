"""Recompute audit measurements without model inference or modifying original evidence.

Raw experiment outputs are observations of model behavior, not endorsed factual prose.
Sources and human-readable claim decisions are maintained in reports/claim-register.json.
"""

import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from .data import check_split_overlap, question_hash
from .io import ROOT, data_dir, file_hash, read_json, read_jsonl, runs_dir, write_json
from .schema import Task
from .verify import verify

BASELINE = "e58dcdf433e82943e8cc862aa8e09968b6cc28a0"


def pilot_evidence() -> dict:
    """Reconcile stored outputs, token counts, verifier decisions and reward algebra."""
    events = read_jsonl(ROOT / "results/runtime/occam-pilot-thinking-events.jsonl")
    rows = [r for r in events if r["event"] == "rollout"]
    outputs = [r for r in events if r["event"] == "llm_output" and r["role"] == "policy"]
    prompts = [r for r in events if r["event"] == "llm_prompt" and r["role"] == "policy"]
    tasks = {r["id"]: Task.model_validate(r) for r in read_jsonl(data_dir() / "train.jsonl")}
    if len(rows) != len(outputs) or len(rows) != len(prompts):
        raise ValueError("Pilot trace/output/prompt counts disagree")
    for row, output, prompt in zip(rows, outputs, prompts, strict=True):
        if row["trace"] != output["trace"] or row["tokens"] != len(output["completion_ids"]):
            raise ValueError("Pilot trace/token records disagree")
        truncated = not output["completion_ids"] or output["completion_ids"][-1] not in prompt["eos_token_ids"]
        checked = verify(tasks[row["task_id"]], row["trace"])
        if truncated != row["truncated"] or checked.model_dump() != row["verifier"]:
            raise ValueError("Recomputed pilot termination/verifier decision differs")
        judged = row["judgment"]
        simplicity = sum(judged[k] for k in ("assumptions", "concepts", "steps", "clarity")) / 16 if judged else 0
        expected_reward = int(checked.correct) * (1 + .2 * int(bool(judged and judged["adequate"])) * simplicity)
        if expected_reward != row["reward"]:
            raise ValueError("Recorded reward differs from registered equation")
    return dict(rollouts=len(rows), distinct_tasks=len({r["task_id"] for r in rows}),
                correct=sum(r["correct"] for r in rows), truncated=sum(r["truncated"] for r in rows),
                verifier_reasons=dict(Counter(r["verifier"]["reason"] for r in rows)),
                reward_judgments=sum(r["judgment"] is not None for r in rows),
                optimizer_steps=[r["step"] for r in events if r["event"] == "optimizer_step"],
                gradient_norms=[r["grad_norm"] for r in events if r["event"] == "training_metrics" and "grad_norm" in r])


def frozen_evidence() -> dict:
    """Check exact bytes, schema, normalized questions and held-out family separation."""
    manifest = read_json(ROOT / "manifests/data.json")
    tasks, files = [], []
    for name, expected in manifest["files"].items():
        path = data_dir() / name
        rows = read_jsonl(path)
        actual = file_hash(path)
        if actual != expected["sha256"] or len(rows) != expected["count"]:
            raise ValueError(f"Frozen data changed: {name}")
        tasks.extend(Task.model_validate(r) for r in rows)
        files.append(dict(file=name, count=len(rows), sha256=actual))
    check_split_overlap(tasks)
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Frozen data contain duplicate task IDs")
    metadata = [{"id": t.id, "split": t.split, "domain": t.domain, "family": t.family,
                 "source": t.source, "license": t.license, "question_sha256": question_hash(t.question)} for t in tasks]
    if sorted(metadata, key=lambda r: r["id"]) != sorted(manifest["tasks"], key=lambda r: r["id"]):
        raise ValueError("Frozen task metadata disagree with manifest")
    train_families = {t.family for t in tasks if t.split == "train"}
    ood_families = {t.family for t in tasks if t.split == "ood"}
    if train_families & ood_families:
        raise ValueError("Train and OOD families overlap")
    return dict(files=files, unique_tasks=len({t.id for t in tasks}),
                normalized_questions=len({question_hash(t.question) for t in tasks}),
                train_ood_family_overlap=sorted(train_families & ood_families))


def preserved_evidence() -> dict:
    """Compare tracked evidence to the pre-audit commit and hash external checkpoints."""
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE], cwd=ROOT, text=True).splitlines()
    prefixes = ("results/runtime/", "results/judge-audits/", "configs/", "manifests/")
    protected = [p for p in tracked if p.startswith(prefixes) or p == "reports/preregistration.md"]
    hashes = {}
    for name in protected:
        original = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT)
        path = ROOT / name
        # Git may normalize checkout CRLF. Compare repository bytes, then local bytes.
        current = subprocess.check_output(["git", "hash-object", "--path", name, str(path)], cwd=ROOT, text=True).strip()
        baseline_blob = subprocess.check_output(["git", "rev-parse", f"{BASELINE}:{name}"], cwd=ROOT, text=True).strip()
        if current != baseline_blob:
            raise ValueError(f"Original experiment evidence changed: {name}")
        hashes[name] = hashlib.sha256(original).hexdigest()
    checkpoint = runs_dir() / "occam-seed17-pilot/checkpoint-4"
    checkpoint_hashes = {p.name: file_hash(p) for p in sorted(checkpoint.glob("*")) if p.is_file()}
    if not {"adapter_model.safetensors", "optimizer.pt", "trainer_state.json"} <= checkpoint_hashes.keys():
        raise ValueError("Original pilot checkpoint evidence is missing")
    mirror = ROOT / "results/runtime/occam-pilot-thinking-events.jsonl"
    live = runs_dir() / "occam-seed17-pilot/events.jsonl"
    if file_hash(mirror) != file_hash(live):
        raise ValueError("Published pilot log and original run log differ")
    return dict(baseline_commit=BASELINE, git_content_sha256=hashes, checkpoint_sha256=checkpoint_hashes,
                pilot_log_sha256=file_hash(mirror))


def test_records() -> list[dict]:
    """Parse actual test artifacts; never type test totals into reports."""
    records = []
    for path in sorted((ROOT / "results/tests").glob("independent-*.xml")):
        cases = list(ElementTree.parse(path).getroot().iter("testcase"))
        # Count leaf records: Node emits direct testcases, pytest wraps a suite.
        totals = {"tests": len(cases), **{key: sum(c.find(tag) is not None for c in cases)
                  for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}}
        records.append(dict(file=path.relative_to(ROOT).as_posix(), **totals))
    return records


def build_factcheck(data: dict) -> None:
    """Regenerate independent checks, claim inventory and report alongside all figures."""
    register = read_json(ROOT / "reports/claim-register.json")
    frozen, preserved, pilot = frozen_evidence(), preserved_evidence(), pilot_evidence()
    # Checkpoint hashes must remain stable across subsequent report/publication passes.
    previous = ROOT / "results/independent-fact-check.json"
    if previous.exists():
        old = read_json(previous)["preserved"]["checkpoint_sha256"]
        if old != preserved["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during the audit")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    coverage = {}
    for name in tracked:
        if name in ("uv.lock", "LICENSE", ".gitignore", ".gitattributes", ".python-version"):
            continue
        coverage[name] = [c["id"] for c in register["claims"] if any(Path(name).match(p) for p in c["locations"])]
    uncovered = [p for p, claims in coverage.items() if not claims]
    if uncovered:
        raise ValueError(f"Claim inventory lacks files: {uncovered}")
    checks = dict(frozen=frozen, preserved=preserved, pilot=pilot, tests=test_records(), coverage=coverage,
                  claims=len(register["claims"]), generated_at=data["generated_at"])
    write_json(previous, checks)
    lines = ["# Independent fact-check", "", "Libor “Burny” Burian · " + data["generated_at"][:10], "",
             "This audit uses source code, raw records, frozen data and primary sources. Previous prose audits are not evidence. No training or model inference was performed.", "",
             "## Recomputed evidence", "",
             f"Reviewed inventory: {len(coverage)} files mapped to {len(register['claims'])} deduplicated substantive claim groups. Mapping is a coverage aid, not proof of semantic correctness.", "",
             "| Frozen split | Records | SHA-256 |", "|---|---:|---|"]
    lines.extend(f"| {r['file']} | {r['count']} | {r['sha256']} |" for r in frozen["files"])
    lines += ["", f"Unique task IDs: {frozen['unique_tasks']}; normalized questions: {frozen['normalized_questions']}. No cross-split normalized-question duplicates or train/OOD family overlap were found.", "",
              "Original tracked experiment artifacts and preregistration match the pre-audit commit. The published pilot log matches the artifact-drive original. Checkpoint hashes are retained for subsequent comparisons; this does not reconstruct initialization or prove resumability.", "",
              "## Findings and decisions", "", "| ID | Claim / correction | Status | Evidence |", "|---|---|---|---|"]
    for claim in register["claims"]:
        evidence = "; ".join(f"[{s}](../{s})" if not s.startswith("https://") else f"[source]({s})" for s in claim["sources"])
        lines.append(f"| {claim['id']} | {claim['claim']} {claim['resolution']} | {claim['status']} | {evidence} |")
    lines += ["", "## Test records", "", "| Artifact | Tests | Failures | Errors | Skipped |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| [{r['file']}](../{r['file']}) | {r['tests']} | {r['failures']} | {r['errors']} | {r['skipped']} |" for r in checks["tests"])
    lines += ["", "## Limits", "",
              "Passing software tests does not validate judges or causal hypotheses. No held-out evaluation exists. The training judge remains unqualified. Loop susceptibility is documented by Qwen, but no causal ablation establishes why this pilot failed. Original initial trainable tensors were not persisted for an independent before/after reconstruction; zero changes is the recorded runtime comparison. Successful interrupted-training resumption remains untested. Commit timestamps and present remote availability do not prove historical push timing. Public pretraining contamination remains unknown.", "",
              "The claim register records exact sources and whether a statement is observation, deduction, hypothesis, implementation description, or instruction. Historical investigator notes and raw model outputs remain immutable and are not endorsed as factual explanations. See [source review](source-verification.md), [amendment](amendment-003-evidence-reporting.md), and [publication verification](publication-verification.md).", ""]
    (ROOT / "reports/independent-fact-check.md").write_text("\n".join(lines), encoding="utf-8")
