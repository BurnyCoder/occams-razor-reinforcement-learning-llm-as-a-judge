"""Validate declared evaluation populations before assigning scientific status.

This enforces this project's preregistration, not a new statistical estimator.
Missing legacy provenance stays readable but cannot certify a comparison.
"""

import math
import re

from pydantic import ValidationError

from .schema import DOMAINS, PROMPT_ARMS, TRAINED_ARMS, Judgment


def evaluation_issues(manifest: dict, rows: list[dict], frozen: dict) -> list[str]:
    """Check membership, metadata and provenance separately from measured outcomes."""
    issues = []
    expected, actual = manifest.get("tasks", []), [r.get("task_id") for r in rows]
    if not expected or len(set(expected)) != len(expected):
        issues.append("Missing or duplicate manifest task IDs")
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        issues.append("Missing, duplicate or substituted result task IDs")
    arm, split = manifest.get("arm"), manifest.get("split")
    if arm not in (*TRAINED_ARMS, *PROMPT_ARMS) or not isinstance(manifest.get("seed"), int):
        issues.append("Missing or invalid arm/seed identity")
    population = {t["id"]: t for t in frozen.get("tasks", []) if t["split"] == split}
    if not population or not set(expected) <= population.keys():
        issues.append("Tasks do not match the frozen split")
    if manifest.get("limit_per_domain") is None and set(expected) != population.keys():
        issues.append("Unlimited evaluation does not declare the full frozen split")
    if manifest.get("data") != frozen.get("files", {}).get(f"{split}.jsonl") or not manifest.get("data"):
        issues.append("Missing or mismatched frozen data provenance")
    provenance = manifest.get("provenance", {})
    if not isinstance(manifest.get("protocol"), dict) or not manifest["protocol"].get("scorer"):
        issues.append("Missing prospective scoring protocol")
    hashes = [(manifest.get("config_hash"), 64), (manifest.get("git_commit"), 40),
              (provenance.get("policy_revision"), 40), (provenance.get("judge_hash"), 64)]
    if arm not in PROMPT_ARMS:
        hashes.append((provenance.get("adapter_sha256"), 64))
    if any(not isinstance(value, str) or not re.fullmatch(fr"[0-9a-f]{{{length}}}", value)
           for value, length in hashes):
        issues.append("Missing or invalid configuration/model/judge/source provenance")
    if {r.get("domain") for r in rows} != set(DOMAINS):
        issues.append("Required domain coverage is incomplete or invalid")
    for row in rows:
        task = population.get(row.get("task_id"), {})
        verifier = row.get("verifier")
        if (not isinstance(verifier, dict) or type(verifier.get("correct")) is not bool
                or verifier["correct"] != row.get("correct") or not verifier.get("reason")
                or row.get("source") != task.get("source")):
            issues.append("Correctness or source disagrees with primary evidence")
        if (row.get("arm"), row.get("seed"), row.get("git_commit")) != (
                arm, manifest.get("seed"), manifest.get("git_commit")):
            issues.append("Result identity disagrees with evaluation manifest")
        if (row.get("domain"), row.get("family")) != (task.get("domain"), task.get("family")):
            issues.append("Result metadata disagree with frozen task")
        # The preregistration scores failed judgments zero; missing/misrepresented
        # evidence is distinct from an explicitly recorded failed judgment.
        try:
            if row.get("judge_invalid") is True and row.get("judgment") is None:
                if row.get("adequate") != 0 or row.get("simplicity") != 0:
                    issues.append("Failed judgment must have zero adequacy and simplicity")
            else:
                judgment = Judgment.model_validate(row.get("judgment"))
                if (row.get("judge_invalid") is not False or row.get("adequate") != int(judgment.adequate)
                        or row.get("simplicity") != judgment.simplicity):
                    issues.append("Judgment disagrees with recorded scores")
        except ValidationError:
            issues.append("Missing or invalid judgment evidence")
        for metric in ("correct", "adequate", "simplicity", "quality", "tokens", "formal_length", "seconds", "truncated"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                issues.append(f"Missing or invalid measurement: {metric}")
            elif metric in ("correct", "adequate", "truncated") and value not in (0, 1):
                issues.append(f"Invalid binary measurement: {metric}")
            elif metric in ("simplicity", "quality") and value > 1:
                issues.append(f"Out-of-range measurement: {metric}")
        if all(isinstance(row.get(k), (int, float)) for k in ("correct", "adequate", "simplicity", "quality")):
            if not math.isclose(row["quality"], row["correct"] * row["adequate"] * row["simplicity"], abs_tol=1e-12):
                issues.append("Composite quality disagrees with its factors")
    return sorted(set(issues))
