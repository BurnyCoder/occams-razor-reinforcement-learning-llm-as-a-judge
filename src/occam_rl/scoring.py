"""Versioned reference decisions for future scoring. Historical task bodies stay immutable."""

import csv
from functools import lru_cache

from .io import ROOT, digest, file_hash

VERSION = "2026-11-26.1"
MBPP = "evalplus/mbppplus-v0.2.0"
UG = "UGPhysics/ugphysics"
EVAL_IMAGE = "occam-evalplus:0.3.1"
REVIEW = ROOT / "configs/physics-review.psv"


@lru_cache(maxsize=1)
def physics_review():
    """Task-level rulings are keyed by the immutable public source identifier."""
    subjects = {"M": "ClassicalMechanics", "E": "ClassicalElectromagnetism"}
    with REVIEW.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="|"))
    review = {f"ugphysics/{subjects[r['key'][0]]}/{r['key'][1:]}": r for r in rows}
    if len(review) != len(rows):
        raise ValueError("Duplicate physics review ID")
    return review


def unresolved(tasks):
    """A missing or disputed reference blocks that evaluation before any policy generation."""
    reviews = physics_review()
    return [t.id for t in tasks if t.source == UG and
            (t.id not in reviews or reviews[t.id]["review"].startswith("!"))]


def analysis_family(task_id, recorded):
    """Cluster known shared source templates while preserving the original family field."""
    family = physics_review().get(task_id, {}).get("cluster", ".")
    return recorded if family == "." else "ugphysics/template/" + family


def scoring_identity():
    """Include actual scorer and adjudication bytes in each prospective evaluation."""
    paths = [REVIEW, ROOT / "src/occam_rl/verify.py", ROOT / "src/occam_rl/scoring.py",
             ROOT / "docker/evalplus_runner.py", ROOT / "docker/EvalPlus.Dockerfile"]
    return digest([VERSION, *(file_hash(path) for path in paths)])


@lru_cache(maxsize=1)
def mbpp_release():
    """Use EvalPlus' native input deserialization for precisely the frozen release."""
    from evalplus.data import get_mbpp_plus
    return get_mbpp_plus(version="v0.2.0")


def mbpp_spec(task):
    """Bind the base/plus boundary, native inputs, reference program and tolerance to source."""
    row = mbpp_release()[task.id]
    inputs = row["base_input"] + row["plus_input"]
    reference = row["canonical_solution"]
    if task.reference not in (reference, row["prompt"] + reference) or row["entry_point"] != task.entrypoint:
        raise RuntimeError(f"MBPP+ reference mismatch for {task.id}")
    if task.question != row["prompt"] or task.inputs != inputs:
        raise RuntimeError(f"MBPP+ frozen inputs or prompt mismatch for {task.id}")
    return {"base_count": len(row["base_input"]), "atol": row["atol"], "reference": task.reference,
            "expected": task.expected}
