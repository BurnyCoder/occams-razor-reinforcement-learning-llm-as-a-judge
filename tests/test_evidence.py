"""Independent counterexamples to completeness and evidence-label claims."""

import json
import pytest

from occam_rl import reporting
from occam_rl.io import write_json
from occam_rl.statistics import paired_effect, wilson


def evaluation_fixture(tmp_path, monkeypatch, mutation=None):
    """Create a complete declared population, then corrupt only the tested property."""
    tasks = [dict(id=d, domain=d, family=d, split="test") for d in ("math", "physics", "code")]
    frozen = dict(files={"test.jsonl": dict(sha256="a" * 64, count=3)}, tasks=tasks)
    write_json(tmp_path / "manifests/data.json", frozen)
    manifest = dict(arm="occam", seed=17, split="test", tasks=[t["id"] for t in tasks],
                    limit_per_domain=None, config_hash="b" * 64, data=frozen["files"]["test.jsonl"],
                    git_commit="c" * 40, provenance=dict(policy_revision="d" * 40,
                    judge_hash="e" * 64, adapter_sha256="f" * 64))
    rows = [dict(task_id=t["id"], domain=t["domain"], family=t["family"], arm="occam", seed=17,
                 git_commit="c" * 40, correct=1, adequate=1, simplicity=1, quality=1,
                 tokens=10, formal_length=2, seconds=1, truncated=False, judge_invalid=False,
                 judgment=dict(adequate=True, assumptions=4, concepts=4, steps=4, clarity=4,
                               missing=[], rationale="Fixture evidence")) for t in tasks]
    if mutation:
        mutation(manifest, rows)
    directory = tmp_path / "runs/occam-seed17/eval-test"
    write_json(directory / "manifest.json", manifest)
    (directory / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    monkeypatch.setattr(reporting, "runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(reporting, "collect", lambda: {})
    return manifest, rows


@pytest.mark.parametrize("mutation", [
    lambda m, r: r.__setitem__(2, dict(r[0])),
    lambda m, r: r[2].update(task_id="substituted"),
    lambda m, r: r[2].update(seed=29),
    lambda m, r: r[2].update(family="different"),
    lambda m, r: m.pop("provenance"),
    lambda m, r: m["data"].update(sha256="0" * 64),
    lambda m, r: r[2]["judgment"].update(missing=["essential premise"]),
    lambda m, r: r[2].pop("judgment"),
    lambda m, r: r[2].update(quality=.5),
])
def test_equal_row_count_cannot_certify_evaluation(tmp_path, monkeypatch, mutation):
    """The same number of rows is not proof of population or provenance identity."""
    evaluation_fixture(tmp_path, monkeypatch, mutation)
    rows = reporting.snapshot()["evaluations"]
    assert rows and not any(r["complete"] for r in rows)
    assert all(r["evidence_issues"] for r in rows)


def test_complete_population_is_recognized(tmp_path, monkeypatch):
    """Validation must accept genuinely complete fixtures, not reject everything."""
    evaluation_fixture(tmp_path, monkeypatch)
    assert all(r["complete"] for r in reporting.snapshot()["evaluations"])


def test_missing_domain_is_not_a_three_domain_macro_average(tmp_path, monkeypatch):
    """Partial domains remain visible, but the registered overall estimand is missing."""
    evaluation_fixture(tmp_path, monkeypatch, lambda m, r: r.pop())
    rows = reporting.snapshot()["evaluations"]
    assert {r["domain"] for r in rows} == {"math", "physics"}
    table = reporting.score_table(rows)
    assert "test / math" in table and "partial" in table and "not evaluated" not in table


def test_explicit_failed_judgment_remains_zero_scored_evidence(tmp_path, monkeypatch):
    """Preserve the registered failure policy, rather than silently dropping failures."""
    evaluation_fixture(tmp_path, monkeypatch, lambda m, r: r[2].update(
        judgment=None, judge_invalid=True, adequate=0, simplicity=0, quality=0))
    assert all(r["complete"] for r in reporting.snapshot()["evaluations"])


def test_partial_status_is_not_hidden_by_exploratory_flag():
    """Both qualifications must survive presentation."""
    row = dict(domain="all", run="r", split="test", n=1, correct=0, adequate=0,
               quality=0, tokens=10, complete=False, exploratory=True, evidence_issues=["missing tasks"])
    assert "partial" in reporting.score_table([row])


def test_distinct_tasks_and_verifier_rejections_are_reported(tmp_path, monkeypatch):
    """Two completions of one question are not two independent tasks."""
    directory = tmp_path / "run"
    directory.mkdir()
    row = dict(event="rollout", time="now", step=0, task_id="a", domain="math", correct=0,
               reward=0, tokens=10, truncated=True, verifier=dict(reason="Expected exactly one final answer"))
    (directory / "events.jsonl").write_text((json.dumps(row) + "\n") * 2)
    monkeypatch.setattr(reporting, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(reporting, "collect", lambda: {})
    run = reporting.snapshot()["runs"][0]
    assert run["unique_tasks"] == 1
    assert run["verifier_failure_counts"] == {"Expected exactly one final answer": 2}


def test_family_metadata_must_be_identical_across_seeds():
    """A seed-dependent cluster definition changes the registered bootstrap unit."""
    rows = [dict(seed=s, task_id="a", domain="math", family=str(s), correct=1) for s in (17, 29)]
    with pytest.raises(ValueError, match="metadata"):
        paired_effect(rows, rows, "correct", replicates=20)


def test_wilson_boundary_and_known_value():
    """Published score-interval formula: all-success upper limit cannot exceed one."""
    assert 0 <= wilson(0, 10)[0] <= wilson(0, 10)[1] <= 1
    assert 0 <= wilson(10, 10)[0] <= wilson(10, 10)[1] <= 1
    assert wilson(5, 10) == pytest.approx([0.2365930905, 0.7634069095])


def test_recount_rejects_invented_judge_summary():
    """A contradictory JSON object must not masquerade as a valid omission rejection."""
    from occam_rl.readiness import recount_audit
    row = dict(rows=[dict(id="audit/omission/0", judgment=None)], wins=0, pair_count=0,
               rejected=1, omission_count=1, invalid=1, passed=False)
    with pytest.raises(ValueError, match="disagrees"):
        recount_audit(row)


def test_missing_evidence_blocks_primary_inference(tmp_path, monkeypatch):
    """Even favorable mocked statistics cannot bypass invalid evidence."""
    evaluation_fixture(tmp_path, monkeypatch, lambda m, r: m.pop("provenance"))
    monkeypatch.setattr(reporting, "paired_effect", lambda *a, **k: pytest.fail("Invalid input reached inference"))
    assert reporting.comparisons({})["status"] == "blocked_invalid_evidence"


@pytest.mark.parametrize("changes", [{"seed": 29}, {"pilot": True}, {"pilot": None}])
def test_first_occam_receipt_must_be_retained_seed17(tmp_path, monkeypatch, changes):
    """A complete pilot or another seed cannot unlock standalone baselines."""
    from occam_rl import publication
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    receipt = dict(complete=True, hf_commit="receipt", arm="occam", seed=17, pilot=False)
    write_json(tmp_path / "manifests/first-occam.json", receipt | changes)
    with pytest.raises(RuntimeError):
        publication.require_first_occam()
    write_json(tmp_path / "manifests/first-occam.json", receipt)
    publication.require_first_occam()


def test_registered_algebra_with_independent_fractions():
    """Explicit examples fix expectations independently of implementation arithmetic."""
    from fractions import Fraction
    from occam_rl.rewards import reward
    from occam_rl.schema import Judgment
    j = Judgment(adequate=True, assumptions=1, concepts=2, steps=3, clarity=4, missing=[], rationale="fixture")
    assert j.simplicity == Fraction(5, 8)
    assert reward(True, j, "occam", 9, 9) == float(Fraction(9, 8))
    assert reward(True, j, "formal", 99, 3, {"formal": 2}) == pytest.approx(float(Fraction(27, 25)))


def test_domain_macro_mean_is_not_a_pooled_task_mean():
    """Unequal domain sizes must not silently change the registered estimand."""
    left = [dict(task_id=str(i), seed=17, domain=d, family=str(i), quality=q)
            for i, (d, q) in enumerate([("math", 1), ("math", 1), ("physics", 0), ("code", 0)])]
    right = [r | {"quality": 0} for r in left]
    result = paired_effect(left, right, "quality", replicates=30, seed=7)
    assert result["effect"] == pytest.approx(1/3)
    assert result["ci95"] == pytest.approx([1/3, 1/3])


def test_conciserl_primitive_root_counterexample():
    """Exact polynomial division verifies the literature example, without a model."""
    from sympy import Symbol, rem
    z = Symbol("z")
    assert rem(z**4 + z**2 + 1, z**2 + z + 1, z) == 0
    assert rem(z**3 - 1, z**2 + z + 1, z) == 0
    assert rem(z - 1, z**2 + z + 1, z) != 0
def test_junit_counts_direct_cases_and_nested_suites(tmp_path, monkeypatch):
    from occam_rl import factcheck
    monkeypatch.setattr(factcheck, "ROOT", tmp_path)
    folder = tmp_path / "results/tests"
    folder.mkdir(parents=True)
    (folder / "independent-node.xml").write_text('<testsuites><testcase/><testcase><failure/></testcase></testsuites>')
    (folder / "independent-python.xml").write_text('<testsuites><testsuite><testcase/><testcase><skipped/></testcase></testsuite></testsuites>')
    a, b = factcheck.test_records()
    assert (a["tests"], a["failures"], a["skipped"]) == (2, 1, 0)
    assert (b["tests"], b["failures"], b["skipped"]) == (2, 0, 1)

