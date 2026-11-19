"""Independent counterexamples to completeness and evidence-label claims."""

import json
from types import SimpleNamespace

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
                 tokens=10, formal_length=2, seconds=1, truncated=False) for t in tasks]
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
