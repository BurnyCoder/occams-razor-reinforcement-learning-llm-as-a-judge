"""Protect prompt-only publication and the distinction between pilot and full evidence."""

from types import SimpleNamespace

import pytest

from occam_rl.io import read_json, write_json
from occam_rl import publishing, reporting


def test_source_upload_uses_exact_tracked_paths(tmp_path, monkeypatch):
    """Untracked files and the local environment are never traversed for source upload."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("project")
    (tmp_path / "docs/hf-artifacts.md").write_text("artifact card")
    (tmp_path / ".env").write_text("must remain local")
    calls = []
    class API:
        def create_repo(self, *args, **kwargs):
            pass
        def create_commit(self, **kwargs):
            calls.extend(op.path_in_repo for op in kwargs["operations"])
            return SimpleNamespace(oid="snapshot")
        def add_collection_item(self, *args, **kwargs):
            pass
    monkeypatch.setattr(publishing, "ROOT", tmp_path)
    monkeypatch.setattr(publishing, "git", lambda *a: "README.md\ndocs/hf-artifacts.md")
    monkeypatch.setattr(publishing, "runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(publishing, "hub", lambda cfg: (API(), "collection"))
    publishing.publish_source({"hf_artifacts_repo": "owner/artifacts"}, SimpleNamespace(emit=lambda *a, **k: None), "commit")
    assert set(calls) == {"research-README.md", "docs/hf-artifacts.md", "README.md"}
    assert read_json(tmp_path / ".local/hf-publication.json")["git_commit"] == "commit"


def test_prompt_only_publication_has_no_model_repository(tmp_path, monkeypatch):
    """An unchanged-model evaluation publishes records, never a fictional adapter."""
    name = "prompt_concise-seed17"
    run = tmp_path / "runs" / name
    write_json(run / "eval-test/manifest.json", {"arm": "prompt_concise", "seed": 17, "tasks": ["a"]})
    (run / "eval-test/results.jsonl").write_text('{"task_id":"a"}\n')
    calls = []
    class API:
        def create_repo(self, repo, **kwargs):
            calls.append(("create", kwargs["repo_type"]))
        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs["repo_type"]))
            return SimpleNamespace(oid="published-evaluation")
        def add_collection_item(self, *args, **kwargs):
            pass
    monkeypatch.setattr(publishing, "ROOT", tmp_path)
    monkeypatch.setattr(publishing, "runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(publishing, "hub", lambda cfg: (API(), "collection"))
    publishing.publish_run({"hf_artifacts_repo": "owner/artifacts"}, SimpleNamespace(emit=lambda *a, **k: None), "commit", name)
    receipt = read_json(tmp_path / "manifests" / f"published-{name}.json")
    assert not receipt["complete"] and receipt["artifact_type"] == "evaluation"
    assert receipt["evidence_issues"]  # Legacy records remain publishable, not certified.
    assert all(kind == "dataset" for _, kind in calls)
    assert not (tmp_path / "manifests/first-occam.json").exists()


@pytest.mark.parametrize("seeds,override,expected", [
    ([17, 29, 43], False, "exploratory_incomplete_seeds"),
    ([17], True, "exploratory_judge_override"),
])
def test_exploratory_evidence_cannot_satisfy_confirmatory_claim(tmp_path, monkeypatch, seeds, override, expected):
    """Favorable intervals cannot erase incomplete seeds or a judge-quality waiver."""
    import json
    from test_evidence import evaluation_fixture
    manifest, rows = evaluation_fixture(tmp_path, monkeypatch)
    for arm in ("occam", "accuracy", "token", "formal"):
        directory = tmp_path / f"runs/{arm}-seed17/eval-test"
        write_json(directory / "manifest.json", manifest | {"arm": arm, "exploratory_judge_override": override})
        (directory / "results.jsonl").write_text("".join(json.dumps(r | {"arm": arm}) + "\n" for r in rows))
    monkeypatch.setattr(reporting, "paired_effect", lambda *a, **k: {"effect": .1, "ci95": [.05, .15], "p_one_sided": .001})
    result = reporting.comparisons({"seeds": seeds, "bootstrap_replicates": 10, "bootstrap_seed": 1})
    assert result["status"] == expected
    assert result["confirmatory_seed_target_met"] == (seeds == [17])


@pytest.mark.parametrize("finished,changed", [(False, "—"), (True, "0")])
def test_pilot_report_distinguishes_no_learning_from_missing_state(tmp_path, monkeypatch, finished, changed):
    """A completed zero-reward pilot is evidence; an unsaved adapter comparison is missing."""
    import json
    directory = tmp_path / "occam-seed17-pilot"
    directory.mkdir()
    events = [
        {"event": "rollout", "time": "now", "step": 0, "correct": 0, "reward": 0,
         "tokens": 2048, "truncated": True},
        {"event": "group", "time": "now", "step": 0, "reward_spread": 0},
        {"event": "training_metrics", "time": "now", "step": 1, "loss": 0},
    ]
    (directory / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    if finished:
        write_json(directory / "status.json", {"steps": 1, "complete": True, "pilot": True,
                                                "changed_parameter_tensors": 0})
    monkeypatch.setattr(reporting, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(reporting, "collect", lambda: {})
    data = reporting.snapshot()
    assert data["evaluations"] == []
    state = "complete pilot" if finished else "in progress"
    assert f"| occam-seed17-pilot | 1 | 0 / 1 | 1 | 0 / 1 | {changed} | {state} |" in reporting.training_table(data["runs"])
