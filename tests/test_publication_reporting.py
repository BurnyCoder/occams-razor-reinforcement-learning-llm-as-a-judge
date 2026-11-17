"""Protect prompt-only publication and the distinction between pilot and full evidence."""

from types import SimpleNamespace

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
    assert receipt["complete"] and receipt["artifact_type"] == "evaluation"
    assert all(kind == "dataset" for _, kind in calls)
    assert not (tmp_path / "manifests/first-occam.json").exists()


def test_one_seed_cannot_satisfy_confirmatory_claim(tmp_path, monkeypatch):
    """Even favorable intervals cannot replace the registered seed requirement."""
    for arm in ("occam", "accuracy", "token", "formal"):
        directory = tmp_path / f"{arm}-seed17/eval-test"
        write_json(directory / "manifest.json", {"arm": arm, "limit_per_domain": None, "tasks": ["a"]})
        (directory / "results.jsonl").write_text('{"task_id":"a","seed":17,"domain":"math"}\n')
    monkeypatch.setattr(reporting, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(reporting, "paired_effect", lambda *a, **k: {"effect": .1, "ci95": [.05, .15], "p_one_sided": .001})
    result = reporting.comparisons({"seeds": [17, 29, 43], "bootstrap_replicates": 10, "bootstrap_seed": 1})
    assert result["status"] == "exploratory_incomplete_seeds"
    assert not result["confirmatory_seed_target_met"]
