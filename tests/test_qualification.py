"""Qualification must belong to the actual model, prompt, and locked audit in use."""

import pytest

from occam_rl import training
from occam_rl.io import digest
from occam_rl.prompts import JUDGE


def test_another_models_qualification_cannot_authorize_training(monkeypatch):
    """A valid old judge must not silently authorize a newly selected judge."""
    previous = dict(passed=True, model="old-judge", revision="old-revision",
                    prompt_hash=digest(JUDGE), audit_hash=digest({"audit": "fixture"}))
    monkeypatch.setattr(training, "read_json", lambda path: previous)
    monkeypatch.setattr("occam_rl.models.revision", lambda name: "old-revision")
    with pytest.raises(RuntimeError, match="not qualified"):
        training.require_qualification("qwen", {"judge": "new-judge"})


def test_explicit_training_override_retains_failed_audit_and_identity_checks(monkeypatch):
    """User-authorized noisy training cannot authorize a different or unaudited judge."""
    audit = {"fixture": True}
    failed = dict(passed=False, model="judge", revision="revision",
                  prompt_hash=digest(JUDGE), audit_hash=digest(audit))
    monkeypatch.setattr(training, "read_json", lambda path: audit if path.name == "judge-audit.json" else failed)
    monkeypatch.setattr("occam_rl.models.revision", lambda name: "revision")
    cfg = {"judge": "judge"}
    with pytest.raises(RuntimeError):
        training.require_qualification("qwen", cfg)
    assert training.require_qualification("qwen", cfg, allow_unqualified=True) is failed
    with pytest.raises(RuntimeError):
        training.require_qualification("gemma", cfg, allow_unqualified=True)
    failed["prompt_hash"] = "stale"
    with pytest.raises(RuntimeError):
        training.require_qualification("qwen", cfg, allow_unqualified=True)
