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
