"""Counterexamples and boundary decisions for the prospective scoring amendment."""

import pytest

from occam_rl.io import config, data_dir, read_jsonl
from occam_rl.schema import Task
from occam_rl.verify import verify


def frozen_task(identifier):
    """Read the recorded task without rebuilding a dataset or loading a model."""
    config()
    return next(Task.model_validate(t) for t in read_jsonl(data_dir() / "test.jsonl") if t["id"] == identifier)


def test_dimensional_work_accepts_equivalent_units():
    task = frozen_task("ugphysics/ClassicalElectromagnetism/298")
    assert task.unit == ""
    for text in ("0.09 J", "90 mJ"):
        assert verify(task, f"<answer>{text}</answer>").correct
    for text in ("0.09 A", "0.09"):
        assert not verify(task, f"<answer>{text}</answer>").correct


@pytest.mark.sandbox
def test_mbpp_order_uses_set_equivalence():
    task = frozen_task("Mbpp/111")
    program = "def common_in_nested_lists(xs): return sorted(set.intersection(*map(set, xs)))"
    assert verify(task, f"<code>{program}</code>").correct
    assert not verify(task, "<code>def common_in_nested_lists(xs): return []</code>").correct


def test_hypothesis_boundaries_and_negative_result():
    from occam_rl.inference import interval_decision
    assert interval_decision([-.02, .10], -.02) == "inconclusive"
    assert interval_decision([-.10, -.021], -.02) == "adverse"
    assert interval_decision([-.019, .01], -.02) == "supported"
    assert interval_decision([0, 0], 0) == "inconclusive"


def test_unobserved_hypotheses_reported_individually(tmp_path, monkeypatch):
    from occam_rl import reporting
    monkeypatch.setattr(reporting, "runs_dir", lambda: tmp_path)
    result = reporting.comparisons(config())
    assert {k: v["status"] for k, v in result["hypotheses"].items()} == dict.fromkeys(("H1", "H2", "H3"), "untested")
    assert set(result["hypotheses"]["H3"]["comparisons"]) == {"prompt_ordinary", "prompt_concise", "prompt_occam"}


def test_full_rubric_is_visible_without_judge_output_instructions():
    from occam_rl.prompts import JUDGE, policy_messages
    instruction = policy_messages(frozen_task("Mbpp/111"), "prompt_occam")[0]["content"]
    for line in JUDGE.splitlines():
        if " rates " in line:
            assert line in instruction
    assert "Return ONLY valid JSON" not in instruction
