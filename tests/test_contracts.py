"""Counterexamples target reward hacking and incorrect reasoning sufficiency."""

import pytest
from pydantic import ValidationError

from occam_rl.prompts import policy_messages
from occam_rl.rewards import formal_length, reward
from occam_rl.schema import Judgment, Task
from occam_rl.verify import numeric, verify


def judgment(adequate=True):
    """Use an explicit constructed label; this is not an LLM-produced result."""
    return Judgment(adequate=adequate, assumptions=4, concepts=4, steps=4,
                    clarity=4, missing=[] if adequate else ["derivation"], rationale="fixture")


def task(**updates):
    """Small exact reference shared by pure verifier tests."""
    fields = dict(id="fixture", domain="math", family="fixture", split="dev",
                  question="Solve 2x+1=7.", answer="3", reference="2x=6, x=3.")
    return Task(**(fields | updates))


@pytest.mark.parametrize("arm", ["occam", "token", "formal", "shuffled"])
def test_missing_reasoning_never_gets_bonus(arm):
    """Correct final answer alone cannot exploit any shaping arm."""
    assert reward(True, judgment(False), arm, 1, 1) == 1
    assert reward(False, judgment(), arm, 1, 1) == 0


def test_correctness_separation_and_length_controls():
    """Worst correct reward exceeds best wrong reward, independent of length."""
    assert reward(True, judgment(), "occam", 10000, 10000) == 1.2
    assert reward(True, None, "accuracy", 10, 10) == 1
    assert reward(True, judgment(), "token", 100, 100, {"token": 100}) == 1.1
    with pytest.raises(ValueError):
        reward(True, judgment(), "token", 10, 10)


def test_judge_schema_fails_closed():
    """String booleans, score coercion and contradictory adequacy are rejected."""
    raw = judgment().model_dump()
    for changed in ({"adequate": "true"}, {"steps": True}, {"steps": 5}, {"missing": ["units"]}):
        with pytest.raises(ValidationError):
            Judgment.model_validate(raw | changed)


def test_only_declared_final_answer_is_verified():
    """Mentioning the correct value in reasoning must not rescue a wrong final answer."""
    assert not verify(task(), "Maybe 3; <answer>4</answer>").correct
    assert not verify(task(), "3").correct
    assert not verify(task(), "<answer>3</answer><answer>4</answer>").correct
    assert verify(task(), "<answer>3.0</answer>").correct
    assert not verify(task(), "<answer>nan</answer>").correct


def test_unit_conversion_and_exact_rationals():
    """Different valid units agree; omission and dimension mismatch fail."""
    physical = task(domain="physics", answer="3", unit="m/s", rtol=1e-6)
    assert verify(physical, "<answer>300 cm/s</answer>").correct
    assert not verify(physical, "<answer>3</answer>").correct
    assert not verify(physical, "<answer>3 kg</answer>").correct
    assert verify(task(domain="physics", answer="0.333333", rtol=1e-3), "<answer>1/3</answer>").correct
    assert numeric("1/3")[0] != numeric("0.333")[0]
    with pytest.raises(ValueError):
        numeric("__import__('os').system('echo bad')")


@pytest.mark.parametrize("answer", [r"3 m\s", "3 (m/s", "1e309 m/s", "1e999999999999 m/s"])
def test_malformed_quantity_is_wrong_without_crashing(answer):
    """Unsupported unit syntax and overflow are candidate errors, not run failures."""
    physical = task(domain="physics", answer="3", unit="m/s", rtol=1e-6)
    assert not verify(physical, f"<answer>{answer}</answer>").correct


def test_length_counts_comment_markers_in_strings():
    """Comment stripping must not erase literal program semantics."""
    assert formal_length('<code>x = "#" # comment\n</code>') == 5
    assert formal_length("$x + 2 = 3$") == 5
    assert formal_length("There is no formal expression.") == 0


def test_prompt_baselines_change_only_instructions():
    """No reference answer or trained weights enter the prompt-only treatment."""
    ordinary = policy_messages(task(), "prompt_ordinary")
    concise = policy_messages(task(), "prompt_concise")
    occam = policy_messages(task(), "prompt_occam")
    assert ordinary[1] == concise[1] == occam[1]
    assert ordinary[0] != concise[0] != occam[0]
    assert "omit necessary reasoning" in occam[0]["content"]
