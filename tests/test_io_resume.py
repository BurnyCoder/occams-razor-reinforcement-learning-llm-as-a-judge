"""Protect Unicode benchmark records and retries that have no learning history."""

import json

import pytest

from occam_rl.io import read_jsonl
from occam_rl import training


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_jsonl_preserves_unicode_inside_strings(tmp_path, separator):
    """JSON Lines uses LF records; Unicode separators remain valid string data."""
    path = tmp_path / "tasks.jsonl"
    record = {"question": "first" + separator + "second"}
    path.write_bytes((json.dumps(record, ensure_ascii=False) + "\r\n").encode())
    assert read_jsonl(path) == [record]


def test_only_unterminated_final_record_can_be_ignored(tmp_path):
    """Live logging may end mid-write; complete corrupt records must still raise."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"event":"ok"}\n{"event":', encoding="utf-8")
    assert read_jsonl(path, tolerate_tail=True) == [{"event": "ok"}]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
    path.write_text('{"event":"ok"}\n{"event":\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, tolerate_tail=True)


@pytest.mark.parametrize("event", ["phase_failed", "llm_prompt", "rollout", "group", "optimizer_step"])
def test_missing_checkpoint_retry_cannot_discard_rollouts(tmp_path, event):
    """Only a startup failure can restart from the seed without a full checkpoint."""
    (tmp_path / "events.jsonl").write_text(json.dumps({"event": event}) + "\n", encoding="utf-8")
    if event == "phase_failed":
        training.assert_unstarted_run(tmp_path)
    else:
        with pytest.raises(RuntimeError):
            training.assert_unstarted_run(tmp_path)
