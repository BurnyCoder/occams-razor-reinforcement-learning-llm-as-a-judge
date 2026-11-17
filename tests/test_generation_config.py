"""A chat terminator and a model end-of-text token may be different IDs."""

from types import SimpleNamespace

from occam_rl.models import generation_eos_ids


def test_both_declared_end_tokens_stop_generation():
    """Reproduce the pinned Qwen tokenizer/model mismatch without loading weights."""
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
    tokenizer = SimpleNamespace(eos_token_id=248046)
    assert generation_eos_ids(model, tokenizer) == [248044, 248046]
    model.generation_config.eos_token_id = [248044, 248046]
    assert generation_eos_ids(model, tokenizer) == [248044, 248046]
