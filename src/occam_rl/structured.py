"""Use an existing JSON grammar; Pydantic still validates judgment semantics.

Source/API: https://github.com/noamgat/lm-format-enforcer (MIT).
Grammar constrains spelling/types, not factual accuracy or reasoning adequacy.
"""

from .schema import Judgment


def judgment_schema() -> dict:
    """Explicit enums enforce the ordinal range on parsers lacking numeric bounds."""
    schema = Judgment.model_json_schema()
    for name in ("assumptions", "concepts", "steps", "clarity"):
        schema["properties"][name]["enum"] = [0, 1, 2, 3, 4]
    return schema


def judgment_prefix(tokenizer_data):
    """Create fresh per-response parser state while reusing the tokenizer trie."""
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(judgment_schema()))
