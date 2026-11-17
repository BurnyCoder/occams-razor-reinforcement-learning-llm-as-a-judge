"""Exercise observed malformed keys and preserve semantic fail-closed validation."""

import json

import pytest
from lmformatenforcer import JsonSchemaParser
from pydantic import ValidationError

from occam_rl.schema import Judgment
from occam_rl.structured import judgment_schema


def accepted(text: str) -> bool:
    """Replay characters through the public parser without running a model."""
    parser = JsonSchemaParser(judgment_schema())
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def test_decoder_blocks_observed_key_typo_and_out_of_range_scores():
    """The actual audit typo and illegal ordinal scores must be unreachable."""
    valid = dict(rationale="A complete direct derivation.", missing=[], adequate=True,
                 assumptions=4, concepts=4, steps=4, clarity=4)
    assert accepted(json.dumps(valid))
    assert not accepted(json.dumps(valid).replace('"rationale"', '"rationle"'))
    assert not accepted(json.dumps(valid | {"steps": 5}))
    assert not accepted(json.dumps(valid | {"adequate": "true"}))


def test_syntax_enforcement_does_not_replace_semantic_validation():
    """A grammatical contradiction remains invalid and cannot earn a bonus."""
    contradictory = dict(rationale="Missing derivation.", missing=["derivation"],
                         adequate=True, assumptions=4, concepts=4, steps=4, clarity=4)
    raw = json.dumps(contradictory)
    assert accepted(raw)
    with pytest.raises(ValidationError):
        Judgment.model_validate_json(raw)
