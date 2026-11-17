"""Validated artifact contracts; scores fail closed instead of being coerced.

Pydantic strict fields prevent a judge's strings or booleans becoming numeric scores.
Reference: https://docs.pydantic.dev/latest/concepts/strict_mode/
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

DOMAINS = ("math", "physics", "code")
TRAINED_ARMS = ("occam", "accuracy", "token", "formal", "shuffled")
PROMPT_ARMS = ("prompt_ordinary", "prompt_concise", "prompt_occam")


class Task(BaseModel):
    """A task keeps verifier information separate from the model-visible question."""

    model_config = ConfigDict(extra="forbid")
    id: str
    domain: Literal["math", "physics", "code"]
    family: str
    split: str
    question: str
    answer: str = ""
    unit: str = ""
    reference: str
    requirements: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    entrypoint: str = ""
    inputs: list = Field(default_factory=list)
    expected: list = Field(default_factory=list)
    preserve_inputs: bool = False
    source: str = "procedural-v1"
    license: str = "Apache-2.0"
    rtol: float = 0.0
    atol: float = 0.0


class Judgment(BaseModel):
    """A validated assessment of reasoning, not an override of correctness."""

    model_config = ConfigDict(extra="forbid")
    adequate: StrictBool
    assumptions: StrictInt = Field(ge=0, le=4)
    concepts: StrictInt = Field(ge=0, le=4)
    steps: StrictInt = Field(ge=0, le=4)
    clarity: StrictInt = Field(ge=0, le=4)
    missing: list[str]
    rationale: str

    @model_validator(mode="after")
    def consistent_gate(self):
        """Missing essential obligations are incompatible with an adequate verdict."""
        if self.adequate and self.missing:
            raise ValueError("Adequate judgment lists missing necessary reasoning")
        return self

    @property
    def simplicity(self) -> float:
        """Four equal-weight ordinal dimensions, normalized as preregistered."""
        return (self.assumptions + self.concepts + self.steps + self.clarity) / 16


class Verification(BaseModel):
    """Deterministic correctness plus diagnostic evidence."""

    correct: bool
    reason: str
    extracted: str = ""
