"""Split leakage, pairing, and uncertainty tests use constructed data, not models."""

import pytest

from occam_rl.data import procedural_tasks, check_split_overlap
from occam_rl.statistics import paired_effect, holm


def test_procedural_reproducibility_and_disjoint_splits():
    """Seen-family evaluation has new instances; OOD has distinct families."""
    train = procedural_tasks("train", 12)
    test = procedural_tasks("test", 12)
    ood = procedural_tasks("ood", 12)
    assert [x.model_dump() for x in train] == [x.model_dump() for x in procedural_tasks("train", 12)]
    check_split_overlap(train + test + ood)
    assert not {x.family for x in train} & {x.family for x in ood}
    with pytest.raises(ValueError):
        check_split_overlap([train[0], train[0].model_copy(update={"split": "test"})])


def test_paired_bootstrap_requires_complete_pairing():
    """Missing tasks never disappear silently from a favorable comparison."""
    left = [{"task_id": "a", "seed": 17, "domain": "math", "family": "f", "correct": 1}]
    right = [dict(left[0], correct=0)]
    result = paired_effect(left, right, "correct", replicates=100, seed=1)
    assert result["effect"] == 1
    assert result["ci95"] == [1, 1]
    with pytest.raises(ValueError):
        paired_effect(left, [], "correct", replicates=100, seed=1)


def test_holm_adjustment_monotone_and_bounded():
    """Known simple example catches incorrect multiplicity scaling."""
    assert holm([0.01, 0.04]) == [0.02, 0.04]
    assert holm([0.9, 0.8]) == [1, 1]
