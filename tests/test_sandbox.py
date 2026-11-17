"""Optional end-to-end checks execute only inside the Docker verifier."""

import pytest

from occam_rl.schema import Task
from occam_rl.verify import verify


@pytest.mark.sandbox
def test_hidden_edge_cases_and_timeout():
    """A plausible example solution must fail hidden empty-input behavior."""
    task = Task(id="sandbox", domain="code", family="fixture", split="dev", question="Sum xs",
                reference="def solve(xs): return sum(xs)", entrypoint="solve",
                inputs=[[[]], [[2, -2]], [[1, 2, 3]]], expected=[0, 0, 6])
    assert verify(task, "<code>def solve(xs): return sum(xs)</code>").correct
    assert not verify(task, "<code>def solve(xs): return xs[0]</code>").correct
    assert not verify(task, "<code>def solve(xs):\n while True: pass</code>").correct


@pytest.mark.sandbox
def test_no_mutation_is_part_of_the_function_contract():
    """Correct output does not excuse violating an explicitly immutable input contract."""
    task = Task(id="mutation", domain="code", family="fixture", split="dev", question="Sum without mutation",
                reference="def solve(xs): return sum(xs)", entrypoint="solve", preserve_inputs=True,
                inputs=[[[3, 1, 2]]], expected=[6])
    assert verify(task, "<code>def solve(xs): return sum(xs)</code>").correct
    assert not verify(task, "<code>def solve(xs):\n xs.sort()\n return sum(xs)</code>").correct
