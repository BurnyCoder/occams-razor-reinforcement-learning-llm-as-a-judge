"""Run the pinned upstream comparator on native Python values inside the CPU sandbox."""

import contextlib
import io
import json
import sys

from evalplus.eval import PASS, is_floats, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec


def evaluate(payload, diagnostics=False):
    """Reconstruct typed expectations before candidate execution, then check base and plus."""
    spec = payload["mbpp"]
    inputs = payload["inputs"]
    entry = payload["entrypoint"]
    accepted = True
    kinds, floats = set(), 0
    split = spec["base_count"]
    for group in (slice(0, split), slice(split, None)):
        if not inputs[group]:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            expected,durations = trusted_exec(spec["reference"],inputs[group],entry,record_time=True)
        if json.loads(json.dumps(expected,allow_nan=False)) != spec["expected"][group]:
            raise RuntimeError("Reference output disagrees with frozen JSON")
        kinds.update(type(x).__name__ for x in expected)
        floats += sum(is_floats(x) for x in expected)
        if entry in MBPP_OUTPUT_NOT_NONE_TASKS:
            expected = [x is not None for x in expected]
        status,_ = untrusted_check("mbpp",payload["code"],inputs[group],entry,expected,spec["atol"],
                                   durations,fast_check=True)
        accepted &= status == PASS
        if not accepted and not diagnostics:
            break
    if diagnostics:
        return {"canonical_pass":accepted,"native_reference_types":sorted(kinds),"float_reference_cases":floats}
    return accepted


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
