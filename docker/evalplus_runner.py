"""Run the pinned upstream comparator on native Python values inside the CPU sandbox."""

import contextlib
import copy
import io
import json
import sys
import time

from evalplus.eval import PASS, untrusted_check


def evaluate(payload):
    """Reconstruct typed expectations before candidate execution, then check base and plus."""
    spec = payload["mbpp"]
    inputs = payload["inputs"]
    entry = payload["entrypoint"]
    expected, durations = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = {}
        exec(compile(spec["reference"], "reference", "exec"), namespace)
        for args in copy.deepcopy(inputs):
            start = time.monotonic()
            expected.append(namespace[entry](*args))
            durations.append(time.monotonic() - start)
    if json.loads(json.dumps(expected, allow_nan=False)) != spec["expected"]:
        raise RuntimeError("Reference output disagrees with frozen JSON")
    split = spec["base_count"]
    for group in (slice(0, split), slice(split, None)):
        if inputs[group]:
            status, _ = untrusted_check("mbpp", payload["code"], inputs[group], entry,
                                        expected[group], spec["atol"], durations[group], fast_check=True)
            if status != PASS:
                return False
    return True


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
