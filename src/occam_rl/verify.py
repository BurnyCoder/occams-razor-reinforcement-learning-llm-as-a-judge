"""Strict final-answer verification and isolated execution of generated Python.

No generated program executes in the host interpreter. Docker isolation follows
https://docs.docker.com/engine/containers/run/ and is defense in depth, not a proof
against container-runtime vulnerabilities. Test expectations stay outside the container.
"""

import json
import math
import re
import subprocess
import uuid
from fractions import Fraction

import pint

from .rewards import regions
from .schema import Task, Verification

UNITS = pint.UnitRegistry()
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# The runner reads only candidate code, function arguments, or benchmark assertions.
# Resource limits and the read-only filesystem are imposed by Docker, outside Python.
RUNNER = '''import contextlib, io, json, sys
p = json.load(sys.stdin)
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(p["code"], "candidate", "exec"), ns)
    if p.get("tests"):
        for test in p["tests"]:
            exec(compile(test, "test", "exec"), ns)
        result = True
    else:
        result = [ns[p["entrypoint"]](*args) for args in p["inputs"]]
print(json.dumps(result, allow_nan=False))
'''


def numeric(answer: str) -> tuple[Fraction, str]:
    """Accept only a finite numeric literal or ratio followed by a unit string."""
    cleaned = answer.strip().replace(",", "").replace("−", "-")
    if len(cleaned) > 256:
        raise ValueError("Answer too long")
    match = re.fullmatch(fr"({NUMBER})(?:\s*/\s*({NUMBER}))?\s*(.*)", cleaned)
    if not match:
        raise ValueError("Expected a numeric final answer")
    value = Fraction(match[1])
    if match[2]:
        value /= Fraction(match[2])
    if not math.isfinite(float(value)):
        raise ValueError("Non-finite answer")
    return value, match[3].strip()


def run_code(code: str, *, entrypoint: str = "", inputs: list | None = None,
             tests: list[str] | None = None, image: str = "occam-verifier:1", timeout: int = 8):
    """Run under a nonroot user, no network/mounts/capabilities, bounded resources."""
    name = "occam-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "32",
               "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--user", "65534:65534",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=8m", "-i", image, "python", "-I", "-c", RUNNER]
    payload = json.dumps({"code": code, "entrypoint": entrypoint, "inputs": inputs or [], "tests": tests or []})
    try:
        result = subprocess.run(command, input=payload, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # docker CLI termination alone does not guarantee container termination.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        raise ValueError("Code exceeded wall-clock limit") from None
    if result.returncode:
        # Infrastructure failures must not be misclassified as wrong model answers.
        if "Cannot connect" in result.stderr or "Unable to find image" in result.stderr:
            raise RuntimeError(result.stderr)
        raise ValueError("Code failed: " + result.stderr[-2000:])
    if len(result.stdout) > 1_000_000:
        raise ValueError("Code output too large")
    return json.loads(result.stdout)


def verify(task: Task, trace: str, image: str = "occam-verifier:1") -> Verification:
    """Correctness cannot be supplied by the LLM judge or a number in the reasoning."""
    extracted = ""
    try:
        if task.domain == "code":
            blocks = regions(trace, "code")
            if len(blocks) != 1:
                return Verification(correct=False, reason="Expected exactly one code submission")
            extracted = blocks[0]
            result = run_code(extracted, entrypoint=task.entrypoint, inputs=task.inputs, tests=task.tests, image=image)
            correct = result is True if task.tests else result == task.expected
            return Verification(correct=correct, reason="tests_passed" if correct else "hidden_case_failed",
                                extracted=extracted)
        answers = regions(trace, "answer")
        if len(answers) != 1:
            return Verification(correct=False, reason="Expected exactly one final answer")
        extracted = answers[0].strip()
        value, unit = numeric(extracted)
        expected, _ = numeric(task.answer)
        if task.unit:
            if not unit:
                return Verification(correct=False, reason="Missing units", extracted=extracted)
            converted = UNITS.Quantity(float(value), unit).to(task.unit).magnitude
            correct = math.isclose(converted, float(expected), rel_tol=task.rtol, abs_tol=task.atol)
        else:
            correct = not unit and value == expected
        return Verification(correct=bool(correct), reason="matched" if correct else "mismatch", extracted=extracted)
    except (ValueError, ZeroDivisionError, pint.errors.PintError, json.JSONDecodeError) as exc:
        return Verification(correct=False, reason=str(exc), extracted=extracted)
