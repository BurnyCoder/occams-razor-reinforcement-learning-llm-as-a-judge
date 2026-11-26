"""Strict final-answer verification and isolated execution of generated Python.

No generated program executes in the host interpreter. Docker isolation follows
https://docs.docker.com/engine/containers/run/ and is defense in depth, not a proof
against container-runtime vulnerabilities. Procedural expected outputs stay on the
host; benchmark assertions (including their expectations) execute in the container.
Procedural cases and training assertions are withheld. Public MBPP+ prompts contain examples.
"""

import json
import math
import re
import subprocess
import tempfile
import uuid
from fractions import Fraction
from tokenize import TokenError

import pint

from .rewards import regions
from .schema import Task, Verification
from .scoring import EVAL_IMAGE, MBPP, UG, mbpp_spec, physics_review, unresolved

UNITS = pint.UnitRegistry()
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# The runner reads only candidate code, function arguments, or benchmark assertions.
# Resource limits and the read-only filesystem are imposed by Docker, outside Python.
RUNNER = '''import contextlib, copy, io, json, sys
p = json.load(sys.stdin)
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(p["code"], "candidate", "exec"), ns)
    if p.get("tests"):
        for test in p["tests"]:
            exec(compile(test, "test", "exec"), ns)
        result = True
    else:
        result = []
        for args in p["inputs"]:
            before = copy.deepcopy(args)
            result.append(ns[p["entrypoint"]](*args))
            if p.get("preserve_inputs") and args != before:
                raise ValueError("Candidate mutated inputs against the task contract")
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
    # Bound exponent parsing before Fraction allocates an enormous integer. This
    # is a declared parser limit, independent of any model's observed answers.
    for literal in (match[1], match[2] or "1"):
        exponent = re.search(r"[eE]([+-]?\d+)$", literal)
        if exponent and abs(int(exponent[1])) > 1000:
            raise ValueError("Numeric exponent exceeds parser limit")
    try:
        value = Fraction(match[1])
        if match[2]:
            value /= Fraction(match[2])
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite answer")
    except ArithmeticError as exc:
        raise ValueError("Invalid or overflowing numeric answer") from exc
    return value, match[3].strip()


def run_code(code: str, *, entrypoint: str = "", inputs: list | None = None,
             tests: list[str] | None = None, preserve_inputs: bool = False,
             image: str = "occam-verifier:1", timeout: int = 8, mbpp: dict | None = None):
    """Run under a nonroot user, no network/mounts/capabilities, bounded resources."""
    name = "occam-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "32",
               "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--user", "65534:65534",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=8m", "-i", image, "python", "-I"]
    command.extend(["/opt/evalplus_runner.py"] if mbpp is not None else ["-c", RUNNER])
    payload = json.dumps({"code": code, "entrypoint": entrypoint, "inputs": inputs or [], "tests": tests or [],
                          "preserve_inputs": preserve_inputs, "mbpp": mbpp})
    # Limit regular-file output outside the container. communicate(capture_output=True)
    # would otherwise let a noisy candidate allocate unbounded host RAM.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(["prlimit", "--fsize=1048576", "--", *command], input=payload.encode(),
                                    stdout=stdout, stderr=stderr, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ValueError("Code exceeded wall-clock limit") from None
        finally:
            # Terminating the Docker CLI does not guarantee container termination.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        stdout.seek(0)
        stderr.seek(0)
        output, errors = stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")
    if result.returncode:
        # Infrastructure failures must not be misclassified as wrong model answers.
        if mbpp is not None or result.returncode == 125 or "Cannot connect" in errors or "Unable to find image" in errors:
            raise RuntimeError(errors)
        raise ValueError("Code failed: " + errors[-2000:])
    if len(output) > 1_000_000:
        raise ValueError("Code output too large")
    return json.loads(output)


def verify(task: Task, trace: str, image: str = "occam-verifier:1") -> Verification:
    """Correctness cannot be supplied by the LLM judge or a number in the reasoning."""
    if task.source == UG and unresolved([task]):
        raise RuntimeError(f"Unresolved scientific reference: {task.id}")
    extracted = ""
    try:
        if task.domain == "code":
            blocks = regions(trace, "code")
            if len(blocks) != 1:
                return Verification(correct=False, reason="Expected exactly one code submission")
            extracted = blocks[0]
            if task.source == MBPP:
                accepted = run_code(extracted, entrypoint=task.entrypoint, inputs=task.inputs,
                                    mbpp=mbpp_spec(task), image=EVAL_IMAGE, timeout=150)
                return Verification(correct=accepted, reason="tests_passed" if accepted else "hidden_case_failed",
                                    extracted=extracted)
            result = run_code(extracted, entrypoint=task.entrypoint, inputs=task.inputs, tests=task.tests,
                              preserve_inputs=task.preserve_inputs, image=image)
            correct = result is True if task.tests else result == task.expected
            return Verification(correct=correct, reason="tests_passed" if correct else "hidden_case_failed",
                                extracted=extracted)
        answers = regions(trace, "answer")
        if len(answers) != 1:
            return Verification(correct=False, reason="Expected exactly one final answer")
        extracted = answers[0].strip()
        value, unit = numeric(extracted)
        expected, _ = numeric(task.answer)
        target_unit, rtol, atol = task.unit, task.rtol, task.atol
        if task.source == UG:
            ruling = physics_review()[task.id]
            target_unit = ruling["unit"] if ruling["unit"] not in ("1", "coordinate") else ""
            # Half the declared last reporting increment, in target units. Never apply
            # the legacy universal 1e-8 tolerance to tiny physical quantities.
            rtol, atol = 0.0, float(ruling["rounding_increment"]) / 2
        if target_unit:
            if not unit:
                return Verification(correct=False, reason="Missing units", extracted=extracted)
            converted = UNITS.Quantity(float(value), unit).to(target_unit).magnitude
            if task.source == UG and target_unit == "rad" and "modulo" in ruling["review"]:
                converted = float(expected) + math.remainder(converted - float(expected), math.tau)
            correct = math.isclose(converted, float(expected), rel_tol=max(rtol, 1e-12), abs_tol=atol)
        else:
            correct = not unit and (math.isclose(float(value), float(expected), rel_tol=rtol, abs_tol=atol)
                                    if rtol or atol else value == expected)
        return Verification(correct=bool(correct), reason="matched" if correct else "mismatch", extracted=extracted)
    # Pint uses Python's tokenizer, which also raises TokenError on malformed
    # expressions: https://docs.python.org/3.12/library/tokenize.html
    except (ValueError, ArithmeticError, SyntaxError, TokenError, pint.errors.PintError) as exc:
        return Verification(correct=False, reason=str(exc), extracted=extracted)
