"""Deterministic procedural tasks, public adapters, and immutable split manifests.

Public records use upstream revisions and licenses. Reasoning Gym is called through
its dataset API; no source is copied. https://github.com/open-thought/reasoning-gym
"""

import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path

from pint.errors import PintError

from .io import ROOT, data_dir, digest, file_hash, read_json, read_jsonl, write_json
from .schema import DOMAINS, Task
from .verify import numeric, run_code

SPLIT_SEEDS = {"train": 1000003, "dev": 2000003, "calibration": 3000003,
               "test": 4000003, "ood": 5000003}


def question_hash(question: str) -> str:
    """Normalize whitespace and case, without erasing task-defining numbers."""
    canonical = re.sub(r"solve_\d+", "solve", question)
    return digest(re.sub(r"\s+", " ", canonical).strip().lower())


def check_split_overlap(tasks: list[Task]) -> None:
    """Any repeated normalized problem is an error, including repeated same-split rows."""
    seen = {}
    for task in tasks:
        key = question_hash(task.question)
        if key in seen:
            raise ValueError(f"Duplicate question: {seen[key]} and {task.id}")
        seen[key] = task.id


def _math(rng: random.Random, variant: int, ood: bool) -> dict:
    """Exact references use Fraction, avoiding floating-point label noise."""
    a, b, x = rng.randint(2, 30), rng.randint(1, 90), rng.randint(-100, 100)
    if ood:
        if variant % 2:
            return dict(family="math_geometric", question=f"A geometric sequence starts at {a}, has ratio 2. Find its {b % 8 + 2}th term; justify the exponent.",
                        answer=str(a * 2 ** (b % 8 + 1)), reference="The nth term is a times ratio^(n-1).")
        return dict(family="math_difference_squares", question=f"Evaluate {abs(x)+a} squared minus {abs(x)+b} squared using an algebraic identity.",
                    answer=str((abs(x)+a)**2 - (abs(x)+b)**2), reference="Use u^2-v^2=(u-v)(u+v).")
    if variant % 3 == 0:
        return dict(family="math_affine", question=f"Solve {a}x + {b} = {a*x+b} for x and justify the transformation.",
                    answer=str(x), reference=f"Subtract {b}, then divide by nonzero {a}: x={x}.")
    if variant % 3 == 1:
        c = rng.randint(1, 70)
        return dict(family="math_fraction", question=f"Compute {a}/{b} + {x}/{c} as a rational number. Show the key arithmetic.",
                    answer=str(Fraction(a,b)+Fraction(x,c)), reference="Use a common denominator and reduce the fraction.")
    return dict(family="math_remainder", question=f"Find the remainder when {abs(x)*b+a} is divided by {b+1}; justify it using quotient and remainder.",
                answer=str((abs(x)*b+a) % (b+1)), reference="Use n=q*d+r with 0<=r<d.")


def _physics(rng: random.Random, variant: int, ood: bool) -> dict:
    """Introductory idealized laws, with units and assumptions in the specification."""
    a, b, c = (rng.randint(2, 60) for _ in range(3))
    common = dict(rtol=1e-5, atol=1e-8, requirements=["Identify the applicable law", "Substitute given quantities", "State the result with units"])
    if ood:
        if variant % 2:
            return common | dict(family="physics_momentum", question=f"An isolated {a} kg cart at {b} m/s sticks to a stationary {c} kg cart. Find their final velocity and justify the conservation law.",
                                 answer=str(Fraction(a*b,a+c)), unit="m/s", reference="Conserve momentum, not kinetic energy: v=mv0/(m+M).")
        return common | dict(family="physics_capacitance", question=f"An ideal {a} F capacitor stores {b} coulombs. Find its voltage; state the defining relation.",
                             answer=str(Fraction(b,a)), unit="V", reference="Q=CV, so V=Q/C.")
    variants = [
        dict(family="physics_kinematics", question=f"A body starts at {a} m/s with constant acceleration {b} m/s^2 for {c} s. Find its final velocity and show the relation.", answer=str(a+b*c), unit="m/s", reference="For constant acceleration, v=u+at."),
        dict(family="physics_force", question=f"A {a} kg mass has net force {b} N. Find its acceleration in an inertial frame; show the governing law.", answer=str(Fraction(b,a)), unit="m/s^2", reference="Newton's second law F=ma gives a=F/m."),
        dict(family="physics_ohm", question=f"An ideal ohmic resistor of {a} ohms has {b} volts across it. Find the current and show the relation.", answer=str(Fraction(b,a)), unit="A", reference="Ohm's law V=IR gives I=V/R."),
        dict(family="physics_series", question=f"Ideal resistors of {a} and {b} ohms are in series across {c} volts. Find current; justify the equivalent resistance.", answer=str(Fraction(c,a+b)), unit="A", reference="Series resistance adds: I=V/(R1+R2)."),
        dict(family="physics_work", question=f"A constant {a} N force acts parallel to a displacement of {b} m. Find the work and state the angular assumption.", answer=str(a*b), unit="J", reference="W=Fd cos(theta); parallel means theta=0.")]
    return common | variants[variant % len(variants)]


def _code(rng: random.Random, variant: int, ood: bool) -> dict:
    """Pure-function hidden cases are calculated independently of candidate execution."""
    k, b = rng.randint(2, 200), rng.randint(-200, 200)
    cases = [[], [0], [-1, 0, 1], [k, k, -k], [b-1, b, b+1], [rng.randint(-500, 500) for _ in range(13)]]
    if ood:
        if variant % 2:
            transform = lambda xs: sum(x*y for i,x in enumerate(xs) for y in xs[i+1:] if x+y > b)
            body = f"return sum(x*y for i,x in enumerate(xs) for y in xs[i+1:] if x+y > {b})"
            spec, family = f"sum x*y over index pairs i<j whose values sum to more than {b}", "code_pair_products"
        else:
            transform = lambda xs: [sum(xs[:i+1]) % k for i in range(len(xs))]
            body = f"return [sum(xs[:i+1]) % {k} for i in range(len(xs))]"
            spec, family = f"the list of inclusive prefix sums modulo {k}", "code_prefix_mod"
    elif variant % 3 == 0:
        transform = lambda xs: sum(x for x in xs if x % k == 0)
        body = f"return sum(x for x in xs if x % {k} == 0)"
        spec, family = f"the sum of all values divisible by {k}", "code_filter_sum"
    elif variant % 3 == 1:
        transform = lambda xs: [k*x+b for x in xs]
        body = f"return [{k}*x+({b}) for x in xs]"
        spec, family = f"a new list with each value x replaced by {k}*x+({b})", "code_affine_map"
    else:
        transform = lambda xs: sum(x > b for x in xs)
        body = f"return sum(x > {b} for x in xs)"
        spec, family = f"the count of values strictly greater than {b}", "code_count_threshold"
    # A specification salt is substantive: require a distinct valid function name.
    name = f"solve_{rng.randrange(100000, 999999)}"
    return dict(family=family, question=f"Write Python function {name}(xs), for any finite list of integers, returning {spec}. Do not mutate xs. Explain empty-list and negative-value behavior.",
                reference=f"def {name}(xs):\n    {body}", requirements=["Satisfy all list sizes", "Handle negatives and duplicates", "Explain empty-list behavior"],
                entrypoint=name, inputs=[[xs] for xs in cases], expected=[transform(xs) for xs in cases])


def procedural_tasks(split: str, per_domain: int) -> list[Task]:
    """Deterministic new instances, with held-out families only for the OOD split."""
    tasks, seen = [], set()
    for domain, generator in zip(DOMAINS, (_math, _physics, _code), strict=True):
        rng = random.Random(SPLIT_SEEDS[split] + DOMAINS.index(domain) * 100000)
        index, attempts = 0, 0
        while index < per_domain:
            fields = generator(rng, attempts, split == "ood")
            attempts += 1
            key = question_hash(fields["question"])
            if key in seen:
                continue
            seen.add(key)
            tasks.append(Task(id=f"proc/{split}/{domain}/{index}", domain=domain, split=split, **fields))
            index += 1
    return tasks


def plain_json(value) -> bool:
    """Reject inputs whose tuple/set/complex semantics would change on JSON transport."""
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(plain_json(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and plain_json(v) for k,v in value.items())
    return False


def _unbox(value: str) -> str:
    """Conservative LaTeX numeric adapter; unsupported symbolic syntax stays excluded."""
    value = value.strip().replace("$", "")
    value = re.sub(r"\\boxed\{(.*)\}", r"\1", value)
    value = re.sub(r"\\(?:d?frac)\{([+-]?[0-9.]+)\}\{([+-]?[0-9.]+)\}", r"\1/\2", value)
    value = re.sub(r"\s*\\times\s*10\^\{([+-]?\d+)\}", r"e\1", value)
    return value.strip()


def public_tasks(cfg: dict, revisions: dict, log) -> list[Task]:
    """Eligibility is based on schemas, supported types and reference checks, never policy results."""
    from datasets import load_dataset

    tasks = []
    for split, source_split in (("train", "train"), ("test", "test")):
        rows = load_dataset("openai/gsm8k", "main", split=source_split, revision=revisions["openai/gsm8k"])
        for i, row in enumerate(rows):
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            try:
                numeric(answer)
            except ValueError:
                log.emit("exclude", source="gsm8k", id=i, reason="non-numeric answer")
                continue
            tasks.append(Task(id=f"gsm8k/{source_split}/{i}", domain="math", family=f"gsm8k/{source_split}/{i}",
                              split=split, question=row["question"], answer=answer, reference=row["answer"],
                              source="openai/gsm8k", license="MIT"))
    rows = load_dataset("google-research-datasets/mbpp", "full", split="train",
                        revision=revisions["google-research-datasets/mbpp"])
    for row in rows:
        tasks.append(Task(id=f"mbpp/{row['task_id']}", domain="code", family=f"mbpp/{row['task_id']}",
                          split="train", question=row["text"], reference=row["code"], tests=row["test_list"],
                          source="google-research-datasets/mbpp", license="CC-BY-4.0"))
    for subject in ("ClassicalMechanics", "ClassicalElectromagnetism"):
        rows = load_dataset("UGPhysics/ugphysics", subject, split="en", revision=revisions["UGPhysics/ugphysics"])
        for row in rows:
            identifier = f"ugphysics/{subject}/{row['index']}"
            try:
                if row["answer_type"] != "NV" or row["is_multiple_answer"] or re.search(r"figure|diagram|image|shown", row["problem"], re.I):
                    raise ValueError("unsupported answer type or figure-dependent wording")
                answer = _unbox(row["answers"])
                value, embedded_unit = numeric(answer)
                if embedded_unit:
                    raise ValueError("unit embedded in answer")
                unit = re.sub(r"\\(?:mathrm|text)\{([^}]+)\}", r"\1", row["unit"] or "").replace("$", "").strip()
                if unit:
                    from .verify import UNITS
                    UNITS.Unit(unit)
                tasks.append(Task(id=identifier, domain="physics", family=identifier, split="test", question=row["problem"],
                                  answer=str(value), unit=unit, reference=row["solution"], rtol=1e-3, atol=1e-8,
                                  source="UGPhysics/ugphysics", license="CC-BY-NC-SA-4.0"))
            except (ValueError, TypeError, KeyError, PintError) as exc:
                # Eligibility failures are retained before any model sees the tasks.
                log.emit("exclude", source="UGPhysics/ugphysics", id=identifier, reason=str(exc))
    from evalplus.data import get_mbpp_plus
    mbpp = get_mbpp_plus(version="v0.2.0")
    train_ids = {t.id.split("/")[-1] for t in tasks if t.source == "google-research-datasets/mbpp"}
    for identifier, row in mbpp.items():
        try:
            if identifier.split("/")[-1] in train_ids:
                raise ValueError("MBPP official training ID")
            inputs = row["base_input"] + row["plus_input"]
            if not plain_json(inputs):
                raise ValueError("non-JSON input semantics")
            reference = row["canonical_solution"]
            if not re.search(r"^def\s", reference, re.M):
                reference = row["prompt"] + reference
            expected = run_code(reference, entrypoint=row["entry_point"], inputs=inputs, image=cfg["docker_image"], timeout=30)
            tasks.append(Task(id=identifier, domain="code", family=identifier, split="test", question=row["prompt"],
                              reference=reference, entrypoint=row["entry_point"], inputs=inputs, expected=expected,
                              source="evalplus/mbppplus-v0.2.0", license="CC-BY-4.0"))
        except (ValueError, SyntaxError, KeyError) as exc:
            log.emit("exclude", source="MBPP+", id=identifier, reason=str(exc))
    return tasks


def prepare(cfg: dict, log, *, public: bool = True) -> None:
    """Freeze tasks, exclusions and file hashes; caller publishes manifest before training."""
    from transformers import AutoTokenizer
    from .prompts import policy_messages

    revisions = read_json(ROOT / "manifests/revisions.json")["revisions"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["policy"], revision=revisions[cfg["policy"]])
    counts = {"train": cfg["train_per_domain"], "dev": cfg["dev_per_domain"],
              "calibration": cfg["calibration_per_domain"], "test": cfg["test_procedural_per_domain"],
              "ood": cfg["ood_per_domain"]}
    procedural = [t for split, count in counts.items() for t in procedural_tasks(split, count * 4)]
    # Reuse a public generator for a subset of training math; its separate RNG seed
    # namespace does not overlap with any local generator or held-out family.
    import reasoning_gym
    gym = reasoning_gym.create_dataset("simple_equations", size=cfg["train_per_domain"], seed=81000003,
                                       min_terms=2, max_terms=4, min_value=1, max_value=30)
    for i in range(cfg["train_per_domain"] // 4):
        row = gym[i]
        procedural.append(Task(id=f"gym/train/{i}", domain="math", family="gym_simple_equations", split="train",
                               question=row["question"], answer=row["answer"], reference="Rearrange the linear equation using inverse operations; verify by substitution.", source="reasoning-gym/simple_equations"))
    upstream = public_tasks(cfg, revisions, log) if public else []
    selected, hashes = [], set()
    for split in counts:
        for domain in DOMAINS:
            proc_pool = [t for t in procedural if t.split == split and t.domain == domain]
            pub_pool = [t for t in upstream if t.split == split and t.domain == domain]
            pub_count = (cfg["test_public_per_domain"] if split == "test" else
                         cfg["train_per_domain"] // 2 if split == "train" and domain != "physics" else 0) if public else 0
            proc_count = counts[split] - (pub_count if split == "train" else 0)
            for pool, target, label in ((proc_pool, proc_count, "procedural"), (pub_pool, pub_count, "public")):
                pool.sort(key=lambda t: digest(t.id))
                kept = 0
                for task in pool:
                    if kept >= target:
                        break
                    key = question_hash(task.question)
                    # Use the longest prompt treatment for eligibility across all arms.
                    ids = tokenizer.apply_chat_template(policy_messages(task, "prompt_occam"), tokenize=True,
                                                        add_generation_prompt=True, enable_thinking=True)
                    if key in hashes or len(ids) > cfg["max_prompt_tokens"]:
                        log.emit("exclude", id=task.id, reason="duplicate or prompt budget", prompt_tokens=len(ids))
                        continue
                    hashes.add(key)
                    selected.append(task)
                    kept += 1
                if kept != target:
                    log.emit("stratum_shortfall", split=split, domain=domain, source=label, target=target, actual=kept)
    check_split_overlap(selected)
    files = {}
    for split in counts:
        path = data_dir() / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [t for t in selected if t.split == split]
        path.write_text("".join(t.model_dump_json() + "\n" for t in rows), encoding="utf-8")
        files[path.name] = {"sha256": file_hash(path), "count": len(rows)}
    manifest = {"config_hash": digest(cfg), "public": public, "files": files,
                "tasks": [{"id": t.id, "split": t.split, "domain": t.domain, "family": t.family,
                           "source": t.source, "license": t.license, "question_sha256": question_hash(t.question)} for t in selected]}
    write_json(ROOT / "manifests/data.json", manifest)
    log.emit("data_prepared", files=files, manifest_hash=digest(manifest))


def load_tasks(split: str) -> list[Task]:
    """Validate every consumed task against its typed schema."""
    return [Task.model_validate(row) for row in read_jsonl(data_dir() / f"{split}.jsonl")]
