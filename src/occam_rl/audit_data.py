"""Reconcile frozen tasks with primary data and independently derived procedural answers on CPU."""

import ast
import json
import os
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .io import ROOT, data_dir, digest, file_hash, read_json, read_jsonl
from .schema import Task
from .scoring import MBPP, UG, mbpp_release, mbpp_spec, physics_review


def example_coverage(task):
    """Match literal call arguments in visible assertions to each scored input occurrence."""
    spec = mbpp_spec(task)
    assertions = [line.strip() for line in task.question.splitlines() if line.strip().startswith("assert ")]
    visible, unparsed = set(), []
    for line in assertions:
        try:
            calls = [node for node in ast.walk(ast.parse(line)) if isinstance(node, ast.Call)
                     and getattr(node.func, "id", None) == task.entrypoint]
            if not calls:
                raise ValueError("No entry-point call")
            for call in calls:
                if call.keywords:
                    raise ValueError("Non-positional arguments")
                args = [ast.literal_eval(arg) for arg in call.args]
                from .data import plain_json
                if not plain_json(args):
                    raise ValueError("Non-JSON literal")
                visible.add(digest(args))
        except (ValueError, SyntaxError):
            unparsed.append(line)
    hits = [digest(args) in visible for args in task.inputs]
    split = spec["base_count"]
    return dict(id=task.id, assertions=len(assertions), unparsed=unparsed, visible_distinct_inputs=len(visible),
                base_inputs=split, plus_inputs=len(hits)-split, base_matches=sum(hits[:split]),
                plus_matches=sum(hits[split:]), additional_inputs=len(hits)-sum(hits),
                release_atol=spec["atol"], entrypoint=task.entrypoint,
                comparison="EvalPlus 0.3.1 native types and v0.2.0 cases")


def procedural_reference(t):
    """Compute from problem text and inputs, independently of generator RNG and reference bodies."""
    q, family = t.question, t.family
    nums = [int(x) for x in re.findall(r"-?\d+", q)]
    if family == "math_affine":
        a, b, c = nums
        return Fraction(c-b, a), ""
    if family == "math_fraction":
        a, b, c, d = nums
        return Fraction(a,b) + Fraction(c,d), ""
    if family == "math_remainder":
        return divmod(*nums)[1], ""
    if family == "math_geometric":
        a, ratio, n = nums
        return a * ratio ** (n-1), ""
    if family == "math_difference_squares":
        a, b = nums
        return (a+b)*(a-b), ""
    if family == "physics_kinematics":
        u, a, _unit_exponent, time = nums
        return u + a*time, "m/s"
    if family == "physics_force":
        return Fraction(nums[1],nums[0]), "m/s^2"
    if family in ("physics_ohm", "physics_series"):
        return Fraction(nums[-1],sum(nums[:-1])), "A"
    if family == "physics_work":
        return nums[0]*nums[1], "J"
    if family == "physics_capacitance":
        return Fraction(nums[1],nums[0]), "V"
    if family == "physics_momentum":
        m,v,other = nums
        return Fraction(m*v,m+other), "m/s"
    # Function identifiers include digits; extract the actual parameters from the specification.
    params = nums[1:]
    results = []
    for (xs,) in t.inputs:
        if family == "code_filter_sum":
            value = sum(v for v in xs if divmod(v, params[0])[1] == 0)
        elif family == "code_count_threshold":
            value = len([v for v in xs if v > params[0]])
        elif family == "code_affine_map":
            value = [v*params[0] + params[1] for v in xs]
        elif family == "code_prefix_mod":
            from itertools import accumulate
            value = [s % params[0] for s in accumulate(xs)]
        elif family == "code_pair_products":
            from itertools import combinations
            value = sum(a*b for a,b in combinations(xs, 2) if a+b > params[-1])
        else:
            raise ValueError(f"Unknown procedural family {family}")
        results.append(value)
    return results, ""


def data_audit():
    """Check all public transfers, every native procedural reference, and the complete split inventory."""
    import pyarrow.parquet as pq
    manifest = read_json(ROOT / "manifests/data.json")
    revisions = read_json(ROOT / "manifests/revisions.json")["revisions"]
    from huggingface_hub.constants import HF_HUB_CACHE
    hub = Path(os.environ.get("HF_HOME", Path(HF_HUB_CACHE).parent)) / "hub"
    def snapshot(repo):
        return hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revisions[repo]
    gsm = {s: pq.read_table(next((snapshot("openai/gsm8k")/"main").glob(s+"-*.parquet"))).to_pylist()
           for s in ("train", "test")}
    mbpp = {r["task_id"]:r for r in pq.read_table(next((snapshot("google-research-datasets/mbpp")/
            "full").glob("train-*.parquet"))).to_pylist()}
    ug = {}
    for subject in ("ClassicalMechanics", "ClassicalElectromagnetism"):
        file = snapshot(UG)/subject/"en.jsonl"
        for line, row in enumerate(read_jsonl(file),1):
            ug[f"ugphysics/{subject}/{row['index']}"] = (row, line)
    proc, sources, files, physics, code = Counter(), Counter(), {}, [], []
    public_code, heldout = [], []
    ids, questions, families = set(), set(), {}
    declared = {r["id"]:r for r in manifest["tasks"]}
    from .data import _unbox
    for name, metadata in manifest["files"].items():
        path = data_dir()/name
        rows = [Task.model_validate(t) for t in read_jsonl(path)]
        files[name] = file_hash(path)
        if files[name] != metadata["sha256"] or len(rows) != metadata["count"]:
            raise ValueError(f"Frozen split identity differs: {name}")
        for t in rows:
            if t.split == "test":
                heldout.append(t)
            text = re.sub(r"solve_\d+", "solve", t.question)
            norm = digest(" ".join(text.lower().split()))
            record = declared[t.id]
            if t.id in ids or norm in questions or norm != record["question_sha256"] or t.split+".jsonl" != name:
                raise ValueError(f"Duplicate or substituted task {t.id}")
            if any(getattr(t,k) != record[k] for k in ("split","source","license","domain","family")):
                raise ValueError(f"Metadata differ for {t.id}")
            ids.add(t.id)
            questions.add(norm)
            families.setdefault(t.split,set()).add(t.family)
            sources[t.source] += 1
            if t.source == "procedural-v1":
                answer, unit = procedural_reference(t)
                ok = (answer == t.expected if t.domain == "code" else answer == Fraction(t.answer) and unit == t.unit)
                proc[t.family] += 1
            elif t.source == UG:
                raw, lineno = ug[t.id]
                ruling = physics_review()[t.id]
                ok = (raw["problem"] == t.question and raw["solution"] == t.reference
                      and Fraction(_unbox(raw["answers"])) == Fraction(t.answer))
                physics.append(dict(id=t.id, question_hash=norm, unit_original=t.unit, original_rtol=t.rtol,
                    original_atol=t.atol, answer=t.answer, unit_reviewed=ruling["unit"], reason=ruling["review"],
                    blocked=ruling["review"].startswith("!"), rounding_increment=ruling["rounding_increment"],
                    cluster=ruling["cluster"], upstream=f"https://huggingface.co/datasets/{UG}/blob/"
                    f"{revisions[UG]}/{t.id.split('/')[1]}/en.jsonl#L{lineno}"))
            elif t.source == "openai/gsm8k":
                raw = gsm[t.split][int(t.id.rsplit("/",1)[1])]
                ok = (t.question == raw["question"] and t.reference == raw["answer"]
                      and Fraction(t.answer) == Fraction(raw["answer"].rsplit("####",1)[1].strip().replace(",","")))
            elif t.source == "google-research-datasets/mbpp":
                raw = mbpp[int(t.id.rsplit("/",1)[1])]
                ok = (t.question.startswith(raw["text"]+"\nRequired function signature: ") and t.reference == raw["code"]
                      and t.tests == ([raw["test_setup_code"]] if (raw.get("test_setup_code") or "").strip() else []) + raw["test_list"])
            elif t.source == MBPP:
                ok = int(t.id.split("/")[1]) not in mbpp
                code.append(example_coverage(t))
                public_code.append(t)
            else:
                import reasoning_gym
                if t.source != "reasoning-gym/simple_equations":
                    raise ValueError(f"Unknown source {t.source}")
                ds = reasoning_gym.create_dataset("simple_equations", size=512, seed=81000003,
                                                  min_terms=2, max_terms=4, min_value=1, max_value=30)
                raw = ds[int(t.id.rsplit("/",1)[1])]
                ok = t.question == raw["question"] and t.answer == raw["answer"]
            if not ok:
                raise ValueError(f"Reference transfer or independent oracle disagrees: {t.id}")
    if ids != declared.keys() or families["ood"] & families["train"] or {r["id"] for r in physics} != physics_review().keys():
        raise ValueError("Frozen population, OOD family isolation or physics-ruling coverage differs")
    from .reference_replay import code_replay, prompt_budgets
    replay, image = code_replay(public_code)
    for row in code:
        row.update(replay[row["id"]])
    return dict(files=files, sources=dict(sources), independent_procedural=dict(proc), total_tasks=len(ids),
                reference_sandbox=image, prompt_budgets=prompt_budgets(heldout),
                public_reference_scope="Source fidelity; public solutions can be wrong or underspecified.",
                physics=physics, mbpp=code)
