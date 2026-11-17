"""Two-stage local evaluation: save complete policy outputs, then load Gemma."""

import statistics
from pathlib import Path

from .data import load_tasks
from .io import ROOT, data_dir, digest, read_json, read_jsonl, runs_dir, write_json
from .models import GemmaJudge, clear_gpu, generate, load_policy
from .prompts import policy_messages
from .publication import require_data_manifest, require_first_occam
from .rewards import code_structure, formal_length
from .schema import DOMAINS, PROMPT_ARMS
from .training import require_qualification
from .verify import verify


def evaluate(cfg: dict, log, commit: str, arm: str, seed: int, split: str, limit: int | None = None) -> None:
    """Resume by task ID; each task gets a stable RNG seed independent of interruption."""
    require_data_manifest(data_dir() / f"{split}.jsonl")
    require_qualification("gemma", cfg)
    if arm != "occam":
        require_first_occam()
    tasks = load_tasks(split)
    if limit is not None:
        # Equal-domain exploratory subset; never silently label a limited run confirmatory.
        tasks = [t for d in DOMAINS for t in [x for x in tasks if x.domain == d][:limit]]
    suffix = f"-limit{limit}" if limit is not None else ""
    output = runs_dir() / f"{arm}-seed{seed}" / f"eval-{split}{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"arm": arm, "seed": seed, "split": split, "limit_per_domain": limit, "tasks": [t.id for t in tasks],
                "config_hash": digest(cfg), "data": read_json(ROOT / "manifests/data.json")["files"][f"{split}.jsonl"]}
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != manifest:
        raise RuntimeError("Evaluation resume mismatch")
    write_json(output / "manifest.json", manifest)
    from .io import EventLog
    generations_log = EventLog(output / "generations.jsonl")
    saved = {row["task_id"]: row for row in read_jsonl(generations_log.path)}
    if any(t.id not in saved for t in tasks):
        adapter = None if arm in PROMPT_ARMS else runs_dir() / f"{arm}-seed{seed}" / "adapter"
        model, tokenizer = load_policy(cfg, seed, adapter)
        from transformers import set_seed
        try:
            for task in tasks:
                if task.id in saved:
                    continue
                set_seed(int(digest([seed, task.id])[:8], 16))
                result = generate(model, tokenizer, policy_messages(task, arm), cfg, log)
                verified = verify(task, result["trace"], cfg["docker_image"])
                saved[task.id] = generations_log.emit("generation", task_id=task.id, domain=task.domain,
                    family=task.family, source=task.source, arm=arm, seed=seed, git_commit=commit,
                    correct=int(verified.correct), verifier=verified.model_dump(), **result,
                    formal_length=formal_length(result["trace"]), **code_structure(result["trace"]))
        finally:
            del model, tokenizer
            clear_gpu()
    results_log = EventLog(output / "results.jsonl")
    done = {r["task_id"] for r in read_jsonl(results_log.path)}
    with GemmaJudge(cfg, log) as judge:
        for task in tasks:
            if task.id in done:
                continue
            row = dict(saved[task.id])
            assessed = judge.assess(task, row["trace"])
            row.pop("event", None)
            row.pop("time", None)
            row.update(judgment=assessed.model_dump() if assessed else None,
                       adequate=int(bool(assessed and assessed.adequate)),
                       simplicity=assessed.simplicity if assessed else 0,
                       judge_invalid=assessed is None)
            row["quality"] = row["correct"] * row["adequate"] * row["simplicity"]
            results_log.emit("evaluation", **row)
    log.emit("evaluation_complete", arm=arm, seed=seed, split=split, count=len(tasks), exploratory=limit is not None)


def calibrate(cfg: dict, log) -> None:
    """Freeze positive median lengths from ordinary prompting on calibration only."""
    require_first_occam()
    path = runs_dir() / "prompt_ordinary-seed17/eval-calibration/generations.jsonl"
    rows = read_jsonl(path)
    expected = {t.id for t in load_tasks("calibration")}
    if {r["task_id"] for r in rows} != expected:
        raise RuntimeError("Run the complete ordinary prompt calibration before freezing normalizers")
    result = {}
    for domain in DOMAINS:
        subset = [r for r in rows if r["domain"] == domain]
        result[domain] = {arm: statistics.median(values) if values else 1 for arm, values in
                          (("token", [r["tokens"] for r in subset if r["tokens"] > 0]),
                           ("formal", [r["formal_length"] for r in subset if r["formal_length"] > 0]))}
    write_json(ROOT / "manifests/normalizers.json", result)
    log.emit("normalizers_frozen", normalizers=result, source_hash=digest(rows))
