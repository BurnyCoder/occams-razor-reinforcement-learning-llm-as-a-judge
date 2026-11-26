"""Orchestrate documented TRL GRPO, leaving its loss and optimizer machinery intact.

Sources: https://huggingface.co/docs/trl/grpo_trainer and
https://unsloth.ai/docs/models/qwen3.5/fine-tune . The rollout hook serializes
generation for the small GPU; frozen judging temporarily parks the policy in RAM.
"""

import random
import signal
import time
from pathlib import Path

from .data import load_tasks
from .gpu import MemoryMonitor
from .io import ROOT, data_dir, digest, read_json, read_jsonl, runs_dir, write_json
from .models import QwenJudge, clear_gpu, generate, load_policy, parked_policy
from .prompts import policy_messages
from .publication import require_data_manifest, require_first_occam, require_hf_publication
from .rewards import formal_length, reward
from .schema import DOMAINS, Task
from .verify import verify


def require_qualification(role: str, cfg: dict, *, allow_unqualified=False) -> dict:
    """Bind the audit to this judge; an explicit training-only waiver keeps its failure."""
    from .prompts import JUDGE
    from .models import revision
    result = read_json(ROOT / f"manifests/judge-{role}.json")
    expected = cfg["judge"] if role == "qwen" else read_json(ROOT / "manifests/gemma-selection.json")["model"]
    audit_path = ROOT / cfg.get(f"{role}_audit", "configs/judge-audit.json")
    if (result["model"] != expected or result["prompt_hash"] != digest(JUDGE)
            or result["revision"] != revision(expected) or result["audit_hash"] != digest(read_json(audit_path))):
        raise RuntimeError(f"{role} judge is not qualified for the current prompt/model")
    if not result["passed"] and not (role == "qwen" and allow_unqualified):
        raise RuntimeError(f"{role} judge is not qualified for the current prompt/model")
    return result


def assert_unstarted_run(output: Path) -> None:
    """A startup retry may reset the seed only if no rollout or optimizer work exists."""
    events = read_jsonl(output / "events.jsonl", tolerate_tail=True)
    if any(row.get("event") in {"llm_prompt", "rollout", "group", "optimizer_step"} for row in events):
        raise RuntimeError("No full checkpoint exists, but rollout/optimizer work was recorded")


def train(cfg: dict, log, commit: str, arm: str, seed: int, *, pilot=False, resume=False) -> Path:
    """Train one run; every dependent artifact must already be published."""
    require_hf_publication(commit)
    require_data_manifest(data_dir() / "train.jsonl")
    # Validate task deserialization before spending time loading GPU libraries/weights.
    tasks = load_tasks("train")
    judge_audit = require_qualification("qwen", cfg,
        allow_unqualified=cfg.get("allow_unqualified_training_judge", False))
    if not judge_audit["passed"]:
        log.emit("exploratory_judge_override", audit=judge_audit,
                 reason="User authorized training despite residual judge-quality failures")
    if arm != "occam":
        require_first_occam()
    preflight = read_json(ROOT / "manifests/preflight.json")
    if not preflight["passed"] or preflight["config_hash"] != digest(cfg):
        raise RuntimeError("GPU profile and state-preservation checks must pass for this configuration")
    if not pilot:
        budget = read_json(ROOT / "manifests/budget.json")
        cfg = cfg | {"updates": budget["updates"]}
    normalizers = read_json(ROOT / "manifests/normalizers.json") if arm in ("token", "formal") else {}
    name = f"{arm}-seed{seed}" + ("-pilot" if pilot else "")
    output = runs_dir() / name
    fingerprint = {"config": cfg, "arm": arm, "seed": seed, "pilot": pilot,
                   "data": read_json(ROOT / "manifests/data.json")["files"]["train.jsonl"],
                   "revisions": read_json(ROOT / "manifests/revisions.json"), "normalizers": normalizers,
                   "judge": read_json(ROOT / "manifests/judge-qwen.json")}
    marker = output / "run.json"
    if marker.exists():
        previous = read_json(marker)
        if not resume or previous["fingerprint"] != digest(fingerprint):
            raise RuntimeError("Existing run: use --resume only with unchanged scientific configuration")
    else:
        if resume:
            raise RuntimeError("Cannot resume a run without its identity manifest")
        write_json(marker, {"name": name, "git_commit": commit, "fingerprint": digest(fingerprint), **fingerprint})

    # Import Unsloth before Transformers/TRL so its documented patches are applied.
    model, tokenizer = load_policy(cfg, seed)
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    set_seed(seed)
    per_domain = {d: [t for t in tasks if t.domain == d] for d in DOMAINS}
    if len({len(v) for v in per_domain.values()}) != 1:
        raise RuntimeError("Equal-domain training counts are required; fix and publish the data")
    rows = []
    for i in range(len(per_domain["math"])):
        for domain in DOMAINS:
            task = per_domain[domain][i]
            rows.append({"prompt": policy_messages(task), "task_json": task.model_dump_json()})
    dataset = Dataset.from_list(rows)

    def rollout(prompts, trainer):
        """TRL repeats each prompt G times; generate one completion per supplied row."""
        results = [generate(trainer.accelerator.unwrap_model(trainer.model), tokenizer, prompt, cfg, log) for prompt in prompts]
        return {"prompt_ids": [r["prompt_ids"] for r in results],
                "completion_ids": [r["completion_ids"] for r in results], "logprobs": None,
                "full_trace": [r["trace"] for r in results], "truncated": [r["truncated"] for r in results],
                "duration": [r["seconds"] for r in results]}

    def occam_reward(completions, completion_ids, task_json, full_trace, truncated, duration,
                     trainer_state=None, **kwargs):
        """Verify first, then assess every eligible full trace with the frozen judge."""
        batch_tasks = [Task.model_validate_json(raw) for raw in task_json]
        verdicts = [verify(t, trace, cfg["docker_image"]) for t,trace in zip(batch_tasks, full_trace, strict=True)]
        judgments = [None] * len(verdicts)
        if arm != "accuracy" and any(v.correct for v in verdicts):
            with parked_policy(model):
                with QwenJudge(cfg, log) as judge:
                    for i, (task, trace, verdict) in enumerate(zip(batch_tasks, full_trace, verdicts, strict=True)):
                        if verdict.correct:
                            judgments[i] = judge.assess(task, trace)
        step = trainer_state.global_step if trainer_state is not None else 0
        shuffled = [None] * len(verdicts)
        if arm == "shuffled":
            for start in range(0, len(verdicts), cfg["group_size"]):
                eligible = [i for i in range(start, start+cfg["group_size"]) if verdicts[i].correct and judgments[i] and judgments[i].adequate]
                scores = [judgments[i].simplicity for i in eligible]
                random.Random(seed * 1000003 + step * 97 + start).shuffle(scores)
                for i, score in zip(eligible, scores, strict=True):
                    shuffled[i] = score
        rewards = []
        for i, (task, trace, verdict, judged) in enumerate(zip(batch_tasks, full_trace, verdicts, judgments, strict=True)):
            symbols = formal_length(trace)
            score = reward(verdict.correct, judged, arm, len(completion_ids[i]), symbols,
                           normalizers.get(task.domain), cfg["bonus"], shuffled[i])
            rewards.append(score)
            log.emit("rollout", run=name, step=step, task_id=task.id, domain=task.domain,
                     family=task.family, arm=arm, seed=seed, trace=trace, correct=int(verdict.correct),
                     verifier=verdict.model_dump(), judgment=judged.model_dump() if judged else None,
                     adequate=int(bool(judged and judged.adequate)), reward=score,
                     tokens=len(completion_ids[i]), formal_length=symbols, truncated=truncated[i], seconds=duration[i])
        log.emit("group", run=name, step=step, reward_spread=max(rewards)-min(rewards),
                 accuracy=sum(v.correct for v in verdicts)/len(verdicts))
        return rewards

    stop_requested = False

    def request_stop(signum, frame):
        """Finish the current optimizer step before saving full resume state."""
        nonlocal stop_requested
        stop_requested = True
        log.emit("interrupt_requested", signal=signum, run=name)

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    class LogCallback(TrainerCallback):
        """Live metrics and clean-interruption checkpointing at optimizer boundaries."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            log.emit("training_metrics", run=name, step=state.global_step, **(logs or {}))

        def on_step_end(self, args, state, control, **kwargs):
            monitor.check()
            log.emit("optimizer_step", run=name, step=state.global_step,
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            if stop_requested:
                control.should_save, control.should_training_stop = True, True
            return control

    steps = cfg["pilot_updates"] if pilot else cfg["updates"]
    args = GRPOConfig(output_dir=str(output), max_steps=steps, learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=1, gradient_accumulation_steps=cfg["group_size"], num_generations=cfg["group_size"],
        max_completion_length=cfg["max_completion_tokens"], beta=cfg["beta"], epsilon=cfg["epsilon"],
        loss_type="dr_grpo", scale_rewards="none", num_iterations=1, bf16=True,
        gradient_checkpointing=True, optim="adamw_torch", lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0, max_grad_norm=1.0, seed=seed, data_seed=seed, remove_unused_columns=False,
        temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"], min_p=0.0,
        repetition_penalty=1.0, use_vllm=False, report_to="none", logging_steps=1,
        save_strategy="steps", save_steps=cfg["checkpoint_updates"], save_total_limit=3,
        dataloader_num_workers=0, mask_truncated_completions=False)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=args, train_dataset=dataset,
                          reward_funcs=occam_reward, rollout_func=rollout, callbacks=[LogCallback()])
    checkpoint = get_last_checkpoint(str(output)) if resume else None
    if resume and checkpoint is None:
        assert_unstarted_run(output)
        log.emit("restart_before_first_rollout", run=name, seed=seed,
                 reason="Previous attempt failed during startup; no checkpoint or rollout exists")
    start = time.monotonic()
    trainable_before = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    try:
        with MemoryMonitor(cfg["headroom_mib"], log, "training") as monitor:
            trainer.train(resume_from_checkpoint=checkpoint)
        # Full-state checkpoint at the final step includes optimizer/scheduler/RNG.
        trainer._save_checkpoint(trainer.model, trial=None)
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        changed = sum(not torch.equal(trainable_before[n], p.detach().cpu()) for n,p in model.named_parameters() if p.requires_grad)
        status = {"name": name, "arm": arm, "seed": seed, "pilot": pilot, "steps": trainer.state.global_step,
                  "judge_qualified": judge_audit["passed"],
                  "waiver_permitted": cfg.get("allow_unqualified_training_judge", False),
                  "exploratory_judge_override": not judge_audit["passed"],
                  "target_steps": steps, "complete": trainer.state.global_step >= steps,
                  "seconds": time.monotonic()-start, "changed_parameter_tensors": changed,
                  "git_commit": commit, "fingerprint": digest(fingerprint)}
        write_json(output / "status.json", status)
        log.emit("training_finished", **status)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        del trainer, model
        clear_gpu()
    return output
