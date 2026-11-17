"""Readiness, constructed judge audits, measured GPU fit, and budget selection."""

import importlib.metadata
import json
import platform
import subprocess
import time

from .gpu import MemoryMonitor, gpu_info
from .io import ROOT, digest, read_json, runs_dir, write_json
from .models import GemmaJudge, QwenJudge, clear_gpu, load_policy, parked_policy, revision
from .prompts import JUDGE, judge_messages
from .schema import Task
from .statistics import wilson


def resolve(cfg: dict, log) -> None:
    """Resolve official model/dataset revisions and one llama.cpp source commit."""
    from huggingface_hub import HfApi
    api = HfApi()
    revisions, files = {}, {}
    for model in (cfg["policy"], cfg["judge"], cfg["gemma_primary"], cfg["gemma_fallback"]):
        info = api.model_info(model)
        revisions[model] = info.sha
        if "gguf" in model:
            choices = [f.rfilename for f in info.siblings if f.rfilename.endswith(".gguf") and "mmproj" not in f.rfilename]
            if len(choices) != 1:
                raise RuntimeError(f"Ambiguous GGUF files: {choices}")
            files[model] = choices[0]
    for dataset in ("openai/gsm8k", "google-research-datasets/mbpp", "UGPhysics/ugphysics"):
        revisions[dataset] = api.dataset_info(dataset).sha
    llama = subprocess.check_output(["git", "ls-remote", "https://github.com/ggml-org/llama.cpp.git", "HEAD"], text=True, timeout=60).split()[0]
    manifest = {"revisions": revisions, "gguf_files": files, "llama_cpp_commit": llama,
                "evalplus_data_version": "v0.2.0"}
    write_json(ROOT / "manifests/revisions.json", manifest)
    log.emit("revisions_resolved", **manifest)


def doctor(cfg: dict, log) -> None:
    """Report actual runtime and fail if CUDA or sandbox infrastructure is unavailable."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model fallback is prohibited")
    docker = subprocess.check_output(["docker", "image", "inspect", cfg["docker_image"], "--format", "{{.Id}}"], text=True, timeout=30).strip()
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "unsloth", "peft", "bitsandbytes")}
    result = {"gpu": gpu_info(), "python": platform.python_version(), "platform": platform.platform(),
              "cuda_runtime": torch.version.cuda, "compute_capability": list(torch.cuda.get_device_capability()),
              "packages": packages, "docker_image": docker}
    write_json(ROOT / "manifests/runtime.json", result)
    log.emit("doctor", **result)


def qualify(cfg: dict, log, role: str) -> None:
    """Locked constructed pairs and omissions; no training examples or test outcomes."""
    audit = read_json(ROOT / "configs/judge-audit.json")
    judge_type = QwenJudge if role == "qwen" else GemmaJudge
    wins, rejected, invalid = 0, 0, 0
    rows = []
    with judge_type(cfg, log) as judge:
        for index, pair in enumerate(audit["pairs"]):
            task = Task(id=f"audit/pair/{index}", family="audit", split="audit", **{k: pair[k] for k in ("domain", "question", "reference")})
            better, worse = judge.assess(task, pair["better"]), judge.assess(task, pair["worse"])
            win = bool(better and worse and better.adequate and better.simplicity > worse.simplicity)
            wins += win
            invalid += int(better is None) + int(worse is None)
            rows.append({"id": task.id, "expected_dominance": win,
                         "better": better.model_dump() if better else None, "worse": worse.model_dump() if worse else None})
        for index, omission in enumerate(audit["omissions"]):
            task = Task(id=f"audit/omission/{index}", family="audit", split="audit", **{k: omission[k] for k in ("domain", "question", "reference")})
            judged = judge.assess(task, omission["trace"])
            # Invalid output is a gate failure, not evidence the judge recognized an omission.
            rejected += bool(judged and not judged.adequate)
            invalid += judged is None
            rows.append({"id": task.id, "judgment": judged.model_dump() if judged else None})
    model = cfg["judge"] if role == "qwen" else read_json(ROOT / "manifests/gemma-selection.json")["model"]
    result = {"role": role, "model": model, "revision": revision(model), "prompt_hash": digest(JUDGE), "audit_hash": digest(audit),
              "wins": wins, "pair_count": len(audit["pairs"]), "dominance_ci95": wilson(wins, len(audit["pairs"])),
              "rejected": rejected, "omission_count": len(audit["omissions"]), "omission_ci95": wilson(rejected, len(audit["omissions"])),
              "invalid": invalid, "passed": wins/len(audit["pairs"]) >= .8 and rejected/len(audit["omissions"]) >= .9, "rows": rows}
    write_json(ROOT / f"manifests/judge-{role}.json", result)
    log.emit("judge_audit_complete", **{k:v for k,v in result.items() if k != "rows"})


def select_gemma(cfg: dict, log) -> None:
    """Select by measured memory only, before looking at evaluative performance."""
    task = Task(id="resource/gemma", domain="math", family="resource", split="resource",
                question="Solve 2x=6 and justify it.", reference="Divide by two, giving x=3.")
    # A deliberately long resource input, not a sampled baseline or a benchmark result.
    trace = ("The calculation checks a previously stated equality. " * 1300) + "2x=6 implies x=3."
    failures = []
    for model in (cfg["gemma_primary"], cfg["gemma_fallback"]):
        try:
            with GemmaJudge(cfg, log, model) as judge:
                judge.assess(task, trace)
            write_json(ROOT / "manifests/gemma-selection.json", {"model": model, "revision": revision(model),
                        "selection_basis": "memory and runtime compatibility only", "failures": failures})
            return
        except (RuntimeError, OSError) as exc:
            failures.append({"model": model, "error": str(exc)})
            log.emit("gemma_fit_failure", **failures[-1])
    write_json(ROOT / "manifests/gemma-selection.json", {"model": None, "failures": failures})
    raise RuntimeError("Neither preregistered Gemma variant met the GPU requirement")


def profile(cfg: dict, log) -> None:
    """Maximum-size differentiable pass and role-switch identity checks on synthetic inputs."""
    model, tokenizer = load_policy(cfg, 991)
    import torch
    from transformers import set_seed
    set_seed(991)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    evidence = {}
    try:
        length = cfg["max_prompt_tokens"] + cfg["max_completion_tokens"]
        # Synthetic repeated tokens are exclusively a memory/load test, not training data.
        token = tokenizer.encode(" calculation", add_special_tokens=False)[0]
        inputs = torch.full((1, length), token, device="cuda", dtype=torch.long)
        with MemoryMonitor(cfg["headroom_mib"], log, "profile_backward") as backward:
            model.train()
            loss = model(input_ids=inputs, labels=inputs).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        evidence["backward_minimum_free_mib"] = backward.minimum
        del loss, inputs
        state = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        rng_before = torch.cuda.get_rng_state().clone()
        with parked_policy(model):
            with QwenJudge(cfg, log) as judge:
                task = Task(id="resource/qwen", domain="math", family="resource", split="resource",
                            question="Solve 2x=6.", reference="Divide by two: x=3.")
                judge.assess(task, "A calculation step is repeated. " * 700 + "<answer>3</answer>")
        evidence["switch_weights_unchanged"] = all(torch.equal(state[n], p.detach().cpu()) for n,p in model.named_parameters() if p.requires_grad)
        evidence["switch_rng_unchanged"] = torch.equal(rng_before, torch.cuda.get_rng_state())
        # Force the full generation length to measure cache growth independent of EOS.
        prompt = torch.full((1, cfg["max_prompt_tokens"]), token, device="cuda", dtype=torch.long)
        model.eval()
        log.emit("resource_prompt", purpose="worst-case generation", input_ids=prompt[0].tolist())
        with torch.inference_mode(), MemoryMonitor(cfg["headroom_mib"], log, "profile_generation") as generation:
            from .models import stream_logger
            result = model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                min_new_tokens=cfg["max_completion_tokens"], max_new_tokens=cfg["max_completion_tokens"],
                streamer=stream_logger(tokenizer, log, "resource"), use_cache=True)
        evidence["generation_minimum_free_mib"] = generation.minimum
        log.emit("resource_output", text=tokenizer.decode(result[0].tolist(), skip_special_tokens=False))
        # Validate serialization of state as an independent prerequisite to long runs.
        checkpoint = runs_dir() / "preflight/optimizer.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint)
        restored = torch.load(checkpoint, weights_only=True)
        optimizer.load_state_dict(restored["optimizer"])
        evidence["optimizer_roundtrip"] = len(optimizer.state_dict()["state"]) == len(restored["optimizer"]["state"])
        evidence["passed"] = all(evidence[k] for k in ("switch_weights_unchanged", "switch_rng_unchanged", "optimizer_roundtrip"))
        evidence["config_hash"] = digest(cfg)
        write_json(ROOT / "manifests/preflight.json", evidence)
        log.emit("preflight_complete", **evidence)
    finally:
        del optimizer, model, tokenizer
        clear_gpu()


def select_budget(cfg: dict, log) -> None:
    """Freeze common updates from the Occam pilot timing rule, not its accuracy."""
    status = read_json(runs_dir() / "occam-seed17-pilot/status.json")
    if not status["complete"] or not status["changed_parameter_tensors"]:
        raise RuntimeError("Pilot must complete and demonstrate nonzero parameter updates")
    seconds_per_update = status["seconds"] / status["steps"]
    candidates = [n for n in (64, 128, 256) if n * seconds_per_update <= 8*3600]
    if not candidates:
        raise RuntimeError("No preregistered update budget fits; publish an amendment before proceeding")
    result = {"updates": max(candidates), "seconds_per_update": seconds_per_update,
              "selection_rule": "largest of 64/128/256 projected within eight hours", "pilot_status": status}
    write_json(ROOT / "manifests/budget.json", result)
    log.emit("budget_frozen", **result)
