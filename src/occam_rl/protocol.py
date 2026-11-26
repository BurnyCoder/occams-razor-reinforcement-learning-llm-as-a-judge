"""Reconstruct retained treatment, evaluator, decoding and scorer identities without models."""

from .io import digest, file_hash, read_json
from .prompts import JUDGE, policy_messages
from .readiness import recount_audit
from .schema import PROMPT_ARMS, Task
from .scoring import scoring_identity, verified_eval_image


def evaluation_protocol(cfg, arm, seed, root, runs):
    """Derive provenance from original files; evaluation-time waiver switches cannot erase history."""
    manifest = root / "manifests"
    revisions = read_json(manifest / "revisions.json")["revisions"]
    judge = read_json(manifest / "judge-gemma.json")
    selected = read_json(manifest / "gemma-selection.json")
    audit = read_json(root / cfg.get("gemma_audit", "configs/judge-audit.json"))
    expected_cases = [f"audit/pair/{i}" for i in range(len(audit["pairs"]))]
    expected_cases += [f"audit/omission/{i}" for i in range(len(audit["omissions"]))]
    if (judge["role"] != "gemma" or judge["model"] != selected["model"]
            or judge["revision"] != selected["revision"] or judge["revision"] != revisions[selected["model"]]
            or judge["prompt_hash"] != digest(JUDGE) or judge["audit_hash"] != digest(audit)
            or [r["id"] for r in judge["rows"]] != expected_cases or not recount_audit(judge)["passed"]):
        raise ValueError("Independent evaluator qualification is missing or inconsistent")
    task = Task(id="identity", domain="math", family="identity", split="dev", question="", reference="")
    protocol = dict(scorer=scoring_identity(), evalplus_image=verified_eval_image(), evaluator=digest(judge), base=revisions[cfg["policy"]],
                    instruction=digest(policy_messages(task, arm)[0]), adapter=None, training=None,
                    waiver=False, decoding=digest({k: cfg[k] for k in ("temperature", "top_p", "top_k",
                    "max_prompt_tokens", "max_completion_tokens")} | {"thinking": True, "min_p": 0,
                    "repetition_penalty": 1, "sampling": True}))
    if arm in PROMPT_ARMS:
        return protocol
    directory = runs / f"{arm}-seed{seed}"
    run, status = (read_json(directory / name) for name in ("run.json", "status.json"))
    fingerprint = {k: run[k] for k in ("config", "arm", "seed", "pilot", "data", "revisions", "normalizers", "judge")}
    if (any((r["arm"], r["seed"], r["pilot"]) != (arm, seed, False) for r in (run, status))
            or not status["complete"] or status["steps"] != status["target_steps"]
            or digest(fingerprint) != status["fingerprint"] or status["fingerprint"] != run["fingerprint"]
            or run["revisions"]["revisions"][run["config"]["policy"]] != protocol["base"]
            or run["data"] != read_json(manifest / "data.json")["files"]["train.jsonl"]
            or run["judge"]["role"] != "qwen" or run["judge"]["model"] != run["config"]["judge"]
            or run["judge"]["revision"] != run["revisions"]["revisions"][run["config"]["judge"]]
            or run["judge"]["prompt_hash"] != digest(JUDGE)
            or run["judge"]["audit_hash"] != digest(read_json(root/run["config"].get("qwen_audit","configs/judge-audit.json")))
            or recount_audit(run["judge"])["passed"] != status["judge_qualified"]):
        raise ValueError("Training identity is incomplete or inconsistent")
    model_digest = file_hash(directory / "adapter/adapter_model.safetensors")
    protocol.update(adapter=model_digest, training=digest([run, status]),
                    waiver=bool(status["exploratory_judge_override"] or not status["judge_qualified"]),
                    matched_training=digest([{k: fingerprint[k] for k in ("config", "data", "revisions", "judge")},
                                             status["target_steps"]]))
    return protocol
