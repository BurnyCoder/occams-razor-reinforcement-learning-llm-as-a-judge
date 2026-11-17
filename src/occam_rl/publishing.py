"""Publish explicit source snapshots and model artifacts, never environment secrets.

Hub APIs: https://huggingface.co/docs/huggingface_hub/guides/upload
"""

from pathlib import Path

from .io import ROOT, data_dir, read_json, read_jsonl, runs_dir, write_json
from .publication import git, require_data_manifest
from .schema import PROMPT_ARMS


def hub(cfg: dict):
    """Use one collection for the project, preserving its returned identity locally."""
    from huggingface_hub import HfApi
    api = HfApi()
    receipt = ROOT / ".local/hub.json"
    if receipt.exists():
        collection = read_json(receipt)["collection"]
    else:
        title = "Occam RL: adequate and parsimonious reasoning"
        matches = [c for c in api.list_collections(owner="BurnyCoder") if c.title == title]
        collection = matches[0].slug if matches else api.create_collection(title=title, namespace="BurnyCoder",
            description="Local Qwen3.5 GRPO, semantic simplicity and matched length/prompt baselines", private=False).slug
        write_json(receipt, {"collection": collection})
    return api, collection


def publish_source(cfg: dict, log, commit: str) -> None:
    """Only git-tracked files are eligible for the source upload."""
    from huggingface_hub import CommitOperationAdd
    api, collection = hub(cfg)
    repo = cfg["hf_artifacts_repo"]
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    tracked = git("ls-files").splitlines()
    if any(Path(name).name == ".env" or "token" in Path(name).name.lower() for name in tracked):
        raise RuntimeError("Potential secret-bearing tracked file; review before upload")
    # upload_folder in the pinned Hub version traverses .venv before filtering.
    # Its documented create_commit API lets Git define the exact eligible files.
    operations = [CommitOperationAdd(path_in_repo="research-README.md" if name == "README.md" else name,
                                     path_or_fileobj=str(ROOT / name)) for name in tracked]
    if (ROOT / "manifests/data.json").exists():
        files = read_json(ROOT / "manifests/data.json")["files"]
        for name in files:
            require_data_manifest(data_dir() / name)
            operations.append(CommitOperationAdd(path_in_repo=f"frozen-data/{name}", path_or_fileobj=str(data_dir() / name)))
    # Publish finished audit/profile/failure traces, excluding this active upload log.
    for path in sorted((runs_dir() / "operations").glob("*.jsonl")):
        events = read_jsonl(path, tolerate_tail=True)
        if events and events[-1].get("event") in ("phase_complete", "phase_failed"):
            operations.append(CommitOperationAdd(path_in_repo=f"operations/{path.name}", path_or_fileobj=str(path)))
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(ROOT / "docs/hf-artifacts.md")))
    result = api.create_commit(repo_id=repo, repo_type="dataset", operations=operations,
                               commit_message=f"Research source and frozen data snapshot {commit}")
    api.add_collection_item(collection, item_id=repo, item_type="dataset", exists_ok=True)
    write_json(ROOT / ".local/hf-publication.json", {"git_commit": commit, "hf_commit": result.oid,
                                                    "repo": repo, "collection": collection})
    log.emit("source_published", git_commit=commit, hf_commit=result.oid, repo=repo, collection=collection)


def publish_evaluations(api, cfg: dict, collection: str, directory: Path):
    """Both trained and prompt-only runs share the same raw artifact publication."""
    artifacts = cfg["hf_artifacts_repo"]
    api.create_repo(artifacts, repo_type="dataset", private=False, exist_ok=True)
    result = api.upload_folder(repo_id=artifacts, repo_type="dataset", folder_path=str(directory), path_in_repo=f"runs/{directory.name}",
                              allow_patterns=["*.json", "*.jsonl", "*.md", "eval-*/**"],
                              ignore_patterns=["adapter/**", "checkpoint-*/**"],
                              commit_message=f"Publish {directory.name} logs and evaluations")
    api.add_collection_item(collection, item_id=artifacts, item_type="dataset", exists_ok=True)
    return result


def publish_run(cfg: dict, log, commit: str, name: str) -> None:
    """Upload adapter and full resume checkpoint, then write a publication receipt."""
    api, collection = hub(cfg)
    directory = (runs_dir() / name).resolve()
    if directory.parent != runs_dir().resolve():
        raise ValueError("Run name must identify a direct child of the run directory")
    if not (directory / "status.json").exists():
        evaluations = [(read_json(path), read_jsonl(path.parent / "results.jsonl"))
                       for path in sorted(directory.glob("eval-*/manifest.json"))]
        if not evaluations or any(m["arm"] not in PROMPT_ARMS for m, _ in evaluations):
            raise RuntimeError("Missing training state; only prompt-only evaluation runs omit an adapter")
        first = evaluations[0][0]
        if any((m["arm"], m["seed"]) != (first["arm"], first["seed"]) for m, _ in evaluations):
            raise RuntimeError("Mixed evaluation identities in one run directory")
        complete = all({r["task_id"] for r in rows} == set(m["tasks"]) and len(rows) == len(m["tasks"])
                       for m, rows in evaluations)
        result = publish_evaluations(api, cfg, collection, directory)
        receipt = {"arm": first["arm"], "seed": first["seed"], "complete": complete, "artifact_type": "evaluation",
                   "hf_commit": result.oid, "repo": cfg["hf_artifacts_repo"], "git_commit": commit, "collection": collection}
        write_json(ROOT / "manifests" / f"published-{name}.json", receipt)
        log.emit("run_published", **receipt)
        return
    status = read_json(directory / "status.json")
    repo = f"BurnyCoder/occam-rl-{name}"
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    card = f"""---
license: apache-2.0
base_model: {cfg['policy']}
library_name: peft
tags:
- reinforcement-learning
- grpo
- lora
---
# {name}

Research author: Libor “Burny” Burian. This is a local LoRA research artifact, not a validated general-purpose model.

Reward arm: {status['arm']}. Training seed: {status['seed']}. Completed steps: {status['steps']}.
Run complete: {status['complete']}. Exploratory pilot: {status['pilot']}.
Changed trainable parameter tensors: {status.get('changed_parameter_tensors', 'not recorded')}.
Training judge qualified: {status.get('judge_qualified', 'not recorded')}.
Exploratory judge-quality waiver: {status.get('exploratory_judge_override', False)}.

The full implementation, prerequisites, data provenance, uncertainty, and limitations are in
https://github.com/{cfg['github_repo']} . No advantage over any baseline is implied by this upload.
Code and adapters retain their applicable Apache-2.0 licenses. Dataset rights and attribution
are described in the research repository; raw benchmark records are separate artifacts.
"""
    from .reporting import score_table, snapshot
    card += "\n## Evaluations\n\n" + score_table([r for r in snapshot()["evaluations"] if r["run"] == name]) + "\n"
    (directory / "adapter/README.md").write_text(card, encoding="utf-8")
    result = api.upload_folder(repo_id=repo, folder_path=str(directory / "adapter"), commit_message=f"Publish {name} adapter")
    # Checkpoints preserve optimizer/scheduler/RNG, while run metadata records their meaning.
    api.upload_folder(repo_id=repo, folder_path=str(directory), path_in_repo="training-state",
                      allow_patterns=["checkpoint-*/**", "run.json", "status.json", "notes.md"],
                      commit_message=f"Publish resumable state for {name}")
    api.add_collection_item(collection, item_id=repo, item_type="model", exists_ok=True)
    publish_evaluations(api, cfg, collection, directory)
    receipt = {"arm": status["arm"], "seed": status["seed"], "complete": status["complete"],
               "pilot": status["pilot"], "hf_commit": result.oid, "repo": repo, "git_commit": commit, "collection": collection}
    write_json(ROOT / "manifests" / f"published-{name}.json", receipt)
    if status["arm"] == "occam" and status["seed"] == 17 and status["complete"] and not status["pilot"]:
        write_json(ROOT / "manifests/first-occam.json", receipt)
    log.emit("run_published", **receipt)
