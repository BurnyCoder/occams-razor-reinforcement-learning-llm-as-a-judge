"""Verify current public bytes; this does not establish historical push times.

Uses the Hub's documented repository API and SHA-256 LFS identities:
https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
Run after source/card upload and Pages deployment, from a clean pushed checkout.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import HfApi

from occam_rl.publication import assert_published


def main():
    root = Path(__file__).resolve().parents[1]
    commit = assert_published()
    cfg = json.loads((root / "configs/experiment.json").read_text())
    audit = json.loads((root / "results/independent-fact-check.json").read_text())
    receipt = json.loads((root / ".local/hf-publication.json").read_text())
    if receipt["git_commit"] != commit:
        raise RuntimeError("Upload the current source before verification")
    checks = []

    def fetch(url):
        with urlopen(Request(url, headers={"User-Agent": "occam-research-verification"}), timeout=60) as response:
            return response.read()

    def compare(url, expected, surface):
        actual = fetch(url)
        if actual != expected:
            raise RuntimeError(f"Published bytes differ: {url}")
        checks.append(dict(surface=surface, url=url, sha256=hashlib.sha256(actual).hexdigest()))

    pages = "https://burnycoder.github.io/occams-razor-reinforcement-learning-llm-as-a-judge/"
    for path in sorted((root / "site").rglob("*")):
        if path.is_file():
            name = path.relative_to(root).as_posix()
            expected = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
            compare(pages + path.relative_to(root / "site").as_posix(), expected, "pages")

    dataset = cfg["hf_artifacts_repo"]
    hub_root = f"https://huggingface.co/datasets/{dataset}/resolve/main/"
    files = {"README.md": "docs/hf-artifacts.md", "research-README.md": "README.md"}
    for name in ("paper/paper.md", "output/pdf/paper.pdf", "reports/independent-fact-check.md",
                 "reports/claim-register.json", "reports/amendment-004-scoring.md", "configs/physics-review.psv",
                 "results/scientific-audit.json", "reports/reviews/2026-11-26-scientific-audit.md",
                 "reports/summary.md", "results/tests/scientific-cpu.xml", "results/summary.json", "site/data.json"):
        files[name] = name
    for remote, local in files.items():
        compare(hub_root + remote, (root / local).read_bytes(), "huggingface-source")
    for row in audit["frozen"]["files"]:
        url = hub_root + "frozen-data/" + row["file"]
        actual = hashlib.sha256(fetch(url)).hexdigest()
        if actual != row["sha256"]:
            raise RuntimeError(f"Published frozen data differ: {url}")
        checks.append(dict(surface="frozen-data", url=url, sha256=actual))

    model = "BurnyCoder/occam-rl-occam-seed17-pilot"
    compare(f"https://huggingface.co/{model}/resolve/main/README.md",
            (root / "docs/model-cards/occam-seed17-pilot.md").read_bytes(), "model-card")
    api = HfApi()
    checkpoint_path = "training-state/checkpoint-4"
    tree = {item.path: item for item in api.list_repo_tree(model, path_in_repo=checkpoint_path, recursive=True)}
    for name, expected in audit["preserved"]["checkpoint_sha256"].items():
        item = tree[checkpoint_path + "/" + name]
        url = f"https://huggingface.co/{model}/resolve/main/{checkpoint_path}/{name}"
        actual = item.lfs.sha256 if item.lfs else hashlib.sha256(fetch(url)).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Published checkpoint differs: {name}")
        checks.append(dict(surface="checkpoint", url=url, sha256=actual))
    collection = api.get_collection(receipt["collection"])
    items = {item.item_id for item in collection.items}
    if not {dataset, model} <= items:
        raise RuntimeError("Existing research collection is missing an artifact repository")
    result = dict(verified_at=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                  hf_source_commit=receipt["hf_commit"], collection=collection.slug,
                  collection_items=sorted(items), checks=checks)
    # Keep the live receipt outside Git: including it in its own source snapshot
    # would create a circular commit identity. A dated copy may be published later.
    destination = root / ".local/publication-verification.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified_commit": commit, "checks": len(checks), "receipt": str(destination)}))


if __name__ == "__main__":
    main()
