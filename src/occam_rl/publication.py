"""Enforce the user's write/push/verify/execute ordering using actual Git state.

Git commands are argument arrays; no shell interpolation or secret-bearing output.
https://git-scm.com/docs/git-ls-remote
"""

import subprocess
from pathlib import Path

from .io import ROOT, file_hash, read_json


def git(*args: str, cwd: Path = ROOT) -> str:
    """Run a bounded Git command and surface failure without guessing state."""
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, timeout=60).strip()


def assert_published(cwd: Path = ROOT) -> str:
    """A clean tracked tree and an identical remote branch are mandatory."""
    dirty = git("status", "--porcelain", "--untracked-files=all", cwd=cwd)
    if dirty:
        raise RuntimeError("Write/push gate: commit and push all publishable changes first:\n" + dirty)
    branch = git("symbolic-ref", "--short", "HEAD", cwd=cwd)
    commit = git("rev-parse", "HEAD", cwd=cwd)
    remote = git("ls-remote", "--exit-code", "origin", f"refs/heads/{branch}", cwd=cwd)
    if not remote or remote.split()[0] != commit:
        raise RuntimeError("Write/push gate: current HEAD is not the remote branch HEAD")
    return commit


def require_hf_publication(commit: str) -> None:
    """Training also requires the same source snapshot uploaded to Hugging Face."""
    receipt = read_json(ROOT / ".local/hf-publication.json")
    if receipt.get("git_commit") != commit or not receipt.get("hf_commit"):
        raise RuntimeError("Run occam publish --source for this exact Git commit before training")


def require_data_manifest(path: Path) -> dict:
    """The consumed bytes must match the already committed data manifest."""
    manifest = read_json(ROOT / "manifests/data.json")
    if file_hash(path) != manifest["files"][path.name]["sha256"]:
        raise RuntimeError(f"Unpublished or changed data: {path}")
    return manifest


def require_first_occam() -> None:
    """Prompt-only inference and baseline training cannot precede published Occam."""
    marker = read_json(ROOT / "manifests/first-occam.json")
    if (not marker.get("complete") or not marker.get("hf_commit") or marker.get("arm") != "occam"
            or marker.get("seed") != 17 or marker.get("pilot") is not False):
        raise RuntimeError("Complete and publish Occam seed 17 before baseline execution")
