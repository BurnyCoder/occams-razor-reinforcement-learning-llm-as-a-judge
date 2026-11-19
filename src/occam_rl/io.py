"""Artifact IO, hashing, UTC logs, and configuration without credential logging."""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    """Use sortable UTC timestamps across Windows and WSL."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(value) -> str:
    """Canonical JSON hashing makes manifests independent of key insertion order."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Stream large artifacts without loading a checkpoint into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    """Read an explicit artifact, raising on missing/corrupt input."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Atomic replacement avoids half-written status after an interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path, *, tolerate_tail: bool = False) -> list[dict]:
    """Only live dashboards may ignore an unfinished final log line."""
    if not path.exists():
        return []
    # JSON Lines uses LF, not Unicode string separators such as GSM8K's U+2028.
    # https://jsonlines.org/ explicitly permits those characters inside JSON strings.
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not tolerate_tail or text.endswith("\n") or index != len(lines) - 1:
                raise
    return rows


def config(path: str | None = None) -> dict:
    """Load public experiment parameters and local paths; never serialize .env."""
    load_dotenv(ROOT / ".env", override=False)
    return read_json(Path(path) if path else ROOT / "configs/experiment.json")


def runs_dir() -> Path:
    """Large checkpoints can live on the user-approved secondary drive."""
    return Path(os.environ.get("OCCAM_RUNS", ROOT / "runs"))


def data_dir() -> Path:
    """Locate task bodies; source publication also uploads the hash-verified JSONL files."""
    return Path(os.environ.get("OCCAM_DATA", ROOT / "data/generated"))


class EventLog:
    """Append complete timestamped JSON events to disk and terminal immediately."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def emit(self, event: str, **fields) -> dict:
        """No truncation or secrets filtering heuristics: callers pass safe fields only."""
        record = {"time": now(), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        with self.lock, self.path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(line + "\n")
            print(line, flush=True)
        return record
