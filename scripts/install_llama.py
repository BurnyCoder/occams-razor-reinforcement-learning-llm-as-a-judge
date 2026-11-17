"""Install checksum-pinned official CUDA binaries when a compiler is unavailable.

Source: https://github.com/ggml-org/llama.cpp/releases . Archive extraction uses
Python's data filter, rejecting traversal and unsafe links. No model is executed.
"""

import tarfile
import urllib.request
from pathlib import Path

from occam_rl.io import ROOT, file_hash, read_json
from occam_rl.publication import assert_published

assert_published()
manifest = read_json(ROOT / "manifests/revisions.json")
destination = Path("/mnt/d/AI/occam-rl/llama-binary") / manifest["llama_release"]
destination.mkdir(parents=True, exist_ok=True)
for artifact in manifest["llama_assets"]:
    archive = destination / artifact["name"]
    if not archive.exists():
        urllib.request.urlretrieve(artifact["url"], archive)
    expected = artifact["digest"].removeprefix("sha256:")
    if file_hash(archive) != expected:
        raise RuntimeError(f"Checksum mismatch for {archive.name}")
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
servers = list(destination.rglob("llama-server"))
if len(servers) != 1:
    raise RuntimeError(f"Expected one llama-server, found {servers}")
print(f"Set OCCAM_LLAMA_SERVER={servers[0]} in ignored .env", flush=True)
libraries = list(destination.rglob("libcudart.so.13"))
if len(libraries) != 1:
    raise RuntimeError(f"Expected one CUDA runtime directory, found {libraries}")
print(f"Set OCCAM_LLAMA_CUDA_LIB={libraries[0].parent} in ignored .env", flush=True)
