"""Backport only the upstream lazy distribution lookup to the locked Transformers.

Apache-2.0 upstream implementation and rationale:
https://github.com/huggingface/transformers/blob/41f9d3992023970fb5a463e70d56232506c80fbc/src/transformers/utils/import_utils.py
Python 3.12 otherwise stats every wheel file during import, very slowly over WSL.
The upstream copyright/license header stays intact. No model math is changed.
"""

import ast
import importlib.metadata
from pathlib import Path

from occam_rl.io import ROOT, file_hash, read_json, write_json
from occam_rl.publication import assert_published


def main():
    """Replace only the project venv symlink, never mutate uv's cached wheel."""
    assert_published()
    if importlib.metadata.version("transformers") != "5.17.0":
        raise RuntimeError("Review this backport before changing Transformers")
    target = ROOT / ".venv/lib/python3.12/site-packages/transformers/utils/import_utils.py"
    receipt = ROOT / "manifests/runtime-patch.json"
    original = "e147f99cf68352e3260e9780930501c3becc0c40a5ff04150ff641b724161732"
    current = file_hash(target)
    if receipt.exists() and current == read_json(receipt)["patched_sha256"]:
        print("Published import backport already installed", flush=True)
        return
    if current != original or not target.parent.resolve().is_relative_to(ROOT / ".venv"):
        raise RuntimeError("Unexpected source bytes or symlinked directory; refusing an unreviewed patch")
    text = target.read_text()
    text = text.replace("PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()",
                        "@functools.cache\ndef _package_distribution_mapping() -> dict[str, list[str]]:\n    return importlib.metadata.packages_distributions()")
    text = text.replace("PACKAGE_DISTRIBUTION_MAPPING.get(pkg_name, [])", "_package_distribution_mapping().get(pkg_name, [])")
    start = text.index("        for distribution_name in _candidate_distribution_names(pkg_name):")
    end = text.index("        logger.debug", start)
    fallback = text[start:end]
    text = text[:start] + "        try:\n            package_version = importlib.metadata.version(pkg_name)\n        except importlib.metadata.PackageNotFoundError:\n" + "".join("    " + line for line in fallback.splitlines(keepends=True)) + text[end:]
    ast.parse(text)
    temporary = target.with_suffix(".occam.tmp")
    temporary.write_text(text)
    temporary.replace(target)
    write_json(receipt, {"package": "transformers", "version": "5.17.0", "original_sha256": original,
                        "patched_sha256": file_hash(target), "scope": "lazy package metadata lookup only",
                        "upstream_commit": "41f9d3992023970fb5a463e70d56232506c80fbc"})
    print("Installed hash-checked upstream import backport; publish its manifest before model execution", flush=True)


if __name__ == "__main__":
    main()
