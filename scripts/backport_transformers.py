"""Backport only the upstream lazy distribution lookup to the locked Transformers.

Apache-2.0 upstream implementation and rationale:
https://github.com/huggingface/transformers/blob/41f9d3992023970fb5a463e70d56232506c80fbc/src/transformers/utils/import_utils.py
Python 3.12 otherwise stats every wheel file during import, very slowly over WSL.
The upstream copyright/license header stays intact. No model math is changed.
"""

import ast
import importlib.metadata

from occam_rl.io import ROOT, file_hash, read_json, write_json
from occam_rl.publication import assert_published


def lazy_imports(text):
    """Reuse upstream's direct version lookup before its deferred mapping fallback."""
    text = text.replace("PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()",
                        "@functools.cache\ndef _package_distribution_mapping() -> dict[str, list[str]]:\n    return importlib.metadata.packages_distributions()")
    text = text.replace("PACKAGE_DISTRIBUTION_MAPPING.get", "_package_distribution_mapping().get")
    start = text.index("        for distribution_name in _candidate_distribution_names(pkg_name):")
    end = text.index("        logger.debug", start)
    fallback = text[start:end]
    text = text[:start] + "        try:\n            package_version = importlib.metadata.version(pkg_name)\n        except importlib.metadata.PackageNotFoundError:\n" + "".join("    " + line for line in fallback.splitlines(keepends=True)) + text[end:]
    return text


def main():
    """Replace only project venv links, including upstream's dependent FA import."""
    assert_published()
    if importlib.metadata.version("transformers") != "5.17.0":
        raise RuntimeError("Review this backport before changing Transformers")
    receipt = ROOT / "manifests/runtime-patch.json"
    previous = read_json(receipt) if receipt.exists() else {}
    prior_files = previous.get("files", {})
    if "patched_sha256" in previous:
        prior_files["utils/import_utils.py"] = {k: previous[k] for k in ("original_sha256", "patched_sha256")}
    expected = {"utils/import_utils.py": "e147f99cf68352e3260e9780930501c3becc0c40a5ff04150ff641b724161732",
                "modeling_flash_attention_utils.py": "b5301ec90664cfe088dc5bd91cb4851cc8affb5ac44e2a1c1c422ba20fa99c11"}
    files = {}
    for name, original in expected.items():
        target = ROOT / ".venv/lib/python3.12/site-packages/transformers" / name
        if not target.parent.resolve().is_relative_to(ROOT / ".venv"):
            raise RuntimeError("Refusing to patch outside the project venv")
        current = file_hash(target)
        text = target.read_text()
        before = text
        if current != prior_files.get(name, {}).get("patched_sha256"):
            if current != original:
                raise RuntimeError(f"Unexpected source bytes: {name}")
            text = lazy_imports(text) if name.startswith("utils/") else text.replace(
                "import PACKAGE_DISTRIBUTION_MAPPING,", "import _package_distribution_mapping,").replace(
                "PACKAGE_DISTRIBUTION_MAPPING.get", "_package_distribution_mapping().get")
        # Migrate earlier partial receipts using the same upstream substitutions.
        text = text.replace("PACKAGE_DISTRIBUTION_MAPPING.get", "_package_distribution_mapping().get")
        if "PACKAGE_DISTRIBUTION_MAPPING" in text:
            raise RuntimeError(f"An eager mapping reference remains in {name}")
        ast.parse(text)
        if text != before:
            temporary = target.with_suffix(".occam.tmp")
            temporary.write_text(text)
            temporary.replace(target)
        files[name] = {"original_sha256": original, "patched_sha256": file_hash(target)}
    write_json(receipt, {"package": "transformers", "version": "5.17.0", "files": files,
                        "scope": "lazy package metadata lookup and its existing callers only",
                        "upstream_commit": "41f9d3992023970fb5a463e70d56232506c80fbc"})
    print("Installed hash-checked upstream import backport; publish its manifest before model execution", flush=True)


if __name__ == "__main__":
    main()
