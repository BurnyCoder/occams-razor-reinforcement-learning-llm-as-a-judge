"""Report prerequisite evidence without confusing it with policy performance."""

from collections import Counter

from .io import ROOT, digest, read_json, write_json
from .schema import Judgment
from .statistics import wilson


def recount_audit(row: dict) -> dict:
    """Recompute counts from validated judgments; never infer omission detection from null."""
    pairs = [r for r in row["rows"] if r["id"].startswith("audit/pair/")]
    omissions = [r for r in row["rows"] if r["id"].startswith("audit/omission/")]
    wins, rejected, invalid = 0, 0, 0
    for pair in pairs:
        better, worse = [Judgment.model_validate(pair[k]) if pair[k] is not None else None
                         for k in ("better", "worse")]
        wins += bool(better and worse and better.adequate and better.simplicity > worse.simplicity)
        invalid += int(better is None) + int(worse is None)
    for omission in omissions:
        judged = Judgment.model_validate(omission["judgment"]) if omission["judgment"] is not None else None
        rejected += bool(judged and not judged.adequate)
        invalid += judged is None
    measured = dict(wins=wins, pair_count=len(pairs), rejected=rejected,
                    omission_count=len(omissions), invalid=invalid,
                    passed=bool(pairs and omissions and wins/len(pairs) >= .8 and rejected/len(omissions) >= .9))
    if any(row.get(k) != v for k, v in measured.items()):
        raise ValueError("Stored judge summary disagrees with raw judgment rows")
    return row | measured | {"dominance_ci95": wilson(wins, len(pairs)),
                             "omission_ci95": wilson(rejected, len(omissions))}


def collect() -> dict:
    """Read complete audit artifacts and frozen task counts; never invent missing data."""
    paths = sorted((ROOT / "results/judge-audits").glob("*.json"))
    paths += sorted((ROOT / "manifests").glob("judge-*.json"))
    seen, audits = set(), []
    for path in paths:
        row = read_json(path)
        key = digest(row)
        if key not in seen:
            audits.append(recount_audit(row) | {"artifact": path.relative_to(ROOT).as_posix()})
            seen.add(key)
    manifest = ROOT / "manifests/data.json"
    tasks = read_json(manifest)["tasks"] if manifest.exists() else []
    counts = Counter((t["split"], t["domain"], t["source"]) for t in tasks)
    profile_path = ROOT / "manifests/preflight.json"
    return {"audits": audits, "profile": read_json(profile_path) if profile_path.exists() else None,
            "data_counts": [dict(split=s, domain=d, source=source, count=n)
             for (s, d, source), n in sorted(counts.items())]}


def audit_table(audits: list[dict]) -> str:
    """Use raw counts and declared decisions, distinct from held-out model scores."""
    lines = ["| Judge | Dominance | Omission rejection | Invalid judgments | Qualified |",
             "|---|---:|---:|---:|---|"]
    for row in audits:
        model = row["model"].split("/")[-1]
        # Distinguish the current revised audit from earlier attempts of the same model.
        if row.get("audit_file"):
            model += " (" + row["audit_file"].rsplit("/", 1)[-1].removeprefix("judge-audit-").removesuffix(".json") + ")"
        lines.append(f"| {model} | {row['wins']}/{row['pair_count']} | {row['rejected']}/{row['omission_count']} | {row['invalid']} | {'Yes' if row['passed'] else 'No'} |")
    if not audits:
        lines.append("| Pending | — | — | — | Pending |")
    return "\n".join(lines)


def write_reports(data: dict) -> str:
    """Regenerate the prerequisite report and one report per completed judge audit."""
    table = audit_table(data["audits"])
    profile = data.get("profile")
    if profile:
        table += "\n\nGPU/state prerequisite measurements (synthetic inputs; not learning results):\n\n"
        table += "| Check | Recorded result |\n|---|---|\n"
        for label, key in (("Backward minimum free VRAM (MiB)", "backward_minimum_free_mib"),
                           ("Generation minimum free VRAM (MiB)", "generation_minimum_free_mib"),
                           ("Role-switch trainable tensors unchanged", "switch_weights_unchanged"),
                           ("Role-switch CUDA RNG unchanged", "switch_rng_unchanged"),
                           ("Optimizer state-entry count preserved after reload", "optimizer_roundtrip"),
                           ("Profile passed", "passed")):
            table += f"| {label} | {profile[key]} |\n"
    text = "# Readiness evidence\n\n" + table
    text += "\n\nConstructed specification checks are not held-out policy accuracy or human preference validation. Invalid judgments include JSON syntax errors, schema violations and contradictory adequacy/missing-reasoning fields; they do not count as successful omission rejections.\n"
    text += "\nProfile evidence: [recorded measurements](../manifests/preflight.json). The optimizer check compares state-entry counts after loading; it does not compare every tensor or test interrupted training. Role-switch equality covers trainable tensors and the current CUDA RNG, not every model buffer or every RNG.\n"
    text += "\n| Split | Domain | Source | Tasks |\n|---|---|---|---:|\n"
    for row in data["data_counts"]:
        text += f"| {row['split']} | {row['domain']} | {row['source']} | {row['count']} |\n"
    for index, row in enumerate(data["audits"], 1):
        report = "# Judge audit\n\n" + audit_table([row])
        report += f"\n\nEvidence: [{row['artifact']}](../{row['artifact']}).\n"
        report += f"\nDominance Wilson interval: {row['dominance_ci95']}. Omission Wilson interval: {row['omission_ci95']}. These intervals do not establish population-wide validity from constructed examples.\n"
        report += "\nThe [qualification replay](reviews/2026-11-26-scientific-audit.md) reports valid pairs, ties, omission judgments, surface cues and interrupted attempts.\n"
        report += "\nThe declared qualification decision is retained. The training-only override in amendment 002 permits exploratory rewards after a failed current audit; it does not qualify the judge or waive runtime checks. Independent evaluation still requires a passing audit.\n"
        (ROOT / f"reports/experiment-judge-{index}.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/readiness.md").write_text(text, encoding="utf-8")
    write_json(ROOT / "results/readiness.json", data)
    return table
