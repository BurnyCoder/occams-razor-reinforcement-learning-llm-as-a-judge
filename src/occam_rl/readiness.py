"""Report prerequisite evidence without confusing it with policy performance."""

from collections import Counter

from .io import ROOT, digest, read_json, write_json


def collect() -> dict:
    """Read complete audit artifacts and frozen task counts; never invent missing data."""
    paths = sorted((ROOT / "results/judge-audits").glob("*.json"))
    paths += sorted((ROOT / "manifests").glob("judge-*.json"))
    seen, audits = set(), []
    for path in paths:
        row = read_json(path)
        key = digest(row)
        if key not in seen:
            audits.append(row | {"artifact": path.relative_to(ROOT).as_posix()})
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
    lines = ["| Judge | Dominance | Omission rejection | Invalid JSON | Qualified |",
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
                           ("Role-switch weights unchanged", "switch_weights_unchanged"),
                           ("Role-switch RNG unchanged", "switch_rng_unchanged"),
                           ("Optimizer serialization round trip", "optimizer_roundtrip"),
                           ("Profile passed", "passed")):
            table += f"| {label} | {profile[key]} |\n"
    text = "# Readiness evidence\n\n" + table
    text += "\n\nConstructed specification checks are not held-out policy accuracy or human preference validation. Invalid or contradictory JSON does not count as a successful omission rejection.\n"
    text += "\n| Split | Domain | Source | Tasks |\n|---|---|---|---:|\n"
    for row in data["data_counts"]:
        text += f"| {row['split']} | {row['domain']} | {row['source']} | {row['count']} |\n"
    for index, row in enumerate(data["audits"], 1):
        report = "# Judge audit\n\n" + audit_table([row])
        report += f"\n\nEvidence: [{row['artifact']}](../{row['artifact']}).\n"
        report += f"\nDominance Wilson interval: {row['dominance_ci95']}. Omission Wilson interval: {row['omission_ci95']}. These intervals do not establish population-wide validity from constructed examples.\n"
        report += "\nThe declared qualification decision is retained. The training-only override in amendment 002 permits exploratory rewards after a failed current audit; it does not qualify the judge or waive runtime checks. Independent evaluation still requires a passing audit.\n"
        (ROOT / f"reports/experiment-judge-{index}.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/readiness.md").write_text(text, encoding="utf-8")
    write_json(ROOT / "results/readiness.json", data)
    return table
