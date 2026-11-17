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
    return {"audits": audits, "data_counts": [dict(split=s, domain=d, source=source, count=n)
             for (s, d, source), n in sorted(counts.items())]}


def audit_table(audits: list[dict]) -> str:
    """Use raw counts and declared decisions, distinct from held-out model scores."""
    lines = ["| Judge | Dominance | Omission rejection | Invalid JSON | Qualified |",
             "|---|---:|---:|---:|---|"]
    for row in audits:
        model = row["model"].split("/")[-1]
        lines.append(f"| {model} | {row['wins']}/{row['pair_count']} | {row['rejected']}/{row['omission_count']} | {row['invalid']} | {'Yes' if row['passed'] else 'No'} |")
    if not audits:
        lines.append("| Pending | — | — | — | Pending |")
    return "\n".join(lines)


def write_reports(data: dict) -> str:
    """Regenerate the prerequisite report and one report per completed judge audit."""
    table = audit_table(data["audits"])
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
