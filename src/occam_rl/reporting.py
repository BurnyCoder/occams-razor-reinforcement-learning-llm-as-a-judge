"""Regenerate measured tables, plots, paper, and site data from immutable run artifacts.

ReportLab authoring reference: https://docs.reportlab.com/reportlab/userguide/
Missing results remain missing; an empty directory never becomes a zero score.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from .io import ROOT, now, read_json, read_jsonl, runs_dir, write_json
from .schema import DOMAINS, PROMPT_ARMS, TRAINED_ARMS
from .statistics import holm, paired_effect

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def snapshot() -> dict:
    """A shared data model powers the static explorer and live local dashboard."""
    runs, evaluations = [], []
    for directory in sorted(runs_dir().glob("*")):
        if not directory.is_dir():
            continue
        status = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        events = read_jsonl(directory / "events.jsonl", tolerate_tail=True)
        rollouts = [r for r in events if r["event"] == "rollout"]
        metrics = [r for r in events if r["event"] == "training_metrics"]
        if events or status:
            by_step = defaultdict(list)
            for row in rollouts:
                by_step[row["step"]].append(row)
            curve = [{"step": step, "accuracy": statistics.mean(r["correct"] for r in rows),
                      "reward": statistics.mean(r["reward"] for r in rows),
                      "tokens": statistics.mean(r["tokens"] for r in rows)} for step,rows in sorted(by_step.items())]
            runs.append({"name": directory.name, "status": status, "curve": curve, "metrics": metrics,
                         "last_event": events[-1]["time"] if events else None,
                         "rollouts": len(rollouts), "training_accuracy": statistics.mean(r["correct"] for r in rollouts) if rollouts else None})
        for path in sorted(directory.glob("eval-*/results.jsonl")):
            manifest = read_json(path.parent / "manifest.json")
            rows = read_jsonl(path)
            for domain in ("all", *DOMAINS):
                selected = rows if domain == "all" else [r for r in rows if r["domain"] == domain]
                if not selected:
                    continue
                # Equal-domain macro weighting for the overall row.
                groups = [[r for r in selected if r["domain"] == d] for d in DOMAINS] if domain == "all" else [selected]
                groups = [group for group in groups if group]
                record = {"run": directory.name, "split": manifest["split"], "domain": domain,
                          "arm": manifest["arm"], "seed": manifest["seed"], "n": len(selected),
                          "complete": len(rows) == len(manifest["tasks"]), "exploratory": manifest["limit_per_domain"] is not None}
                for key in ("correct", "adequate", "simplicity", "quality", "tokens", "formal_length", "seconds", "truncated"):
                    record[key] = statistics.mean(statistics.mean(r[key] for r in group) for group in groups)
                evaluations.append(record)
    return {"generated_at": now(), "runs": runs, "evaluations": evaluations}


def comparisons(cfg: dict) -> dict:
    """Only complete, un-limited test evaluations can enter primary paired analysis."""
    arms = defaultdict(list)
    for path in runs_dir().glob("*-seed*/eval-test/results.jsonl"):
        manifest = read_json(path.parent / "manifest.json")
        rows = read_jsonl(path)
        if len(rows) == len(manifest["tasks"]) and manifest["limit_per_domain"] is None:
            arms[manifest["arm"]].extend(rows)
    result = {"status": "pending", "comparisons": {}}
    if not all(arm in arms for arm in ("occam", "accuracy", "token", "formal")):
        return result
    common_seeds = set(cfg["seeds"])
    for arm in ("occam", "accuracy", "token", "formal"):
        common_seeds &= {r["seed"] for r in arms[arm]}
    if not common_seeds:
        return result
    arm_rows = {arm: [r for r in rows if r["seed"] in common_seeds] for arm,rows in arms.items()}
    primary = []
    for baseline in ("token", "formal"):
        effect = paired_effect(arm_rows["occam"], arm_rows[baseline], "quality",
                               replicates=cfg["bootstrap_replicates"], seed=cfg["bootstrap_seed"])
        effect["seed_sensitivity"] = paired_effect(arm_rows["occam"], arm_rows[baseline], "quality",
            replicates=cfg["bootstrap_replicates"], seed=cfg["bootstrap_seed"], resample_seeds=True)
        result["comparisons"][baseline] = effect
        primary.append(effect["p_one_sided"])
    for baseline, adjusted in zip(("token", "formal"), holm(primary), strict=True):
        result["comparisons"][baseline]["p_holm"] = adjusted
    accuracy = {}
    for domain in ("all", *DOMAINS):
        left = arm_rows["occam"] if domain == "all" else [r for r in arm_rows["occam"] if r["domain"] == domain]
        right = arm_rows["accuracy"] if domain == "all" else [r for r in arm_rows["accuracy"] if r["domain"] == domain]
        accuracy[domain] = paired_effect(left, right, "correct", replicates=cfg["bootstrap_replicates"], seed=cfg["bootstrap_seed"])
    result["accuracy_noninferiority"] = accuracy
    quality_pass = all(r["ci95"][0] > 0 and r["p_holm"] < .05 for r in result["comparisons"].values())
    accuracy_pass = all(r["ci95"][0] > -.02 for r in accuracy.values())
    result["status"] = "criteria_met" if quality_pass and accuracy_pass else "not_established"
    result["seeds"] = sorted(common_seeds)
    result["confirmatory_seed_target_met"] = set(common_seeds) == set(cfg["seeds"])
    return result


def score_table(rows: list[dict]) -> str:
    """All displayed measured numbers are formatted here from result records."""
    header = "| Run | Split | N | Accuracy | Adequacy | C × A × S | Tokens | Status |\n|---|---|---:|---:|---:|---:|---:|---|"
    lines = []
    for row in rows:
        if row["domain"] != "all":
            continue
        status = "exploratory" if row["exploratory"] else "complete" if row["complete"] else "partial"
        lines.append(f"| {row['run']} | {row['split']} | {row['n']} | {row['correct']:.3f} | {row['adequate']:.3f} | {row['quality']:.3f} | {row['tokens']:.1f} | {status} |")
    if not lines:
        lines = [f"| {arm} | pending | — | — | — | — | — | not evaluated |" for arm in (*TRAINED_ARMS, *PROMPT_ARMS)]
    return header + "\n" + "\n".join(lines)


def figures(data: dict) -> list[Path]:
    """Use zero-based accuracy axes; never synthesize points or silently smooth curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = ROOT / "site/figures"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in data["runs"]:
        if not run["curve"]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax,key,label in zip(axes, ("accuracy", "reward", "tokens"), ("Training rollout accuracy", "Training reward", "Generated tokens"), strict=True):
            ax.plot([r["step"] for r in run["curve"]], [r[key] for r in run["curve"]], marker=".")
            ax.set(xlabel="Optimizer step", ylabel=label)
            ax.set_ylim(bottom=0, top=1 if key == "accuracy" else None)
            ax.grid(alpha=.2)
        fig.suptitle(run["name"] + " — individual rollout groups, no held-out claim")
        fig.tight_layout()
        path = target / f"{run['name']}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def examples() -> list[dict]:
    """Select by task hash order, not favorable scores; retain complete traces."""
    records = []
    for path in sorted(runs_dir().glob("*-seed*/eval-*/results.jsonl")):
        rows = sorted(read_jsonl(path), key=lambda r: r["task_id"])
        for domain in DOMAINS:
            records.extend(r for r in [x for x in rows if x["domain"] == domain][:3])
    return records


def build_pdf(markdown: str, paths: list[Path]) -> None:
    """Render a portable paper with readable paragraphs/tables and linked references."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize, styles["BodyText"].leading = 9, 13
    styles["Title"].fontSize = 20
    story, table_rows = [], []

    def flush_table():
        if table_rows:
            widths = [110, 55, 30, 55, 55, 60, 55, 65] if len(table_rows[0]) == 8 else None
            table = Table([[Paragraph(escape(cell.strip()).replace("—", "pending"), styles["BodyText"]) for cell in row] for row in table_rows], repeatRows=1, colWidths=widths)
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5eee8")),
                                      ("VALIGN", (0,0), (-1,-1), "TOP"), ("GRID", (0,0), (-1,-1), .3, colors.lightgrey)]))
            story.extend([table, Spacer(1, 10)])
            table_rows.clear()
    for line in markdown.splitlines():
        if line.startswith("|"):
            if not set(line.replace("|", "").replace(" ", "")) <= set("-:"):
                table_rows.append(line.strip("|").split("|"))
            continue
        flush_table()
        if not line.strip():
            continue
        style = "Title" if line.startswith("# ") else "Heading2" if line.startswith("## ") else "Heading3" if line.startswith("### ") else "BodyText"
        text = escape(line.lstrip("# ")).replace("“", '"').replace("”", '"').replace("’", "'").replace("—", " - ").replace("×", " x ")
        story.extend([Paragraph(text, styles[style]), Spacer(1, 5)])
    flush_table()
    for path in paths:
        story.append(Image(str(path), width=6.5*inch, height=1.84*inch))
    destination = ROOT / "paper/paper.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(destination), pagesize=(8.5*inch, 11*inch), rightMargin=.6*inch,
                      leftMargin=.6*inch, topMargin=.6*inch, bottomMargin=.6*inch,
                      title="Adequate Simplicity under Local GRPO", author='Libor "Burny" Burian').build(story)


def report(cfg: dict, log) -> None:
    """One command regenerates measured artifacts; publish outputs before the next run."""
    data = snapshot()
    data["analysis"] = comparisons(cfg)
    data["examples"] = examples()
    write_json(ROOT / "site/data.json", data)
    write_json(ROOT / "results/summary.json", data)
    table = score_table(data["evaluations"])
    summary = "# Experiment summary\n\nGenerated from run artifacts at " + data["generated_at"] + ".\n\n" + table
    summary += "\n\nAccuracy is the fraction verified correct. Adequacy asks whether necessary reasoning is present. Simplicity summarizes four judge dimensions. C × A × S is zero for wrong or inadequate solutions. Training rollout accuracy is not held-out accuracy.\n"
    summary += "\nPrimary analysis status: `" + data["analysis"]["status"] + "`. See results/summary.json for paired intervals, multiplicity adjustments, and seed sensitivity when available.\n"
    (ROOT / "reports/experiments.md").write_text(summary, encoding="utf-8")
    for run in data["runs"]:
        notes_path = runs_dir() / run["name"] / "notes.md"
        notes = notes_path.read_text() if notes_path.exists() else "No investigator notes recorded yet."
        text = f"# {run['name']}\n\n" + json.dumps(run["status"], indent=2) + "\n\n" + notes
        text += "\n\n" + score_table([r for r in data["evaluations"] if r["run"] == run["name"]])
        (ROOT / "reports" / f"experiment-{run['name']}.md").write_text(text, encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generated = START + "\n\n" + table + "\n\n" + END
    if START in readme:
        readme = readme[:readme.index(START)] + generated + readme[readme.index(END)+len(END):]
    else:
        readme += "\n\n## Generated results\n\n" + generated + "\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    paper = (ROOT / "paper/template.md").read_text(encoding="utf-8").replace("{{RESULTS}}", table)
    paper = paper.replace("{{ANALYSIS_STATUS}}", data["analysis"]["status"]).replace("{{GENERATED_AT}}", data["generated_at"])
    (ROOT / "paper/paper.md").write_text(paper, encoding="utf-8")
    paths = figures(data)
    build_pdf(paper, paths)
    log.emit("report_built", evaluations=len(data["evaluations"]), runs=len(data["runs"]), analysis=data["analysis"]["status"])
