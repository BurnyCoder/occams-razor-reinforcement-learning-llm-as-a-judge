"""Regenerate measured tables, plots, paper, and site data from immutable run artifacts.

ReportLab authoring reference: https://docs.reportlab.com/reportlab/userguide/
Missing results remain missing; an empty directory never becomes a zero score.
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from .io import ROOT, now, read_json, read_jsonl, runs_dir, write_json
from .schema import DOMAINS, PROMPT_ARMS, TRAINED_ARMS
from .readiness import collect, write_reports
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
        groups = [r for r in events if r["event"] == "group"]
        if events or status:
            by_step = defaultdict(list)
            for row in rollouts:
                by_step[row["step"]].append(row)
            curve = [{"step": step, "accuracy": statistics.mean(r["correct"] for r in rows),
                      "reward": statistics.mean(r["reward"] for r in rows),
                      "tokens": statistics.mean(r["tokens"] for r in rows)} for step,rows in sorted(by_step.items())]
            runs.append({"name": directory.name, "status": status, "curve": curve, "metrics": metrics,
                         "last_event": events[-1]["time"] if events else None,
                         "training_correct": sum(r["correct"] for r in rollouts),
                         "training_truncated": sum(r["truncated"] for r in rollouts),
                         "groups": len(groups), "nonzero_reward_groups": sum(r["reward_spread"] > 0 for r in groups),
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
                          "complete": len(rows) == len(manifest["tasks"]),
                          "exploratory": manifest["limit_per_domain"] is not None or manifest.get("exploratory_judge_override", False)}
                for key in ("correct", "adequate", "simplicity", "quality", "tokens", "formal_length", "seconds", "truncated"):
                    record[key] = statistics.mean(statistics.mean(r[key] for r in group) for group in groups)
                evaluations.append(record)
    return {"generated_at": now(), "runs": runs, "evaluations": evaluations, "readiness": collect()}


def comparisons(cfg: dict) -> dict:
    """Only complete, un-limited test evaluations can enter primary paired analysis."""
    arms = defaultdict(list)
    judge_override = False
    for path in runs_dir().glob("*-seed*/eval-test/results.jsonl"):
        manifest = read_json(path.parent / "manifest.json")
        rows = read_jsonl(path)
        if len(rows) == len(manifest["tasks"]) and manifest["limit_per_domain"] is None:
            arms[manifest["arm"]].extend(rows)
            judge_override |= manifest.get("exploratory_judge_override", False)
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
    if not result["confirmatory_seed_target_met"]:
        result["status"] = "exploratory_incomplete_seeds"
    if judge_override:
        result["status"] = "exploratory_judge_override"
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


def training_table(runs: list[dict]) -> str:
    """Expose failed learning signals without treating training samples as evaluations."""
    lines = []
    for run in runs:
        if not run["rollouts"]:
            continue
        status = run["status"]
        steps = status.get("steps", max((r["step"] for r in run["metrics"]), default=0))
        state = "in progress" if not status else "interrupted"
        if status.get("complete"):
            state = "complete pilot" if status.get("pilot") else "complete"
        changed = status.get("changed_parameter_tensors", "—")
        lines.append(f"| {run['name']} | {steps} | {run['training_correct']} / {run['rollouts']} | "
                     f"{run['training_truncated']} | {run['nonzero_reward_groups']} / {run['groups']} | {changed} | {state} |")
    if not lines:
        return "No training rollout group has completed."
    return ("| Run | Updates | Correct / traces | Truncated | Groups with reward spread | Changed tensors | State |\n"
            "|---|---:|---:|---:|---:|---:|---|\n" + "\n".join(lines))


def figures(data: dict) -> list[Path]:
    """Use zero-based accuracy axes; never synthesize points or silently smooth curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = ROOT / "site/figures"
    target.mkdir(parents=True, exist_ok=True)
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set(xlim=(0, 10), ylim=(0, 4.8))
    ax.axis("off")
    blocks = [(0.2, 2.9, "Frozen task\n+ local policy"), (3.7, 2.9, "Verified answer\n+ sufficient reasoning"),
              (7.2, 2.9, "Occam or length\nsimplicity score"), (7.2, .7, "Correctness +\ngated bonus"),
              (3.7, .7, "GRPO update\nLoRA policy"), (.2, .7, "Held-out tasks\nGemma evaluation")]
    for x,y,label in blocks:
        ax.add_patch(FancyBboxPatch((x,y), 2.5, 1.1, boxstyle="round,pad=.08", edgecolor="#286b52", facecolor="#eef3e9"))
        ax.text(x+1.25, y+.55, label, ha="center", va="center", fontsize=11)
    for start,end in [((2.8,3.45),(3.55,3.45)),((6.3,3.45),(7.05,3.45)),((8.45,2.75),(8.45,1.95)),((7.05,1.25),(6.35,1.25)),((3.55,1.25),(2.8,1.25))]:
        ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","color":"#286b52"})
    ax.text(5,4.45,"Adequate simplicity: separate correctness, reasoning and economy",ha="center",fontsize=13)
    ax.text(5,.15,"Policy and judges occupy the GPU sequentially; all generated reasoning is retained.",ha="center",fontsize=10)
    architecture = target / "architecture.png"
    fig.savefig(architecture, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths = [architecture]
    for run in data["runs"]:
        if not run["curve"]:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax,key,label in zip(axes.flat, ("accuracy", "reward", "tokens"), ("Training rollout accuracy", "Training reward", "Generated tokens")):
            ax.plot([r["step"] for r in run["curve"]], [r[key] for r in run["curve"]], marker=".")
            ax.set(xlabel="Optimizer step", ylabel=label)
            ax.set_ylim(bottom=0, top=1 if key == "accuracy" else None)
            ax.grid(alpha=.2)
        loss = [r for r in run["metrics"] if "loss" in r]
        axes[1,1].plot([r["step"] for r in loss], [r["loss"] for r in loss], marker=".")
        axes[1,1].set(xlabel="Optimizer step", ylabel="Optimization loss")
        axes[1,1].grid(alpha=.2)
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
            table = Table([[Paragraph(escape(cell.strip()).replace("—", "-"), styles["BodyText"]) for cell in row] for row in table_rows], repeatRows=1, colWidths=widths)
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5eee8")),
                                      ("VALIGN", (0,0), (-1,-1), "TOP"), ("GRID", (0,0), (-1,-1), .3, colors.lightgrey)]))
            story.extend([table, Spacer(1, 10)])
            table_rows.clear()
    main_text, _, ai_statement = markdown.partition("\n## AI statement\n")
    for line in main_text.splitlines():
        if line.startswith("|"):
            if not set(line.replace("|", "").replace(" ", "")) <= set("-:"):
                table_rows.append(line.strip("|").split("|"))
            continue
        flush_table()
        if not line.strip():
            continue
        style = "Title" if line.startswith("# ") else "Heading2" if line.startswith("## ") else "Heading3" if line.startswith("### ") else "BodyText"
        text = escape(line.lstrip("# ")).replace("“", '"').replace("”", '"').replace("’", "'").replace("—", " - ").replace("×", " x ")
        text = re.sub(r"https?://[^\s<]+", lambda match: f'<link href="{match[0]}" color="#245f49">{match[0]}</link>', text)
        # Carry a heading through its following spacer so its table/text stays attached.
        gap = Spacer(1, 5)
        gap.keepWithNext = style.startswith("Heading") or style == "Title"
        paragraph = Paragraph(text, styles[style])
        paragraph.keepWithNext = gap.keepWithNext
        story.extend([paragraph, gap])
    flush_table()
    for path in paths:
        from reportlab.lib.utils import ImageReader
        width, height = ImageReader(str(path)).getSize()
        story.append(Image(str(path), width=6.5*inch, height=6.5*inch*height/width))
    if ai_statement:
        story.extend([Spacer(1, 12), Paragraph("AI statement", styles["Heading2"]),
                      Paragraph(escape(ai_statement.strip()), styles["BodyText"])])
    destination = ROOT / "output/pdf/paper.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, document):
        """Keep page identity visible when the living report is printed or excerpted."""
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(.6*inch, .3*inch, "Adequate Simplicity under Local GRPO")
        canvas.drawRightString(7.9*inch, .3*inch, str(document.page))
        canvas.restoreState()
    SimpleDocTemplate(str(destination), pagesize=(8.5*inch, 11*inch), rightMargin=.6*inch,
                      leftMargin=.6*inch, topMargin=.6*inch, bottomMargin=.6*inch,
                      title="Adequate Simplicity under Local GRPO", author='Libor "Burny" Burian').build(story, onFirstPage=footer, onLaterPages=footer)


def report(cfg: dict, log) -> None:
    """One command regenerates measured artifacts; publish outputs before the next run."""
    data = snapshot()
    readiness_table = write_reports(data["readiness"])
    data["analysis"] = comparisons(cfg)
    data["examples"] = examples()
    write_json(ROOT / "site/data.json", data)
    write_json(ROOT / "results/summary.json", data)
    table = score_table(data["evaluations"])
    training = training_table(data["runs"])
    summary = "# Experiment summary\n\nGenerated from run artifacts at " + data["generated_at"] + ".\n\n" + table
    summary += "\n\n## Training diagnostics\n\n" + training + "\n"
    summary += "\n\nAccuracy is the fraction verified correct. Adequacy asks whether necessary reasoning is present. Simplicity summarizes four judge dimensions. C × A × S is zero for wrong or inadequate solutions. Training rollout accuracy is not held-out accuracy.\n"
    summary += "\nPrimary analysis status: `" + data["analysis"]["status"] + "`. See results/summary.json for paired intervals, multiplicity adjustments, and seed sensitivity when available.\n"
    summary += "\n## Prerequisite audits\n\n" + readiness_table + "\n\nSee [readiness evidence](readiness.md) for data counts and individual audit reports.\n"
    (ROOT / "reports/experiments.md").write_text(summary, encoding="utf-8")
    for run in data["runs"]:
        notes_path = runs_dir() / run["name"] / "notes.md"
        notes = notes_path.read_text() if notes_path.exists() else "No investigator notes recorded yet."
        text = f"# {run['name']}\n\n" + json.dumps(run["status"], indent=2) + "\n\n" + notes
        text += "\n\n## Training diagnostics\n\n" + training_table([run])
        text += "\n\n" + score_table([r for r in data["evaluations"] if r["run"] == run["name"]])
        (ROOT / "reports" / f"experiment-{run['name']}.md").write_text(text, encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generated = START + "\n\n### Training diagnostics\n\n" + training + "\n\n### Held-out evaluation\n\n" + table + "\n\n" + END
    if START in readme:
        readme = readme[:readme.index(START)] + generated + readme[readme.index(END)+len(END):]
    else:
        readme += "\n\n## Generated results\n\n" + generated + "\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    paper = (ROOT / "paper/template.md").read_text(encoding="utf-8").replace("{{RESULTS}}", table)
    paper = paper.replace("{{ANALYSIS_STATUS}}", data["analysis"]["status"]).replace("{{GENERATED_AT}}", data["generated_at"])
    paper = paper.replace("{{READINESS}}", readiness_table)
    paper = paper.replace("{{TRAINING}}", training)
    (ROOT / "paper/paper.md").write_text(paper, encoding="utf-8")
    paths = figures(data)
    build_pdf(paper, paths)
    log.emit("report_built", evaluations=len(data["evaluations"]), runs=len(data["runs"]), analysis=data["analysis"]["status"])
