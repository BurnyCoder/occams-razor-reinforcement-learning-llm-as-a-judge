"""Reproduce the independent November 26 audit from primary records without model execution."""

import json
import re
import subprocess
from collections import Counter

from pydantic import ValidationError

from .audit_data import data_audit
from .io import ROOT, digest, file_hash, read_json, read_jsonl, runs_dir, write_json
from .schema import Judgment
from .scoring import VERSION, scoring_identity
from .statistics import wilson


def parse_judgment(text):
    """Reapply the recorded JSON schema after stripping the documented chat terminators."""
    text = re.sub(r"<\|(?:im_end|endoftext|end)\|>|<end_of_turn>|<turn\|>", "", text).strip()
    try:
        return Judgment.model_validate_json(text)
    except ValidationError:
        return None


def judge_audit():
    """Pair original prompt/output events with frozen stimuli and replay the recorded strict parser."""
    suites = {digest(read_json(p)): (p.name, read_json(p)) for p in (ROOT/"configs").glob("judge-audit*.json")}
    saved = {r["audit_hash"]:r for r in (read_json(p) for p in (ROOT/"results/judge-audits").glob("*.json"))}
    attempts, completed = [], set()
    for path in sorted((runs_dir()/"operations").glob("*-qualify.jsonl")):
        rows = read_jsonl(path)
        prompts = [r for r in rows if r["event"] == "llm_prompt"]
        outputs = [r for r in rows if r["event"] == "llm_output"]
        summary = next((r for r in rows if r["event"] == "judge_audit_complete"), None)
        info = dict(log=path.name, log_sha256=file_hash(path), prompts=len(prompts), outputs=len(outputs),
                    complete=summary is not None, end=rows[-1]["event"])
        attempts.append(info)
        if not summary:
            parsed = [parse_judgment(out["trace"]) for out in outputs]
            info.update(valid_outputs=sum(x is not None for x in parsed), invalid=parsed.count(None))
            if len(parsed) == 2 and len(prompts) == 3:
                # This run records only the better and worse responses of the first pair.
                a,b = parsed
                info["complete_first_pair"] = a is not None and b is not None
                info["first_pair_tie"] = a.simplicity == b.simplicity if info["complete_first_pair"] else None
            continue
        key = summary["audit_hash"]
        name, suite = suites[key]
        raw = saved[key]
        expected = [(p,p[side]) for p in suite["pairs"] for side in ("better","worse")]
        expected.extend((p,p["trace"]) for p in suite["omissions"])
        stored = [r[k] for r in raw["rows"] for k in (("better","worse") if "better" in r else ("judgment",))]
        validated = []
        for (case,trace),prompt,out,old in zip(expected,prompts,outputs,stored,strict=True):
            messages = prompt["messages"]
            payload = json.loads(messages[-1]["content"])
            if (payload["problem"],payload["reference"],payload["candidate_full_trace"]) != (case["question"],case["reference"],trace):
                raise ValueError(f"Judge prompt/stimulus mismatch: {path.name}")
            if digest(messages[0]["content"]) != raw["prompt_hash"]:
                raise ValueError("Frozen judge prompt hash differs")
            judgment = parse_judgment(out["trace"])
            if (judgment.model_dump() if judgment else None) != old:
                raise ValueError(f"Raw generation disagrees with qualification row: {path.name}")
            validated.append(judgment)
        size = len(suite["pairs"])
        pairs = list(zip(validated[:2*size:2],validated[1:2*size:2],strict=True))
        valid = [(a,b) for a,b in pairs if a is not None and b is not None]
        omissions = validated[2*size:]
        wins = sum(a.adequate and a.simplicity>b.simplicity for a,b in valid)
        rejects = sum(x is not None and not x.adequate for x in omissions)
        invalid = validated.count(None)
        if (wins,rejects,invalid) != (raw["wins"],raw["rejected"],raw["invalid"]):
            raise ValueError("Replayed qualifications disagree with historical totals")
        info.update(suite=name, model=raw["model"], pairs=size, valid_pairs=len(valid), wins=wins,
                    ties=sum(a.simplicity==b.simplicity for a,b in valid), omissions=len(omissions),
                    valid_omissions=len(omissions)-omissions.count(None), rejected=rejects, invalid=invalid,
                    valid_outputs=len(outputs)-invalid, pass_gate=wins/size>=.8 and rejects/len(omissions)>=.9,
                    dominance_interval=wilson(wins,size), omission_interval=wilson(rejects,len(omissions)),
                    shorter_preferred=sum(len(t["better"])<len(t["worse"]) for t in suite["pairs"]),
                    marked_only_preferred=sum("$" in t["better"] and "$" not in t["worse"] for t in suite["pairs"]))
        completed.add(key)
    if completed != saved.keys():
        raise ValueError("Missing complete raw qualification log")
    return attempts


def prepare_history():
    """Record all attempts; exclusion reason counts refer to logged events before task selection."""
    out = []
    for p in sorted((runs_dir()/"operations").glob("*-prepare.jsonl")):
        rows = read_jsonl(p)
        excludes = [r for r in rows if r["event"]=="exclude"]
        out.append(dict(log=p.name, sha256=file_hash(p), end=rows[-1]["event"],
            exclusions=len(excludes), reasons=dict(Counter(r["reason"] for r in excludes)),
            shortfalls=[r for r in rows if r["event"]=="stratum_shortfall"]))
    return out


def inventory():
    """Cover every tracked path, including nonprose inputs, without equating a path glob with proof."""
    from pathlib import Path
    registry = read_json(ROOT/"reports/claim-register.json")["claims"]
    names = subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
    rows = {}
    for name in names:
        claims = [c["id"] for c in registry if any(Path(name).match(p) for p in c["locations"])]
        if not claims and name not in ("LICENSE","uv.lock",".python-version",".gitignore",".gitattributes"):
            raise ValueError(f"Unmapped tracked file: {name}")
        rows[name] = claims or ["dependency, license or repository mechanics"]
    return rows


def calculate(analysis):
    """Assemble claim evidence independently of previously generated audit verdicts."""
    from .factcheck import pilot_evidence
    data = data_audit()
    trials = judge_audit()
    pilot = pilot_evidence()
    data.update(scorer=VERSION, scorer_hash=scoring_identity(), qualifications=trials, pilot=pilot,
                preparation=prepare_history(), inventory=inventory(), hypotheses=analysis["hypotheses"])
    data["decision_evidence"] = {"frozen_test_sha256": data["files"]["test.jsonl"],
        "version": VERSION, "scorer_sha256": data["scorer_hash"], "eligible_comparisons": analysis.get("evidence",{}),
        "pilot_status": read_json(ROOT/"results/runtime/occam-pilot-thinking-status.json"),
        "independent_qualification_present": (ROOT/"manifests/judge-gemma.json").is_file(),
        "retained_training_status_files": [p.parent.name for p in sorted(runs_dir().glob("*/status.json"))
                                          if not read_json(p).get("pilot", True)]}
    mb = data["mbpp"]
    done = [t for t in trials if t["complete"]]
    data["counts"] = dict(mbpp_tasks=len(mb), with_assert=sum(bool(r["assertions"]) for r in mb),
        with_overlap=sum(bool(r["base_matches"]+r["plus_matches"]) for r in mb),
        scored=sum(r["base_inputs"]+r["plus_inputs"] for r in mb), base_visible=sum(r["base_matches"] for r in mb),
        plus_visible=sum(r["plus_matches"] for r in mb), additional=sum(r["additional_inputs"] for r in mb),
        canonical_pass=sum(r["canonical_pass"] for r in mb), physics=len(data["physics"]), blocked=sum(r["blocked"] for r in data["physics"]),
        missing_original_unit=sum(not r["unit_original"] for r in data["physics"]),
        shorter=sum(r["shorter_preferred"] for r in done), pairs=sum(r["pairs"] for r in done),
        formatting=sum(r["marked_only_preferred"] for r in done), completed=len(done),
        qualified=sum(r["pass_gate"] for r in done))
    return data


def audit_summary(data):
    """Shared prose for the README, paper and model card. All measured numbers come from records."""
    n = data["counts"]
    return (f"The frozen MBPP+ prompts contain assertions in {n['with_assert']}/{n['mbpp_tasks']} tasks. "
            f"Literal example inputs overlap scored inputs in {n['with_overlap']} tasks. "
            f"The physics reference review flags {n['blocked']}/{n['physics']} public targets for adjudication. "
            f"Choosing the shorter candidate selects the designated preference on {n['shorter']}/{n['pairs']} constructed pairs. "
            f"Qualification passed in {n['qualified']}/{n['completed']} completed sets. "
            + " ".join(f"{h} is {r['status']}." for h,r in data["hypotheses"].items()))


def write_audit(data):
    """Publish machine-readable per-task evidence and a concise human-readable review."""
    result = ROOT/"results/scientific-audit.json"
    write_json(result,data)
    n = data["counts"]
    lines = ["# Scientific audit of the registered comparisons", "", "Libor “Burny” Burian · 2026-11-26", "",
        audit_summary(data), "", "## What the pilot measured", "",
        f"The trace and token records reproduce {data['pilot']['correct']}/{data['pilot']['rollouts']} verifier acceptances "
        f"on {data['pilot']['distinct_tasks']} distinct training tasks, with {data['pilot']['reward_judgments']} semantic judgments. "
        "The observed reward gate prevented a test of the semantic bonus. The saved status reports the trainable-tensor comparison. "
        "A cause for the unsuccessful completions and a population performance effect remain open.",
        "", "## Qualification replay", "",
        "| Set | Valid outputs | Invalid outputs | Valid pairs | Strict wins | Simplicity ties among valid pairs | Valid omissions | Omissions rejected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["qualifications"]:
        if r["complete"]:
            lines.append(f"| {r['suite']} | {r['valid_outputs']}/{r['outputs']} | {r['invalid']} | {r['valid_pairs']}/{r['pairs']} | "
                         f"{r['wins']} | {r['ties']} | {r['valid_omissions']}/{r['omissions']} | {r['rejected']} |")
    for r in data["qualifications"]:
        if not r["complete"]:
            lines += ["",f"Interrupted `{r['log']}` recorded {r['prompts']} prompts and {r['outputs']} outputs "
                      f"({r['valid_outputs']} valid, {r['invalid']} invalid). These attempts have no qualification verdict."]
    lines += ["",f"Math delimiters appear only in the preferred response on {n['formatting']} pairs. "
              "These are deliberately constructed examples. Comparing sets confounds changed stimuli, prompts and runtime. "
              "Both semantic recognition and reliance on surface cues require counterbalanced evidence. "
              "The Wilson intervals in the JSON describe the constructed checks. The logs include interrupted attempts.",
              "", "## Public task scoring", "",
              f"Across {n['scored']} scored input occurrences, literal prompt examples match {n['base_visible']} base and "
              f"{n['plus_visible']} additional-suite occurrences. The remaining {n['additional']} occurrences differ from matched examples. "
              f"The native checker accepted {n['canonical_pass']}/{n['mbpp_tasks']} frozen canonical programs. "
              "Occurrences can repeat. The parser compares literal arguments and preserves an explicit unparsed list. "
              "The original MBPP+ training/test ID exclusion still holds. Visible worked examples are part of the public task design.",
              "", "The prospective scorer runs the pinned EvalPlus comparator on native candidate and reference values in a CPU container. "
              "This includes ordering equivalence, float tolerances and alternative pyramid-height interpretation. The project still uses its own "
              "frozen subset, prompts, resource limits and three-domain aggregation. [Upstream evaluator](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/__init__.py)",
              "", f"The physics review covers all {n['physics']} public targets; {n['missing_original_unit']} had empty unit fields. "
              "Each row links the frozen target to the original question and worked solution. Some empty units designate ratios. "
              "Others conceal physical dimensions. Source contradictions and missing context are flagged before evaluation. "
              "The stored population remains frozen. [Per-task decisions](../../configs/physics-review.psv) · [Raw audit](../../results/scientific-audit.json)",
              "", "## Hypotheses, sources and limits", "",
              "The primary quality estimand uses paired task differences within seed, equal domain weights, family resampling and two Holm-adjusted "
              "one-sided tests. Accuracy uses the registered lower confidence bound overall and in each domain. H3 has three secondary paired "
              "intervals. A concentrated cluster distribution, a degenerate bootstrap or few seeds limits population inference. "
              "A literal registered threshold alone cannot validate a flawed measurement.",
              "", "The original and revised versions have distinct scoring identities. Public source fidelity and byte equality can be checked; "
              "independent correctness of every external solution, general judge validity, model pretraining contamination, and historical push times remain unresolved. "
              "The claim register and file inventory give evidence or limitations for the repository's substantive claims. "
              "[Registered tests](../preregistration.md) · [Amendment](../amendment-004-scoring.md) · [Claim register](../claim-register.json) · [Primary-source review](../source-verification.md)."]
    lines += ["", "## Prompt-only control context", "",
              "Counts use the pinned tokenizer, chat template and complete frozen test questions. Over-limit IDs are in the raw audit. "
              "A context failure prevents the complete registered comparison.", "",
              "| Control | Test prompts | Maximum tokens | Configured limit | Over limit |","|---|---:|---:|---:|---:|"]
    for arm,b in data["prompt_budgets"].items():
        lines.append(f"| {arm} | {b['tasks']} | {b['max_tokens']} | {b['limit']} | {len(b['over_limit'])} |")
    lines += ["","## Frozen selection and generalization","",
              f"The frozen files contain {data['total_tasks']} distinct task IDs. Each selected public record was reconciled with its upstream source. "
              "The native procedural answers were independently recomputed from the questions or input lists. "
              "Normalized questions do not repeat and the held-out family split shares no frozen family labels with training. "
              "This check cannot identify benchmark examples encountered in model pretraining or every pair of semantically similar tasks.",
              "", "| Completed preparation · exclusion review | Events |", "|---|---:|"]
    for reason,count in sorted(Counter(r["review"] for r in data["exclusions"]).items()):
        lines.append(f"| {reason} | {count} |")
    out = ROOT/"reports/reviews/2026-11-26-scientific-audit.md"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(lines)+"\n",encoding="utf-8")
