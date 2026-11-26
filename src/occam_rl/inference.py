"""Registered comparisons. Point estimates condition on eligible tasks and observed training seeds."""

from .evidence import evaluation_issues
from .io import digest, file_hash, read_json, read_jsonl
from .protocol import evaluation_protocol
from .schema import DOMAINS, PROMPT_ARMS
from .scoring import analysis_family
from .statistics import holm


def interval_decision(interval, threshold):
    """A confidence bound equal to the margin does not establish a strict inequality."""
    lower, upper = interval
    if lower > threshold:
        return "supported"
    return "adverse" if upper < threshold else "inconclusive"


def joint(decisions):
    """All registered contrasts are required to support a conjunctive claim."""
    if "adverse" in decisions:
        return "adverse"
    if "untested" in decisions:
        return "untested"
    return "supported" if all(d == "supported" for d in decisions) else "inconclusive"


def analyze(cfg, root, runs, estimate):
    """Analyze each available contrast independently; retain the joint primary success criterion."""
    slots = ["token", "formal", "accuracy", *PROMPT_ARMS]
    h1 = {"status": "untested", "comparisons": {arm: {"status": "untested"} for arm in slots[:2]}}
    h2 = {"status": "untested", "domains": {d: {"status": "untested"} for d in ("all", *DOMAINS)}}
    h3 = {"status": "untested", "comparisons": {arm: {"status": "untested"} for arm in PROMPT_ARMS}}
    result = {"status": "pending", "comparisons": {}, "hypotheses": {"H1": h1, "H2": h2, "H3": h3}}
    paths = sorted(runs.glob("*-seed*/eval-test/results.jsonl"))
    if not paths:
        return result
    frozen = read_json(root / "manifests/data.json")
    arms, invalid, evidence = {}, [], {}
    for path in paths:
        manifest, rows = read_json(path.parent / "manifest.json"), read_jsonl(path)
        arm, seed = manifest.get("arm"), manifest.get("seed")
        if arm not in ("occam", *slots):
            continue
        issues = evaluation_issues(manifest, rows, frozen)
        if path.parent.parent.name != f"{arm}-seed{seed}" or manifest.get("split") != "test":
            issues.append("Directory disagrees with retained test identity")
        if manifest.get("limit_per_domain") is not None:
            issues.append("Limited population in retained test directory")
        if not issues:
            try:
                actual = evaluation_protocol(cfg, arm, seed, root, runs)
                prov = manifest["provenance"]
                if (manifest.get("protocol") != actual or manifest.get("config_hash") != digest(cfg)
                        or prov["judge_hash"] != actual["evaluator"] or prov["policy_revision"] != actual["base"]
                        or prov.get("adapter_sha256") != actual["adapter"]
                        or manifest.get("exploratory_judge_override") != actual["waiver"]):
                    issues.append("Evaluation disagrees with source protocol or training history")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                issues.append(f"Unverified protocol: {exc}")
        if issues:
            invalid.append({"run": path.parent.parent.name, "issues": issues})
            continue
        arms.setdefault(arm, {})[seed] = (actual, rows)
        name = path.parent.parent.name
        evidence[name] = {"protocol": actual, "frozen_split": manifest["data"],
                          "evaluation_commit": manifest["git_commit"], "result_sha256": file_hash(path),
                          "manifest_sha256": file_hash(path.parent / "manifest.json")}
    result["evidence"] = evidence
    if invalid:
        return result | {"status": "blocked_invalid_evidence", "invalid": invalid}
    target = set(cfg["seeds"])
    kwargs = {"replicates": cfg["bootstrap_replicates"], "seed": cfg["bootstrap_seed"]}
    def contrast(arm, metric, domain=None):
        seeds = sorted(target & arms.get(arm, {}).keys() & arms.get("occam", {}).keys())
        if not seeds:
            return {"status": "untested"}
        left, right, waived = [], [], False
        for seed in seeds:
            a, lrows = arms["occam"][seed]
            b, rrows = arms[arm][seed]
            fields = ["scorer", "evaluator", "base", "decoding"]
            if arm not in PROMPT_ARMS:
                fields += ["matched_training", "instruction"]
            if any(a[key] != b[key] for key in fields):
                raise ValueError(f"Incompatible treatment provenance: {arm}/{seed}")
            waived |= a["waiver"] or b["waiver"]
            left.extend(r for r in lrows if domain is None or r["domain"] == domain)
            right.extend(r for r in rrows if domain is None or r["domain"] == domain)
        effect = estimate(left, right, metric, **kwargs)
        effect["seed_sensitivity"] = estimate(left, right, metric, **kwargs, resample_seeds=True)
        # Preserve the frozen family labels for the registered analysis. Shared
        # public question templates define a prospective dependence sensitivity.
        if any(analysis_family(r["task_id"],r["family"]) != r["family"] for r in left):
            def regroup(rows):
                return [r | {"family": analysis_family(r["task_id"],r["family"])} for r in rows]
            effect["template_cluster_sensitivity"] = estimate(regroup(left),regroup(right),metric,**kwargs)
        effect["evidence"] = {str(s): [f"occam-seed{s}",f"{arm}-seed{s}"] for s in seeds}
        effect.update(seeds=seeds, waiver=waived, exploratory=bool(waived or set(seeds) != target))
        effect["interval_decision"] = interval_decision(effect["ci95"], -.02 if metric == "correct" else 0)
        effect["status"] = "exploratory" if effect["exploratory"] else effect["interval_decision"]
        return effect
    try:
        for arm in slots[:2]:
            h1["comparisons"][arm] = contrast(arm, "quality")
        primary = h1["comparisons"]
        if all("ci95" in e for e in primary.values()):
            if primary["token"]["seeds"] != primary["formal"]["seeds"]:
                h1["status"] = "exploratory"
            else:
                for e, p in zip(primary.values(), holm([v["p_one_sided"] for v in primary.values()]), strict=True):
                    e["p_holm"] = p
                    if e["status"] == "supported" and p > .05:
                        e["status"] = "inconclusive"
                h1["status"] = "exploratory" if any(e["exploratory"] for e in primary.values()) else joint([e["status"] for e in primary.values()])
        elif any("ci95" in e for e in primary.values()):
            for e in primary.values():
                if e["status"] == "supported":
                    e["status"] = "awaiting_holm_family"
        result["comparisons"] = {k: v for k, v in primary.items() if "ci95" in v}
        h2["domains"] = {d: contrast("accuracy", "correct", None if d == "all" else d) for d in ("all", *DOMAINS)}
        ds = list(h2["domains"].values())
        h2["status"] = "exploratory" if any(e.get("exploratory") for e in ds) else joint([e["status"] for e in ds])
        h3["comparisons"] = {arm: contrast(arm, "quality") for arm in PROMPT_ARMS}
        es = list(h3["comparisons"].values())
        h3["status"] = "exploratory" if any(e.get("exploratory") for e in es) else joint([e["status"] for e in es])
        if all("ci95" in e for e in [*primary.values(), ds[0]]):
            effects = [*primary.values(), ds[0]]
            result["seeds"] = sorted(set.intersection(*(set(e["seeds"]) for e in effects)))
            result["confirmatory_seed_target_met"] = set(result["seeds"]) == target
            result["accuracy_noninferiority"] = h2["domains"]
            result["status"] = "criteria_met" if h1["status"] == h2["status"] == "supported" else "not_established"
            if not result["confirmatory_seed_target_met"]:
                result["status"] = "exploratory_incomplete_seeds"
            if any(e["waiver"] for e in effects):
                result["status"] = "exploratory_judge_override"
    except ValueError as exc:
        return {"status": "blocked_invalid_evidence", "invalid": [str(exc)], "comparisons": {},
                "hypotheses": {h: {"status": "blocked_invalid_evidence"} for h in ("H1", "H2", "H3")}}
    return result
