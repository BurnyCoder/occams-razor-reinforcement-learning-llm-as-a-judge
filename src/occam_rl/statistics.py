"""Paired family-cluster bootstrap with explicit seed sensitivity.

The resampling unit matches the preregistration; no independent-response bootstrap.
Method context: https://neurips.cc/public/guides/PaperChecklist
"""

import numpy as np


def holm(pvalues: list[float]) -> list[float]:
    """Step-down familywise adjustment, returned in original comparison order."""
    order = np.argsort(pvalues)
    adjusted, maximum = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, maximum)
    return adjusted


def paired_effect(left: list[dict], right: list[dict], metric: str, *, replicates: int = 10000,
                  seed: int = 1997, resample_seeds: bool = False) -> dict:
    """Resample families within domain with the same draw for paired arms and seeds."""
    def index(rows):
        result = {(r["seed"], r["task_id"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate seed/task evaluation rows")
        return result
    lhs, rhs = index(left), index(right)
    if not lhs or lhs.keys() != rhs.keys():
        raise ValueError("Paired inference requires identical, nonempty seed/task sets")
    seeds = sorted({key[0] for key in lhs})
    domains = sorted({r["domain"] for r in lhs.values()})
    effects, families = {}, {}
    for domain in domains:
        domain_keys = [key for key,r in lhs.items() if r["domain"] == domain]
        ids_by_seed = [{key[1] for key in domain_keys if key[0] == s} for s in seeds]
        if any(ids != ids_by_seed[0] for ids in ids_by_seed):
            raise ValueError("All seeds must evaluate the same tasks")
        ids = sorted(ids_by_seed[0])
        families[domain] = np.array([lhs[seeds[0], i]["family"] for i in ids])
        for s in seeds:
            for i in ids:
                if (lhs[s,i]["domain"], lhs[s,i]["family"]) != (rhs[s,i]["domain"], rhs[s,i]["family"]):
                    raise ValueError("Paired task metadata disagree")
        effects[domain] = np.array([[float(lhs[s,i][metric])-float(rhs[s,i][metric]) for i in ids] for s in seeds])
    point = float(np.mean([values.mean() for values in effects.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for iteration in range(replicates):
        seed_indices = rng.integers(len(seeds), size=len(seeds)) if resample_seeds else np.arange(len(seeds))
        domain_means = []
        for domain in domains:
            labels = np.unique(families[domain])
            sampled = rng.choice(labels, size=len(labels), replace=True)
            task_indices = np.concatenate([np.flatnonzero(families[domain] == family) for family in sampled])
            domain_means.append(effects[domain][np.ix_(seed_indices, task_indices)].mean())
        draws[iteration] = np.mean(domain_means)
    # Centered bootstrap null: P(T* - T >= T) for a positive-effect alternative.
    pvalue = float((1 + np.count_nonzero(draws-point >= point)) / (replicates+1))
    return {"effect": point, "ci95": np.quantile(draws, [.025, .975]).tolist(), "p_one_sided": pvalue,
            "seeds": seeds, "per_seed": {str(s): float(np.mean([v[j].mean() for v in effects.values()])) for j,s in enumerate(seeds)},
            "clusters": {d: int(len(np.unique(families[d]))) for d in domains},
            "resample_seeds": resample_seeds, "replicates": replicates}


def wilson(successes: int, n: int) -> list[float]:
    """Finite-sample proportion interval, useful for small constructed audits."""
    if not n:
        return [0.0, 1.0]
    z, p = 1.959963984540054, successes/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [float(center-half), float(center+half)]
