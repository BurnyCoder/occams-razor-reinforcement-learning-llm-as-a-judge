"""Counterexamples and boundary decisions for the prospective scoring amendment."""

import pytest

from occam_rl.io import config, data_dir, read_jsonl
from occam_rl.schema import Task
from occam_rl.verify import verify


def frozen_task(identifier):
    """Read the recorded task without rebuilding a dataset or loading a model."""
    config()
    return next(Task.model_validate(t) for t in read_jsonl(data_dir() / "test.jsonl") if t["id"] == identifier)


def test_dimensional_work_accepts_equivalent_units():
    task = frozen_task("ugphysics/ClassicalElectromagnetism/298")
    assert task.unit == ""
    for text in ("0.09 J", "90 mJ"):
        assert verify(task, f"<answer>{text}</answer>").correct
    for text in ("0.09 A", "0.09"):
        assert not verify(task, f"<answer>{text}</answer>").correct


@pytest.mark.sandbox
def test_mbpp_order_uses_set_equivalence():
    task = frozen_task("Mbpp/111")
    program = "def common_in_nested_lists(xs): return sorted(set.intersection(*map(set, xs)))"
    assert verify(task, f"<code>{program}</code>").correct
    assert not verify(task, "<code>def common_in_nested_lists(xs): return []</code>").correct


def test_hypothesis_boundaries_and_negative_result():
    from occam_rl.inference import interval_decision
    assert interval_decision([-.02, .10], -.02) == "inconclusive"
    assert interval_decision([-.10, -.021], -.02) == "adverse"
    assert interval_decision([-.019, .01], -.02) == "supported"
    assert interval_decision([0, 0], 0) == "inconclusive"


def test_unobserved_hypotheses_reported_individually(tmp_path, monkeypatch):
    from occam_rl import reporting
    monkeypatch.setattr(reporting, "runs_dir", lambda: tmp_path)
    result = reporting.comparisons(config())
    assert {k: v["status"] for k, v in result["hypotheses"].items()} == dict.fromkeys(("H1", "H2", "H3"), "untested")
    assert set(result["hypotheses"]["H3"]["comparisons"]) == {"prompt_ordinary", "prompt_concise", "prompt_occam"}


def test_full_rubric_is_visible_without_judge_output_instructions():
    from occam_rl.prompts import JUDGE, policy_messages
    instruction = policy_messages(frozen_task("Mbpp/111"), "prompt_occam")[0]["content"]
    for line in JUDGE.splitlines():
        if " rates " in line:
            assert line in instruction
    assert "Return ONLY valid JSON" not in instruction
    assert "Set adequate=false" not in instruction


def test_unequal_family_sizes_match_enumerated_task_average():
    from occam_rl.statistics import paired_effect
    rows = [dict(task_id=str(i), domain="math", family="pair" if i<2 else "singleton",
                 quality=int(i<2), seed=17) for i in range(3)]
    effect = paired_effect(rows, [r | {"quality":0} for r in rows], "quality", replicates=1000)
    assert effect["effect"] == pytest.approx(2/3)
    assert effect["ci95"] == [0,1]
    assert paired_effect(rows, rows, "quality", replicates=30)["degenerate_bootstrap"]


def test_secondary_contrasts_and_accuracy_survive_missing_length_arms(tmp_path, monkeypatch):
    import json
    from occam_rl import inference, reporting
    from occam_rl.io import digest, write_json
    from test_evidence import evaluation_fixture
    manifest, rows = evaluation_fixture(tmp_path,monkeypatch)
    cfg = {"seeds":[17], "bootstrap_replicates":40, "bootstrap_seed":7}
    write_json(tmp_path/"configs/experiment.json",cfg)
    def protocol(cfg,arm,*args):
        return dict(scorer="v",evaluator="e"*64,base="d"*40,decoding="common",matched_training="matched",
                    instruction=arm if arm.startswith("prompt") else "ordinary", waiver=False,
                    adapter=None if arm.startswith("prompt") else "f"*64)
    monkeypatch.setattr(inference,"evaluation_protocol",protocol)
    for arm in ("occam","accuracy","prompt_ordinary","prompt_concise","prompt_occam"):
        directory=tmp_path/f"runs/{arm}-seed17/eval-test"
        pr=protocol(cfg,arm)
        write_json(directory/"manifest.json",manifest | {"arm":arm, "protocol":pr, "config_hash":digest(cfg),
                   "exploratory_judge_override":False,"provenance":manifest["provenance"] | {"adapter_sha256":pr["adapter"]}})
        adjusted=[]
        for r in rows:
            c=int(arm=="accuracy" or (arm=="occam" and r["domain"]!="code"))
            adjusted.append(r | {"arm":arm,"correct":c,"quality":c,"verifier":{"correct":bool(c),"reason":"fixture"}})
        (directory/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in adjusted))
    result=reporting.comparisons(cfg)["hypotheses"]
    assert result["H1"]["status"]=="untested"
    assert result["H2"]["status"]=="adverse" and result["H2"]["domains"]["code"]["effect"]==-1
    assert all(r["effect"]==pytest.approx(2/3) for r in result["H3"]["comparisons"].values())
    write_json(tmp_path/"configs/experiment.json",cfg | {"seeds":[17,29,43]})
    restricted = reporting.comparisons(cfg)
    assert restricted["registered_design"]["seeds"]==[17,29,43]
    assert restricted["hypotheses"]["H3"]["status"]=="exploratory"


def test_unresolved_reference_cannot_be_scored_as_incorrect():
    task=frozen_task("ugphysics/ClassicalMechanics/1815")
    with pytest.raises(RuntimeError, match="Unresolved scientific reference"):
        verify(task,"<answer>6 a</answer>")


@pytest.mark.sandbox
def test_upstream_alternate_pyramid_height():
    task=frozen_task("Mbpp/581")
    code="def surface_Area(b,h): return round(b*b + 2*b*((b/2)**2+h*h)**0.5)"
    assert verify(task,f"<code>{code}</code>").correct


@pytest.mark.sandbox
def test_native_tuple_and_floating_comparison_on_frozen_programs():
    swap = frozen_task("Mbpp/257")
    assert verify(swap,"<code>def swap_numbers(a,b): return b,a</code>").correct
    assert not verify(swap,"<code>def swap_numbers(a,b): return [b,a]</code>").correct
    sphere = frozen_task("Mbpp/82")
    within_tolerance = sphere.reference.replace("(r**3)","(r**3) + 1e-9")
    assert within_tolerance != sphere.reference
    assert verify(sphere,f"<code>{within_tolerance}</code>").correct


def test_equal_seed_weight_and_scope_of_dispersion():
    from occam_rl.statistics import paired_effect
    rows = [dict(task_id="q",domain="math",family="one",quality=c,seed=s) for s,c in ((17,0),(29,1))]
    result = paired_effect(rows,[r | {"quality":0} for r in rows],"quality",replicates=25)
    assert result["effect"] == .5
    assert result["between_seed_sample_sd"] == pytest.approx(2**-.5)
    assert result["ci95"] == [.5,.5] and result["degenerate_bootstrap"]


def test_frozen_physics_has_full_review_and_records_empty_original_units():
    from occam_rl.scoring import UG,physics_review
    tasks = [Task.model_validate(t) for t in read_jsonl(data_dir() / "test.jsonl") if t["source"]==UG]
    assert {t.id for t in tasks} == physics_review().keys()
    assert all(r["review"] and float(r["rounding_increment"])>=0 for r in physics_review().values())
    task = next(t for t in tasks if t.id.endswith("Electromagnetism/298"))
    assert not task.unit and physics_review()[task.id]["unit"]=="J"


def test_visible_tuple_discloses_list_values_but_not_its_container_type():
    from occam_rl.audit_data import example_coverage
    r = example_coverage(frozen_task("Mbpp/98"))
    assert r["base_matches"] and not r["exact_base_matches"] and not r["unparsed"]


def test_missing_retained_manifest_is_visible_and_blocks_inference(tmp_path,monkeypatch):
    from occam_rl import reporting
    from test_evidence import evaluation_fixture
    evaluation_fixture(tmp_path,monkeypatch)
    (tmp_path/"runs/occam-seed17/eval-test/manifest.json").unlink()
    assert reporting.snapshot()["invalid_evaluations"]
    assert all(v["status"]=="blocked_invalid_evidence" for v in reporting.comparisons({})["hypotheses"].values())
