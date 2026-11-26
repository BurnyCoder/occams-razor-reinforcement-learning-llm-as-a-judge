"""Replay selected public reference programs in the pinned CPU scorer and measure prompt lengths."""

import json
import subprocess

from .io import ROOT, digest, read_json
from .prompts import policy_messages
from .schema import PROMPT_ARMS
from .scoring import mbpp_spec, verified_eval_image


def code_replay(tasks):
    """Amortize startup for trusted source programs only. EvalPlus isolates each candidate check."""
    payload = [{"id": t.id, "code": t.reference, "inputs": t.inputs, "mbpp": mbpp_spec(t),
                "entrypoint": t.entrypoint} for t in tasks]
    script = """import json, runpy, sys
from evalplus.eval._special_oracle import MBPP_OUTPUT_SET_EQ_TASKS, MBPP_OUTPUT_NOT_NONE_TASKS
evaluate = runpy.run_path('/opt/evalplus_runner.py')['evaluate']
rows = {}
for p in json.load(sys.stdin):
    entry = p['entrypoint']
    if entry in MBPP_OUTPUT_SET_EQ_TASKS:
        rule = 'set equality'
    elif entry in MBPP_OUTPUT_NOT_NONE_TASKS:
        rule = 'non-None result'
    elif entry in ('are_equivalent','sum_div','surface_Area','digit_distance_nums'):
        rule = 'special: ' + entry
    else:
        rule = 'generic exact or native float comparison'
    rows[p['id']] = evaluate(p,diagnostics=True) | {'comparison': rule}
print(json.dumps(rows))"""
    image = verified_eval_image()
    cmd = ["docker","run","--rm","--network","none","--read-only","--cap-drop","ALL","--security-opt",
           "no-new-privileges","--pids-limit","32","--memory","256m","--memory-swap","256m","--cpus","1",
           "--user","65534:65534","--tmpfs","/tmp:rw,noexec,nosuid,size=8m","-i",image,"python","-I","-c",script]
    out = subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=1200)
    if out.returncode:
        raise RuntimeError("Reference replay infrastructure failed: " + out.stderr[-5000:])
    rows = json.loads(out.stdout)
    if set(rows) != {t.id for t in tasks}:
        raise ValueError("Incomplete MBPP+ reference replay")
    return rows, image


def prompt_budgets(tasks):
    """Apply the pinned tokenizer and real chat template on CPU. No model weights are loaded."""
    from transformers import AutoTokenizer
    cfg = read_json(ROOT/"configs/experiment.json")
    revision = read_json(ROOT/"manifests/revisions.json")["revisions"][cfg["policy"]]
    tok = AutoTokenizer.from_pretrained(cfg["policy"],revision=revision,local_files_only=True)
    out = {}
    for arm in PROMPT_ARMS:
        messages = [policy_messages(t,arm) for t in tasks]
        tokens = tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                         enable_thinking=True,return_dict=False)
        sizes = {t.id: len(ids) for t,ids in zip(tasks,tokens,strict=True)}
        out[arm] = {"tasks":len(sizes),"max_tokens":max(sizes.values()),"limit":cfg["max_prompt_tokens"],
                    "over_limit":{k:v for k,v in sizes.items() if v>cfg["max_prompt_tokens"]},
                    "instruction_hash":digest(messages[0][0]),"tokenizer_revision":revision}
    return out
