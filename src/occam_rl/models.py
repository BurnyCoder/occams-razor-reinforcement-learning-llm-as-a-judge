"""Local GPU policy and judge runtimes; imports are lazy to keep CPU tests light.

Loader sources: Unsloth Qwen3.5 guide and Transformers Qwen3.5/BitsAndBytes docs.
No cloud inference client is used. Gemma's HTTP transport binds only loopback.
"""

import gc
import json
import os
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from pydantic import ValidationError

from .gpu import MemoryMonitor
from .io import ROOT, read_json
from .prompts import judge_messages
from .schema import Judgment, Task


def revision(model: str) -> str:
    """Reject mutable 'main' once revisions are frozen."""
    value = read_json(ROOT / "manifests/revisions.json")["revisions"][model]
    if not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError(f"Unpinned model: {model}")
    return value


def clear_gpu() -> None:
    """Release unused allocator blocks after dropping all references to a role."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def assert_gpu_model(model) -> None:
    """Every parameter must be on CUDA; auto device maps may otherwise hide CPU work."""
    devices = {str(parameter.device) for parameter in model.parameters()}
    if not devices or any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"Model is not wholly on GPU: {sorted(devices)}")


def load_policy(cfg: dict, seed: int, adapter: Path | None = None):
    """Use documented BF16 LoRA with local Transformers generation, no vLLM."""
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["policy"], revision=revision(cfg["policy"]),
        max_seq_length=cfg["max_prompt_tokens"] + cfg["max_completion_tokens"],
        dtype=torch.bfloat16, load_in_4bit=False, load_in_16bit=True,
        fast_inference=False, device_map={"": 0}, trust_remote_code=False)
    if adapter is None:
        model = FastLanguageModel.get_peft_model(model, r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=seed)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    tokenizer.padding_side = "left"
    assert_gpu_model(model)
    return model, tokenizer


def stream_logger(tokenizer, log, role: str):
    """Transformers streams complete text fragments immediately to timestamped logs."""
    from transformers import TextStreamer

    class LoggedStreamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            if text:
                log.emit("llm_stream", role=role, text=text, stream_end=stream_end)
    return LoggedStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)


def generate(model, tokenizer, messages: list[dict], cfg: dict, log, *, role="policy", thinking=True,
             max_new_tokens: int | None = None, prefix_allowed_tokens_fn=None) -> dict:
    """Serial generation preserves all emitted tokens and never truncates input."""
    import torch
    assert_gpu_model(model)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=thinking, return_tensors="pt", return_dict=False)
    limit = cfg["max_prompt_tokens"] if role == "policy" else cfg["judge_context"] - cfg["judge_max_new_tokens"]
    if ids.shape[-1] > limit:
        raise ValueError(f"Full {role} input exceeds context; refusing silent truncation: {ids.shape[-1]}>{limit}")
    ids = ids.to("cuda")
    new_tokens = max_new_tokens or (cfg["max_completion_tokens"] if role == "policy" else cfg["judge_max_new_tokens"])
    log.emit("llm_prompt", role=role, messages=messages, prompt_tokens=ids.shape[-1], thinking=thinking)
    start = time.monotonic()
    sampling = {"do_sample": True, "temperature": cfg["temperature"], "top_p": cfg["top_p"], "top_k": cfg["top_k"],
                "min_p": 0.0, "repetition_penalty": 1.0} if role == "policy" else {"do_sample": False}
    training = model.training
    model.eval()
    try:
        with torch.inference_mode(), MemoryMonitor(cfg["headroom_mib"], log, role):
            outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=new_tokens,
                                     pad_token_id=tokenizer.pad_token_id, use_cache=True,
                                     streamer=stream_logger(tokenizer, log, role),
                                     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, **sampling)
        completion = outputs[0, ids.shape[-1]:].tolist()
        trace = tokenizer.decode(completion, skip_special_tokens=False)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        result = {"prompt_ids": ids[0].tolist(), "completion_ids": completion, "trace": trace,
                  "tokens": len(completion), "truncated": not completion or completion[-1] not in eos,
                  "seconds": time.monotonic()-start}
        log.emit("llm_output", role=role, **result)
        return result
    finally:
        model.train(training)


@contextmanager
def parked_policy(model):
    """Preserve parameter objects and RNG while the frozen judge occupies the GPU."""
    import torch
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    training = model.training
    ids_before = [id(p) for p in model.parameters()]
    model.to("cpu")
    clear_gpu()
    try:
        yield
    finally:
        clear_gpu()
        model.to("cuda")
        model.train(training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        if ids_before != [id(p) for p in model.parameters()]:
            raise RuntimeError("Model switching replaced parameter objects; optimizer state invalid")
        assert_gpu_model(model)


class QwenJudge:
    """Frozen NF4 local judge; lifecycle is explicit to release all weights between roles."""

    def __init__(self, cfg: dict, log):
        self.cfg, self.log = cfg, log

    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["judge"], revision=revision(self.cfg["judge"]))
        # Reuse the library's public Transformers integration, not a custom decoder.
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
        with MemoryMonitor(self.cfg["headroom_mib"], self.log, "judge_load"):
            self.model = Qwen3_5ForCausalLM.from_pretrained(self.cfg["judge"], revision=revision(self.cfg["judge"]),
                dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=False,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        self.model.eval()
        self.model.requires_grad_(False)
        assert_gpu_model(self.model)
        return self

    def assess(self, task: Task, trace: str) -> Judgment | None:
        """Invalid JSON receives no adequacy bonus, with the complete failure retained."""
        from .structured import judgment_prefix
        raw = generate(self.model, self.tokenizer, judge_messages(task, trace), self.cfg, self.log,
                       role="judge", thinking=False,
                       prefix_allowed_tokens_fn=judgment_prefix(self.tokenizer_data))["trace"]
        return parse_judgment(raw, self.log, task.id)

    def __exit__(self, *args):
        del self.model, self.tokenizer, self.tokenizer_data
        clear_gpu()


def parse_judgment(raw: str, log, task_id: str) -> Judgment | None:
    """Strip only known transport tokens, never repair malformed judgment semantics."""
    clean = raw.strip()
    for token in ("<|im_end|>", "<|endoftext|>", "<|end|>", "<end_of_turn>", "<turn|>"):
        clean = clean.replace(token, "").strip()
    try:
        result = Judgment.model_validate_json(clean)
        log.emit("judgment", task_id=task_id, judgment=result.model_dump(), simplicity=result.simplicity)
        return result
    except ValidationError as exc:
        log.emit("judge_invalid", task_id=task_id, raw=raw, error=str(exc))
        return None


class GemmaJudge:
    """Pinned llama.cpp server with all tensors forced to CUDA and no context shifting."""

    def __init__(self, cfg: dict, log, model_id: str | None = None):
        self.cfg, self.log = cfg, log
        self.model_id = model_id or read_json(ROOT / "manifests/gemma-selection.json")["model"]

    def __enter__(self):
        from huggingface_hub import hf_hub_download
        manifest = read_json(ROOT / "manifests/revisions.json")
        model_file = hf_hub_download(self.model_id, manifest["gguf_files"][self.model_id], revision=revision(self.model_id))
        executable = Path(os.environ["OCCAM_LLAMA_SERVER"])
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.stderr_path = self.log.path.parent / ("llama-" + self.model_id.split("/")[-1] + ".log")
        self.stream = self.stderr_path.open("w", encoding="utf-8")
        command = [str(executable), "-m", model_file, "-ngl", "all", "--split-mode", "none",
                   "--override-tensor", ".*=CUDA0", "--lazy-mode", "off", "--no-mmap", "--no-context-shift",
                   "-c", str(self.cfg["judge_context"]), "-b", "128", "-ub", "128", "--parallel", "1",
                   "--host", "127.0.0.1", "--port", str(port), "--jinja", "--reasoning-format", "none",
                   "--chat-template-kwargs", '{"enable_thinking":false}']
        # Official CUDA release assets separate the runtime libraries from binaries.
        # Scope this path to llama.cpp so Torch keeps its own pinned CUDA runtime.
        server_env = dict(os.environ)
        cuda_libraries = os.environ.get("OCCAM_LLAMA_CUDA_LIB")
        if cuda_libraries:
            server_env["LD_LIBRARY_PATH"] = cuda_libraries + ":" + server_env.get("LD_LIBRARY_PATH", "")
        self.process = subprocess.Popen(command, stdout=self.stream, stderr=subprocess.STDOUT, env=server_env)
        try:
            with MemoryMonitor(self.cfg["headroom_mib"], self.log, "gemma_load"):
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError("llama-server failed: " + self.stderr_path.read_text()[-6000:])
                    try:
                        if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError("llama-server readiness timeout")
            startup = self.stderr_path.read_text()
            matches = re.findall(r"offloaded (\d+)/(\d+) layers", startup)
            if not matches or any(a != b for a,b in matches):
                raise RuntimeError("Cannot verify full GPU layer placement from llama.cpp log")
            self.log.emit("gemma_ready", model=self.model_id, server_log=startup)
            return self
        except BaseException:
            self.__exit__()
            raise

    def assess(self, task: Task, trace: str) -> Judgment | None:
        """Use local streaming JSON-schema-constrained decoding; full prompts are retained."""
        messages = judge_messages(task, trace)
        rendered = httpx.post(self.url + "/apply-template", json={"messages": messages, "add_generation_prompt": True,
                              "chat_template_kwargs": {"enable_thinking": False}}, timeout=30)
        rendered.raise_for_status()
        tokenized = httpx.post(self.url + "/tokenize", json={"content": rendered.json()["prompt"]}, timeout=30)
        tokenized.raise_for_status()
        count = len(tokenized.json()["tokens"])
        if count + self.cfg["judge_max_new_tokens"] > self.cfg["judge_context"]:
            raise ValueError(f"Full Gemma judgment exceeds context: {count}; refusing truncation")
        self.log.emit("llm_prompt", role="gemma", messages=messages, prompt_tokens=count, thinking=False)
        payload = {"messages": messages, "temperature": 0, "max_tokens": self.cfg["judge_max_new_tokens"],
                   "stream": True, "chat_template_kwargs": {"enable_thinking": False},
                   "response_format": {"type": "json_schema", "json_schema": {"name": "occam", "schema": Judgment.model_json_schema()}}}
        raw = ""
        with MemoryMonitor(self.cfg["headroom_mib"], self.log, "gemma_request"):
            with httpx.stream("POST", self.url + "/v1/chat/completions", json=payload, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("choices"):
                        chunk = event["choices"][0].get("delta", {}).get("content") or ""
                        raw += chunk
                        if chunk:
                            self.log.emit("llm_stream", role="gemma", text=chunk)
        self.log.emit("llm_output", role="gemma", trace=raw)
        return parse_judgment(raw, self.log, task.id)

    def __exit__(self, *args):
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if hasattr(self, "stream"):
            self.stream.close()
