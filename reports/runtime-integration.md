# Local policy runtime integration

2026-11-17. These are prerequisite integration experiments, not policy-quality results. The model, task splits, reward and optimizer settings remain those in the published configuration.

## Initial GPU-profile attempt

The `profile` phase at source `fbf8604d18b180c793d3674bdbb3449eaa6d7a49` loaded the policy but failed before computing its synthetic loss: `AttributeError: 'Qwen3VLProcessor' object has no attribute 'encode'`. The complete operation record is `2026-11-17T03-33-36.516+00-00-profile.jsonl` in the published operation logs. It supplies the timestamps and traceback. There is no successful backward, GPU-fit or training result from this attempt.

Unsloth's default route loaded the multimodal wrapper and returned a processor. A [Transformers processor](https://huggingface.co/docs/transformers/main_classes/processors) groups modality-specific objects; it does not implement every tokenizer method. Inspecting the installed Unsloth loader and its [upstream implementation](https://github.com/unslothai/unsloth/blob/3384411d336a5b43cf81e8fdbf73c010eee08e6b/unsloth/models/loader.py) revealed its existing `text_only=True` route. For a supported family decoder it selects the language configuration, remaps the checkpoint's text weights and skips the vision tower. The [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) also exposes the language-only causal model.

The fix selects that upstream route rather than implementing another loader or manually stripping a processor. It applies to every policy arm before policy outputs or adapter training exist. The original failed full-profile execution is the regression evidence; the same profile is rerun after publication to test the real library integration, backward pass, role switch and generation. A passing unit mock would not establish those properties.

The runtime also reported that its optional xFormers extension could not load with the installed Torch and selected PyTorch GPU attention. That fallback is retained for measurement. Initial imports were dominated by WSL filesystem access, as observed through read-only process stack samples. These observations do not establish that the subsequent training run fits or learns; both remain to be tested.

## Corrected profile

2026-11-19 precision amendment: the profile compares only trainable tensors and the current CUDA RNG across the switch. Optimizer reload compares the number of state entries, not equality of every optimizer tensor. The repeated judge input is not a maximum-context construction. The following historical “full profile” wording refers to that implemented procedure, not a proof of complete state preservation or worst-case memory fit. See [implementation](../src/occam_rl/preflight.py) and [amendment 003](amendment-003-evidence-reporting.md).

The same full profile passed at source `944de00fb71ca322f4eca0498579b723172d22ea`. Its immutable operation log is `2026-11-17T04-02-15.178+00-00-profile.jsonl`; [the manifest](../manifests/preflight.json) supplies the values regenerated into [readiness evidence](readiness.md). The maximum configured sequence completed a differentiable forward/backward/optimizer step and maximum-length generation. Policy adapter weights and CUDA RNG survived the role switch unchanged, and optimizer state serialized and reloaded. This is state-preservation evidence, not yet a complete interrupted TRL-training resume test.

The repeated synthetic judge input produced an overlong rationale that exhausted the output budget and failed JSON validation. Its complete output is retained; it receives no valid adequacy/simplicity score. It is a resource stress input, not a judge-audit or policy-quality observation. The current frozen audit remains required before the exploratory training attempt. No semantic prompt revision follows this resource result.

## Pilot startup: JSON Lines record boundaries

The first Occam pilot attempt passed publication and identity checks and loaded the policy, but failed while reading the frozen training file, before any rollout or optimizer step. The retained [startup log](../results/runtime/occam-pilot-startup.jsonl) records the traceback. The record `gsm8k/train/2381` contains a Unicode line separator inside a valid JSON string. Python's `str.splitlines()` split that string into separate records. The [JSON Lines specification](https://jsonlines.org/) defines the record separator as LF; a Unicode separator inside a string must remain data.

The reader now splits only on LF, retains support for CRLF, and permits an incomplete final record only for live-log reads when it lacks its terminating newline. The frozen dataset bytes and hashes are unchanged. Task deserialization runs before model loading so any future data error fails promptly. Regression cases were pushed and failed before the fix.

The failed attempt has no checkpoint because optimization never began. `--resume` retains its identity and logs and restarts from the original seed only when there is no recorded model prompt, rollout, reward group or optimizer step. Once any such work exists, a full checkpoint remains mandatory. This is a startup retry, not evidence of a successful interrupted-training resume.
