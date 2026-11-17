# Local policy runtime integration

2026-11-17. These are prerequisite integration experiments, not policy-quality results. The model, task splits, reward and optimizer settings remain those in the published configuration.

## Initial GPU-profile attempt

The `profile` phase at source `fbf8604d18b180c793d3674bdbb3449eaa6d7a49` loaded the policy but failed before computing its synthetic loss: `AttributeError: 'Qwen3VLProcessor' object has no attribute 'encode'`. The complete operation record is `2026-11-17T03-33-36.520+00-00-profile.jsonl` in the published operation logs. It supplies the timestamps and traceback. There is no successful backward, GPU-fit or training result from this attempt.

Unsloth's default route loaded the multimodal wrapper and returned a processor. A [Transformers processor](https://huggingface.co/docs/transformers/main_classes/processors) groups modality-specific objects; it does not implement every tokenizer method. Inspecting the installed Unsloth loader and its [upstream implementation](https://github.com/unslothai/unsloth/blob/3384411d336a5b43cf81e8fdbf73c010eee08e6b/unsloth/models/loader.py) revealed its existing `text_only=True` route. For a supported family decoder it selects the language configuration, remaps the checkpoint's text weights and skips the vision tower. The [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) also exposes the language-only causal model.

The fix selects that upstream route rather than implementing another loader or manually stripping a processor. It applies to every policy arm before policy outputs or adapter training exist. The original failed full-profile execution is the regression evidence; the same profile is rerun after publication to test the real library integration, backward pass, role switch and generation. A passing unit mock would not establish those properties.

The runtime also reported that its optional xFormers extension could not load with the installed Torch and selected PyTorch GPU attention. That fallback is retained for measurement. Initial imports were dominated by WSL filesystem access, as observed through read-only process stack samples. These observations do not establish that the subsequent training run fits or learns; both remain to be tested.
