# Initial implementation review

2026-11-17, before the first project test. This is static review evidence, not a passing test claim.

The orchestration uses TRL's GRPO loss rather than an independently reimplemented optimizer. The custom rollout hook only serializes GPU sampling and retains raw token IDs and full text. Reward code keeps verified correctness separate from the semantic judge and applies the same adequacy gate to both length arms. Ordinary, concise, and full-rubric prompt-only controls use original model weights and differ only in instructions.

Review addressed several potential validity errors: function-name salts are removed from question deduplication; code hidden cases include thresholds and empty inputs; public input types that would change under JSON transport are excluded before policy results; all paired statistical comparisons require complete task pairing; and primary quality success also requires a positive confidence interval. Public subset selection and small template-family counts remain limitations, not full-benchmark coverage.

Runtime concerns requiring measured checks: Blackwell kernels, Unsloth/TRL version compatibility, Qwen3.5 text-only loading, full parameter/RNG preservation during GPU role switching, Gemma tensor placement, maximum-context memory, structured judge output, and complete checkpoint resume behavior. These are not asserted to work before execution. Dependency versions are the newest set permitted by the selected Unsloth release's declared bounds, resolved in uv.lock.

Security and reliability review: generated code runs in a nonroot, read-only, networkless, resource-limited container without host mounts. This is not a formal container escape guarantee. Numeric parsing accepts only finite numeric literals/ratios and explicitly handles units; it never evaluates candidate Python as a numeric answer. Candidate strings in the browser use textContent. Git and Hugging Face publication select tracked source and known artifacts; .env is ignored. All models run locally; the only inference HTTP endpoint is loopback llama.cpp.

Next evidence: run the authored tests only after the complete implementation is pushed. Record failures and publish every fix before rerunning. No experimental claims are supported by this review alone.
