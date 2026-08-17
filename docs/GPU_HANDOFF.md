# GPU handoff (RTX 5080 / Claude)

This crate’s Phase 1 **cannot** be finished in a CPU-only sandbox.
When a change needs a device, hand it to Claude on the workstation.

## Default early-phase profile

| Knob | Value |
|------|--------|
| Build | `cargo test --features cuda` **debug** (not `--release`) |
| Logs | `RUST_LOG=triton_bridge=debug,info` + `RUST_BACKTRACE=1` |
| Artifacts | upload to the PR, **sanitized** |
| Secrets | run Tyler’s secret toolchain **first** (Claude already knows it), including the secret-redaction step |
| PII | strip home paths, hostnames, usernames, tokens, `~/.cache` dumps |

Do **not** attach raw `nvidia-smi -q`, env dumps, or `/proc` that leak
serial numbers / driver license keys without redaction.

## What to run (Phase 1, when code exists)

```bash
export CUDA_COMPUTE_CAP=90   # pin; 5080 is CC 12.0, older nvcc cannot target 120
export RUST_LOG=triton_bridge=debug
export RUST_BACKTRACE=1
cargo test --features cuda -- --nocapture 2>&1 | tee /tmp/triton-bridge-gpu-debug.log
```

Then redact `/tmp/triton-bridge-gpu-debug.log` through the secret toolchain
and attach `artifacts/gpu-debug-redacted.log` + a short `artifacts/gpu-summary.md`
(pass/fail, SM, MAE vs Candle if any, first error).

## What *not* to claim from a GPU run

- 2× Unsloth numbers
- `bridge_ready() == true` unless load **and** launch succeeded on a
  device pointer (no host `to_vec` of the working set)

## Prompt stub for Claude

```text
Repo: tzervas/triton-bridge-rs @ <sha or PR>
Task: run Phase 1 GPU tests in debug on the 5080.
Constraints: CUDA_COMPUTE_CAP=90; do not flip bridge_ready unless launch
is device-resident; sanitize logs with the secret toolchain + redaction
before upload; attach artifacts to the PR; do not commit secrets.
Return: redacted log, summary, and any patch as a PR on this repo only.
```

## When to hand off

Hand off when the next commit needs any of: NVRTC, cubin load, `cuModule`,
kernel launch, numerical FA vs Candle on CUDA. CPU argument-validation
and docs stay in the sandbox.
