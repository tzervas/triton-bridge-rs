# GPU handoff (RTX 5080 / Grok Build CLI)

CPU-side Phase 1 is **done**. Next commits need a device.
Give the CLI [GROK_BUILD_CLI.md](GROK_BUILD_CLI.md) — that is the job list.

## Default profile

| Knob | Value |
|------|--------|
| Build | `cargo test --features cuda` **debug** |
| Arch pin | `CUDA_COMPUTE_CAP=90` (5080 is CC 12.0) |
| Logs | `RUST_LOG=triton_bridge=debug` `RUST_BACKTRACE=1` |
| First payload | `precompiled/identity_f32.ptx` (`tests/identity_smoke.rs`) |
| Second payload | Unsloth FA via `scripts/compile_unsloth_fa.py` |

## What not to claim

- 2× Unsloth numbers
- `bridge_ready() == true` unless load **and** launch succeeded on a device pointer
- GPU suite green without `/dev/nvidia0`

## When to hand off

Hand off when the next commit needs: `cuMemAlloc`, `cuModuleLoadData`, kernel
launch, or FA vs Candle on CUDA. That is **now**.
