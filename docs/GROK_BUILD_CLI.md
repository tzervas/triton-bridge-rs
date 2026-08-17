# Grok Build CLI — GPU job brief

Hand this file to **Grok Build CLI** on the workstation (RTX 5080 + 3090 Ti).
CPU sandbox work is done. Do not re-do Phase 0/1 loader design.

## Repos / SHAs (update if the PR moved)

| Repo | Branch / PR | Job |
|------|-------------|-----|
| `tzervas/triton-bridge-rs` | `feat/phase1-cuda-loader` [#9](https://github.com/tzervas/triton-bridge-rs/pull/9) | **Primary.** Identity PTX launch, then FA CUBIN |
| `tzervas/unsloth-rs` | `feat/p1-rope-ids-fused-ce` + [#93](https://github.com/tzervas/unsloth-rs/pull/93) | CustomOp CUDA gates (RMS/RoPE/SwiGLU/CE/attention) |
| `tzervas/rust-ai-core` | [#10](https://github.com/tzervas/rust-ai-core/pull/10) | **No GPU.** Merge when CI green |
| peft / qlora / axolotl | consume hooks | **Do not start** until FA + CustomOp CUDA are honest |

## Environment

```bash
export CUDA_COMPUTE_CAP=90          # 5080 is CC 12.0; older nvcc cannot target 120
export RUST_LOG=triton_bridge=debug
export RUST_BACKTRACE=1
# WSL only:
# export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

## Job A — triton-bridge identity (must pass first)

```bash
cd triton-bridge-rs
git checkout feat/phase1-cuda-loader
cargo test --features cuda --test identity_smoke -- --nocapture
```

**Pass:** `identity_f32` launch, bit-exact copy, `bridge_ready() == true`.
**FAIL_ENV:** no `/dev/nvidia0`, `CUDA_ERROR_NO_DEVICE`, wrong `libcuda` — report, do not fake green.
**Do not** flip any other crate’s dispatch after this job alone.

## Job B — CustomOp CUDA gates (unsloth-rs)

```bash
cd unsloth-rs
git checkout feat/p1-rope-ids-fused-ce
CUDA_COMPUTE_CAP=90 cargo test --features cuda -- --nocapture
```

Expect MAE gates on RMS / RoPE / SwiGLU / CE / `attention_device`.
CubeCL FA is **not** the default. Do not claim 2× / 70% VRAM.

## Job C — Unsloth FA CUBIN (only after A)

```bash
# workstation venv with torch + triton + unsloth (Apache-2.0 kernels only)
CUDA_COMPUTE_CAP=90 python scripts/compile_unsloth_fa.py \
  --out precompiled/sm90_flash_fwd.ptx --sm 90
```

Then:

1. Fill in the compile call in that script (do not vendor Unsloth `.py`).
2. Add `precompiled/NOTICE` (Apache-2.0 attribution).
3. Add a Rust test: launch vs Candle softmax, MAE < 1e-5 (f32), seq 128/512.
4. **Do not** add a hard `unsloth-rs` → `triton-bridge` dep until this test is green on device pointers (no `to_vec1` of Q/K/V).

## Honesty rules

- Never mark GPU suite green without `/dev/nvidia0` evidence.
- Never publish Unsloth speed numbers from this job.
- Sanitize logs (no hostnames, home paths, `nvidia-smi -q`, tokens).
- Attach `artifacts/gpu-summary.md` + redacted log to the PR.
- Stay on the listed branches; one PR per repo.

## Return

A short comment on triton-bridge #9:

- Job A/B/C: PASS / FAIL (accuracy) / FAIL_ENV
- SM, toolkit, first error
- Whether `bridge_ready()` was true
- Whether unsloth may take an optional dep yet (**no** unless Job C launched on device pointers)
