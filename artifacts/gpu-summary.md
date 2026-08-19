# GPU summary — triton-bridge-rs PR #9

Sanitized (no hostname, no home paths, no tokens, no `nvidia-smi -q`).

## Verdicts

| Job | Result | Notes |
|-----|--------|-------|
| A identity PTX launch | **PASS** | `identity_f32` bit-exact copy |
| B unsloth-rs CustomOp CUDA gates | **PASS** | RMS / RoPE / SwiGLU / CE / `attention_device` |
| C Unsloth FA CUBIN + MAE | compile **FAIL_ENV**, launch **SKIP** | persist-volume Unsloth 2026.8.18 imported; no FA Triton JIT; no PTX written; `device_pointers=false` |

## Environment

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce RTX 5080 |
| Compute capability (SM) | **12.0** |
| Toolkit | CUDA **13.1** (`nvcc` / `ptxas` 13.1.115) |
| `/dev/nvidia0` | present |
| Rust | 1.96.0 |
| Python | 3.13.5 (system) |
| torch | 2.10.0+cu128, `cuda=True`, capability `(12, 0)` |
| triton | 3.6.0 |
| unsloth (Python) | **2026.8.18** in compare persist volume (`unsloth-rs-compare-site`). Host workstation venv still has no unsloth. |
| `bridge_ready()` | **true** (Job A load + launch on device pointers) |
| Arch pin | `CUDA_COMPUTE_CAP=90` (host is SM 12.0; sm90 is a compile target, not a launch guarantee) |

## First error

Job C container retry, exit 2 (2026-08-17):

```text
./scripts/compile_unsloth_fa_container.sh
HONEST: host GPU is SM 12.0; --sm 90 is a compile target, not a launch guarantee.
triton 3.6.0 torch 2.11.0+cu130 sm=90
host gpu='NVIDIA GeForce RTX 5080' compute_capability=12.0
lookup tried:
  unsloth.kernels.flash_attention_2 (No module named ...)
  unsloth.kernels.flex_attention (imported; jit=none)
  unsloth_zoo.flex_attention (imported; jit=none)
FAIL_ENV: no Apache-2.0 Unsloth FA fwd JIT kernel found.
Current unsloth.kernels has no flash_attention_2.py
(flex_attention is torch.compile, not a Triton JIT).
Refusing to invent PTX.
```

Earlier host-venv attempt was also FAIL_ENV (`unsloth` missing). Persist-volume import is no longer the blocker.

## Job A — PASS

`cargo test --features cuda --test identity_smoke` launched `precompiled/identity_f32.ptx` (`identity_f32`) and copied bit-exact. `bridge_ready()` is runtime-true (driver + device).

## Job B — PASS

`CUDA_COMPUTE_CAP=90 cargo test --features cuda` on unsloth-rs `feat/p1-rope-ids-fused-ce`. CustomOp CUDA gates passed. CubeCL FA is not the default. No 2× / 70% VRAM claims from this job.

## Job C — compile FAIL_ENV, launch SKIP (container + persist volume)

Command:

```text
CUDA_COMPUTE_CAP=90 ./scripts/compile_unsloth_fa_container.sh \
  precompiled/sm90_flash_fwd.ptx
```

- Unsloth **imported** (`2026.8.18` from named volume `unsloth-rs-compare-site`).
- Compile: **FAIL_ENV** (exit 2). No Apache-2.0 FA fwd Triton JIT. `flex_attention` imported with `jit=none`. `triton.compile` never ran.
- Launch: **SKIP** (no `precompiled/sm90_flash_fwd.ptx`).
- `precompiled/NOTICE` was **not** added (compile produced no output).
- Did **not** invent PTX.
- Current Unsloth **main** has no `flash_attention_2.py`; `flex_attention` is torch.compile, not a Triton JIT.

Rust MAE gate (`tests/flash_fwd_mae.rs`, seq 128/512, f32, Candle-compatible softmax, MAE budget `1e-5`, `KernelArg::DevicePtr` only):

```text
FAIL_ENV: precompiled/sm90_flash_fwd.ptx missing — Unsloth FA was not
compiled (no invented PTX). device_pointers=false
test flash_fwd_vs_candle_softmax_or_fail_env ... ok
```

Cargo test is green (honest skip). **MAE was not measured.** Device-pointer FA launch did **not** happen (`device_pointers=false`).

## Dependency decision

**unsloth-rs may not take an optional / hard dep on triton-bridge.** Job C did not launch FA on device pointers. Do not start peft / qlora / axolotl.
