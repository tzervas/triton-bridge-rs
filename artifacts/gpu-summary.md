# GPU summary — triton-bridge-rs PR #9

Sanitized (no hostname, no home paths, no tokens, no `nvidia-smi -q`).

## Verdicts

| Job | Result | Notes |
|-----|--------|-------|
| A identity PTX launch | **PASS** | `identity_f32` bit-exact copy |
| B unsloth-rs CustomOp CUDA gates | **PASS** | RMS / RoPE / SwiGLU / CE / `attention_device` |
| C Unsloth FA CUBIN + MAE | compile **FAIL**, launch **SKIP** | unsloth missing; no PTX written; `device_pointers=false` |

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
| unsloth (Python) | **missing** (`ModuleNotFoundError: No module named 'unsloth'`) |
| `bridge_ready()` | **true** (Job A load + launch on device pointers) |
| Arch pin | `CUDA_COMPUTE_CAP=90` (host is SM 12.0; sm90 is a compile target, not a launch guarantee) |

## First error

Job C, exit 2:

```text
FAIL_ENV: need unsloth in the workstation venv
  (torch+triton present; unsloth missing: No module named 'unsloth')
```

Preceded by the honesty line:

```text
HONEST: host SM is 12.0; compiling for sm90. sm90 PTX may not be the
right binary to launch on SM 12.0 (5080).
```

## Job A — PASS

`cargo test --features cuda --test identity_smoke` launched `precompiled/identity_f32.ptx` (`identity_f32`) and copied bit-exact. `bridge_ready()` is runtime-true (driver + device).

## Job B — PASS

`CUDA_COMPUTE_CAP=90 cargo test --features cuda` on unsloth-rs `feat/p1-rope-ids-fused-ce`. CustomOp CUDA gates passed. CubeCL FA is not the default. No 2× / 70% VRAM claims from this job.

## Job C — compile FAIL, launch SKIP

Command:

```text
CUDA_COMPUTE_CAP=90 python scripts/compile_unsloth_fa.py \
  --out precompiled/sm90_flash_fwd.ptx --sm 90
```

- Compile: **FAIL** (exit 2). `triton.compile` never ran; Python `unsloth` is missing.
- Launch: **SKIP** (no `precompiled/sm90_flash_fwd.ptx`).
- `precompiled/NOTICE` was **not** added (compile produced no output).
- Did **not** invent PTX. Did **not** `pip install unsloth`.
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
