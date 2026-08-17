# GPU summary — Job C (Unsloth FA CUBIN)

Draft. Logs redacted (no home paths, no hostname, no `nvidia-smi -q`).

## Host

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce RTX 5080 |
| Compute capability | **12.0** (not 9.0) |
| `/dev/nvidia0` | present |
| Toolkit | CUDA 13.1 (`nvcc` / `ptxas` 13.1.115) |
| Rust | 1.96.0 |
| Python | 3.13.5 (system) |
| torch | 2.10.0+cu128, `cuda=True`, capability `(12, 0)` |
| triton | 3.6.0 (site-packages) |
| unsloth | **missing** (`ModuleNotFoundError: No module named 'unsloth'`) |

Probed system Python plus other local venvs (hypha, hypha-forge, hf-cli, uv cpython 3.12/3.14). None had `unsloth`. No conda/mamba. Did **not** invent PTX. Did **not** `pip install unsloth`.

## Job A (identity) — still PASS

`cargo test --features cuda --test identity_smoke` launched `identity_f32` and bit-exact copied. `bridge_ready()` is runtime-true on this box (driver + device).

## Job C — FAIL_ENV

Command:

```text
CUDA_COMPUTE_CAP=90 python scripts/compile_unsloth_fa.py \
  --out precompiled/sm90_flash_fwd.ptx --sm 90
```

Exit 2. Stderr (sanitized):

```text
HONEST: host SM is 12.0; compiling for sm90. sm90 PTX may not be the
right binary to launch on SM 12.0 (5080).
FAIL_ENV: need unsloth in the workstation venv
  (torch+triton present; unsloth missing: No module named 'unsloth')
```

Stdout (sanitized):

```text
triton 3.6.0 torch 2.10.0+cu128 sm=90
host gpu='NVIDIA GeForce RTX 5080' compute_capability=12.0
```

- `precompiled/sm90_flash_fwd.ptx` was **not** written.
- `precompiled/NOTICE` was **not** added (compile produced no output).
- Current Unsloth **main** has no `flash_attention_2.py`; `flex_attention` is torch.compile, not a Triton JIT. Script looks those paths up and refuses to invent PTX.

## Rust MAE gate

`tests/flash_fwd_mae.rs`: seq 128/512, f32, Candle-compatible softmax, MAE budget `1e-5`. Launch path uses `KernelArg::DevicePtr` only (no `to_vec1` of Q/K/V).

`--features cuda` result:

```text
FAIL_ENV: precompiled/sm90_flash_fwd.ptx missing — Unsloth FA was not
compiled (no invented PTX). device_pointers=false
test flash_fwd_vs_candle_softmax_or_fail_env ... ok
```

Cargo test is green (honest skip). **MAE was not measured.** Device-pointer FA launch did **not** happen.

## Honesty

- Host is SM **12.0**. Even if sm90 PTX existed, it may not be the right binary to launch on this 5080.
- `device_pointers=false`
- Do **not** add `unsloth-rs` → `triton-bridge` (hard or default).
- No Unsloth speed numbers from this job.
