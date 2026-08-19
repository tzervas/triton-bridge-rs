# Precompiled payloads

| File | License | Role |
|------|---------|------|
| `identity_f32.ptx` | MIT (this repo) | Phase 1 smoke: `out[i] = in[i]` |

Flash Attention CUBIN/PTX is **not** here yet. Produce it on a workstation
venv that has **torch + triton + unsloth** with:

```text
CUDA_COMPUTE_CAP=90 python scripts/compile_unsloth_fa.py \
  --out precompiled/sm90_flash_fwd.ptx --sm 90
```

Add `NOTICE` (Unsloth Apache-2.0) **only if** that command writes a payload.
Do not invent PTX. Do not vendor Unsloth Python source into this directory.

Honesty: a 5080 is SM **12.0**. sm90 PTX may JIT on a newer driver; it is
not the native binary for this GPU.
