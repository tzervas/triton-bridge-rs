# Precompiled payloads

| File | License | Role |
|------|---------|------|
| `identity_f32.ptx` | MIT (this repo) | Phase 1 smoke: `out[i] = in[i]` |

Flash Attention CUBIN is **not** here yet. Produce it on the 5080 with
`scripts/compile_unsloth_fa.py` and add `NOTICE` (Unsloth Apache-2.0) before
checking it in.

Do not vendor Unsloth Python source into this directory.
