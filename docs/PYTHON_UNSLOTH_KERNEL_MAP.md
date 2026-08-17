# Python Unsloth Triton kernels → Rust

Source of truth for “what Triton files exist in `unslothai/unsloth` and where
they go.” Update when Python Unsloth moves files. Paths are approximate
(`unsloth/kernels/…` as of 2026).

| Python / Triton kernel | Role | Rust home | Status |
|------------------------|------|-----------|--------|
| `rms_layernorm.py` | RMSNorm fwd/bwd | `unsloth-rs` `custom_op::rmsnorm` | CustomOp landed (P0d) |
| `rope_embedding.py` | RoPE apply | `unsloth-rs` `custom_op::rope` | CustomOp landed (P1b) |
| `swiglu.py` / `geglu` | `silu(gate)*up` | `unsloth-rs` `custom_op::swiglu` | CustomOp landed (P1c) |
| `cross_entropy_loss.py` | chunked CE | `unsloth-rs` `custom_op::ce` | CustomOp landed (P1a); fused linear+CE still open |
| `flash_attention_2.py` / FA3 | tiled attention | **this crate Phase 1** then unsloth-rs CustomOp | **Gap** — current Unsloth **main** has no `flash_attention_2.py` (flex_attention is torch.compile, not a Triton JIT). CubeCL path D2H; default now Candle (O(S²)) |
| `fast_lora.py` | fused LoRA add | peft-rs (consume unsloth, don’t fork) | Not started |
| `qK_dot` / `fast_linear` | fused GEMM+scale | unsloth-rs later; or CUBIN here | Not started |
| `flex_attention` / packing | ragged / packed | axolotl-rs + RoPE gather | RoPE `position_ids` still unused |
| `moe` / `gpt_oss` | routing | out of scope | — |
| RL / GRPO kernels | extras | out of scope | — |

## Rule

If we **own** the math and it is small (RMS, RoPE, SwiGLU, CE): rewrite as
Candle `CustomOp` in unsloth-rs. Do not FFI Triton for those.

If the kernel is **large and already correct in Triton** (Flash Attention
tiles, FA3): Phase 1 offline CUBIN is allowed as a stopgap so we do not
re-introduce CubeCL host copies.

## How to add a CUBIN (Phase 1 procedure — not automated yet)

1. In a throwaway Python env: `triton.compile` the kernel for the target SM
   (`CUDA_COMPUTE_CAP`).
2. Write the cubin/PTX under `precompiled/<sm>_<name>.ptx` (license: Unsloth
   is Apache-2.0 — **compatible with MIT consumers**, keep NOTICE).
3. `ModuleCache::load` + launch from a Candle `CustomOp` in unsloth-rs.
4. Numerical gate vs Candle softmax (MAE budget in unsloth-rs DEBT.md).

Do not vendor Unsloth Python source into this repo until legal/NOTICE is
filed (issue on Phase 1).
