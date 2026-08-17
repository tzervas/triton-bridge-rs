# Why a separate repo

Decision 2026-08-17. Revisit only if Phase 3 becomes the whole product.

## Not unsloth-rs

unsloth-rs is a **kernel library**: `RmsNorm`, RoPE, CE, attention on Candle
tensors. Its public API is tensors in / tensors out.

A Triton bridge is a **compiler + module cache + driver loader**. Different
crate type, different deps (optional CPython, file-system CUBIN assets),
different release cadence, different failure mode (`BLOCKED:env` vs
`BLOCKED:api`).

Mixing them made the 1.0.x README lie (CubeCL “2×” while `to_vec1` ran).
Keep the compiler out of the op crate.

## Not tritter-accel

tritter-accel is **Rust → Python** (PyO3) for BitNet / VSA. This crate is
**Triton/PTX → Rust launch**. Opposite arrow. Different users.

## Not rust-ai-core

rust-ai-core must not grow leaf kernel or compiler deps (DAG invert is
already a tracked bug). This crate is a leaf.

## Not CubeCL-in-process

CubeCL is a valid Rust GPU stack. It is the **wrong interop** for Candle
0.9: no zero-copy `Tensor` ↔ `Handle`. Using it as the Triton replacement
re-introduced the host copy we are deleting. CubeCL can still be a Phase 3
*backend* if they add external-buffer import.

## When to spin Phase 3 out

If a native DSL grows a parser, IR, and more than one backend, create
`rstriton` (name TBD) and leave **this** repo as the Phase 1/2 loader only.
Until then one repo keeps the story in one place.
