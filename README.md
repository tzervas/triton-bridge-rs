# triton-bridge-rs

**Temporary** bridge between OpenAI Triton (and precompiled PTX/CUBIN) and the
tzervas Rust LLM stack. **Not** a product port of Triton. **Not** Unsloth.
**Not** ready to compile Python `@triton.jit` kernels.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this repo exists

Python Unsloth (and a lot of other training kernels) are written in
[Triton](https://github.com/triton-lang/triton). The Rust fleet
([unsloth-rs](https://github.com/tzervas/unsloth-rs), axolotl-rs, peft-rs)
needs those **algorithms** without a permanent Python runtime.

Two different jobs got conflated in planning:

| Job | Where it lives | Status |
|-----|----------------|--------|
| Transformer **ops** (RMSNorm, RoPE, CE, attention) | **unsloth-rs** Candle `CustomOp*` | Shipping on `CudaStorage` (no CubeCL `to_vec1`) |
| **Compile / load / launch** foreign GPU kernels (what Triton *the compiler* does) | **this crate** | Scaffold + contract only |

Putting a Python/Triton FFI inside `unsloth-rs` would:

- pull CPython / libtriton into a kernel library
- invert the DAG (kernels depending on a compiler runtime)
- block consumers who only want `CustomOp` RMSNorm

`tritter-accel` is the **opposite** direction (Rust → Python via PyO3 for
BitNet/VSA). Do not merge this into that crate.

## Honest status (2026-08-17)

| Surface | State |
|---------|--------|
| Crate compiles | ✅ stub (`triton_bridge`) |
| Load PTX/CUBIN + launch | ❌ Phase 1 — [issue #3](https://github.com/tzervas/triton-bridge-rs/issues/3) |
| CPython / libtriton FFI | ❌ Phase 2 — [issue #4](https://github.com/tzervas/triton-bridge-rs/issues/4) |
| Native Rust kernel DSL (“Rust Triton”) | ❌ Phase 3 — [issue #5](https://github.com/tzervas/triton-bridge-rs/issues/5) |
| Host-roundtrip Candle↔CubeCL | **Out of scope** — that is unsloth-rs G0 (`CustomOp`) |

`bridge_ready()` is **`false`**. Do not depend on this crate for throughput.

## Phases

See [ROADMAP.md](ROADMAP.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```text
Phase 0  honesty stub (this commit)
   │
Phase 1  load precompiled PTX/CUBIN via cudarc — no Python
   │     (offline: compile Unsloth Triton kernels once, check in CUBIN)
   │
Phase 2  optional `python` feature: FFI to Triton's compiler
   │     (temporary; requires a Python env; not the deploy target)
   │
Phase 3  native Rust kernel DSL + NVRTC/PTX
         spin out to its own crate if it grows a compiler
```

Phase 1 is the useful temporary bridge: **algorithms from Triton, launch from
Rust, no CPython in the training process.**

## Consumer contract (unsloth-rs)

```text
unsloth-rs  ──CustomOp / NVRTC──►  CudaStorage   (default, device-resident)
     │
     └──optional──►  triton-bridge-rs  (precompiled PTX only, when a
                    Triton kernel is not yet rewritten as CustomOp)
```

unsloth-rs **must not** take a hard dependency on this crate until Phase 1
loads a real module. Tracked in unsloth-rs #87.

## Non-goals

- Replacing Candle or CubeCL
- Shipping a Python interpreter inside axolotl-rs
- Claiming Unsloth 2× / 70% VRAM numbers
- Ternary / VSA / BitNet (tritter-accel, trit-vsa, bitnet-quantize)

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — FFI options and why
- [docs/PYTHON_UNSLOTH_KERNEL_MAP.md](docs/PYTHON_UNSLOTH_KERNEL_MAP.md) — which Python kernels map where
- [docs/WHY_SEPARATE_REPO.md](docs/WHY_SEPARATE_REPO.md)
- [ROADMAP.md](ROADMAP.md) · [DEBT.md](DEBT.md)

## License

MIT — Tyler Zervas
