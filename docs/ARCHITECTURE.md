# Architecture — triton-bridge-rs

## Problem

Triton is three things people mix up:

1. **A language** (`@triton.jit` Python DSL)
2. **A compiler** (Triton IR → TTIR → PTX/CUBIN)
3. **A launch runtime** (grid, shared memory, driver API)

unsloth-rs needs (1)’s *algorithms* and (3)’s *device-resident launch*.
It does not need (2) in-process for the default training path.

## Options considered

| Option | Host copy? | Python in-process? | Verdict |
|--------|------------|--------------------|---------|
| **A. CPython + `import triton`** | Launch can stay on GPU if you pass `CUdeviceptr` | Yes — GIL, env, 200MB+ | Phase 2 only. Temporary. Never default. |
| **B. libtriton C API** | Same as A if compile is offline | Maybe (API is unstable / not a product) | Research in #4. Do not block G0 on it. |
| **C. Offline CUBIN + cudarc launch** | No, if pointers stay on device | No | **Phase 1. This is the bridge.** |
| **D. Candle `CustomOp` + NVRTC CUDA C** | No | No | Already in **unsloth-rs**. Prefer this when we own the kernel. |
| **E. CubeCL from Candle Tensor** | **Yes** (`to_vec1` / `read_one`) today | No | Rejected for G0. Separate memory managers. |
| **F. Native Rust DSL** (Phase 3) | No | No | Long-term Triton replacement. May spin out. |

## Phase 1 shape (not implemented)

```text
                    ┌─────────────────────┐
   *.ptx / *.cubin  │  ModuleCache        │
   (checked in or   │  load once / device │
    build.rs)       └─────────┬───────────┘
                              │ CudaFunction
                              ▼
   Candle Tensor ──CustomOp──► as_cuda_slice ──► launch(grid, smem)
                              │
                              ▼
                         CudaStorage out
```

No `Tensor::to_vec1`. The only FFI is CUDA driver (already used by Candle).

Unsloth Python kernels we would *offline-compile* first (see
[PYTHON_UNSLOTH_KERNEL_MAP.md](PYTHON_UNSLOTH_KERNEL_MAP.md)):

- `cross_entropy_loss` / `chunked_cross_entropy` — **already CustomOp in unsloth-rs**
- `rms_layernorm` — **already CustomOp**
- RoPE / SwiGLU elementwise — **already CustomOp**
- Flash Attention Triton — **still needed** (tiled SRAM; Candle softmax is O(S²))

So Phase 1’s first real payload is **Flash Attention CUBIN**, not RMSNorm.

## Phase 2 (optional `python` feature)

```text
PyO3 / cpython  →  triton.compile(src, sig)  →  cubin bytes  →  Phase 1 loader
```

Exists so we can ingest new Unsloth Triton files without hand-translating to
CUDA C. The **training process** should still launch via Phase 1 (cached
CUBIN), not call CPython every step.

## Phase 3 (native DSL)

A Rust-side tile language (maybe CubeCL *without* Candle handle interop, or
a small NVRTC DSL). Success criterion: we can delete Phase 2.

Do **not** start Phase 3 until Phase 1 launches one real FA kernel and
unsloth-rs measures it against Candle softmax.

## DAG

```text
triton-bridge-rs     (this crate — compiler/runtime)
        ▲
        │ optional, never default
        │
   unsloth-rs        (ops; CustomOp is the default)
        ▲
        │
   axolotl-rs / peft-rs / qlora-rs
```

This crate must not depend on peft, qlora, axolotl, or rust-ai-core.
