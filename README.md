# triton-bridge-rs

**0.2** Triton / PTX launch bridge. Default features: contract only.
`--features cuda`: real `libcuda` load + launch. **`bridge_ready()` is runtime.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![semver](https://img.shields.io/badge/semver-0.x-orange.svg)](.cz.toml)

```toml
triton-bridge = { version = "0.2", features = ["cuda"] }  # workstation
triton-bridge = "0.2"                                     # CPU / CI
```

```rust
use triton_bridge::{bridge_ready, load_ptx};

if !bridge_ready() {
    // CPU CI, or cuda feature without /dev/nvidia0
    assert!(load_ptx("flash", ".version 8.0\n", Some(90)).unwrap_err().is_not_ready()
        || load_ptx("flash", ".version 8.0\n", Some(90)).unwrap_err().is_fail_env());
}
```

## Why this repo exists

Python Unsloth kernels are written in [Triton](https://github.com/triton-lang/triton).
The Rust fleet ([unsloth-rs](https://github.com/tzervas/unsloth-rs), axolotl-rs,
peft-rs) needs those **algorithms** without a permanent Python runtime.

| Job | Where | 0.2 status |
|-----|--------|----------------|
| Transformer **ops** (RMS, RoPE, SwiGLU, CE) | **unsloth-rs** `CustomOp*` | Device-resident on Candle storage |
| **Compile / load / launch** foreign GPU kernels | **this crate** | Driver loader behind `cuda`; first payload is FA CUBIN |

`tritter-accel` is the **opposite** direction (Rust → Python). Do not merge.

## Honest status

| Surface | State |
|---------|--------|
| Crate compiles | ✅ 0.2.0 |
| Default `load_ptx` / `launch` | Validates, then `NotReady` |
| `--features cuda` | `libloading` + `cuModuleLoadData` / `cuLaunchKernel` |
| `bridge_ready()` | **Runtime** — true only if driver + device |
| No device / no libcuda | `BridgeError::Device` with `FAIL_ENV` (not a green skip) |
| CPython / `triton.compile` | ❌ Phase 2 — issue #4 |
| Native Rust kernel DSL | ❌ Phase 3 — issue #5 |

Do not depend on this crate for throughput until `bridge_ready()` is true
on the target machine. unsloth-rs must not take a **hard** default dependency.

## Consumer contract

```text
unsloth-rs  ──CustomOp / NVRTC──►  CudaStorage   (default math)
     │
     └──optional --features triton-bridge──►  this crate (FA CUBIN)
```

## DAG (no cycles)

```text
triton-bridge-rs     (leaf — never depends on unsloth / peft / axolotl / core)
        ▲
        │ optional
   unsloth-rs
        ▲
   peft-rs / qlora-rs / axolotl-rs
        ▲
   rust-ai-core (optional re-exports only)
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PYTHON_UNSLOTH_KERNEL_MAP.md](docs/PYTHON_UNSLOTH_KERNEL_MAP.md)
- [docs/WHY_SEPARATE_REPO.md](docs/WHY_SEPARATE_REPO.md)
- [docs/GPU_HANDOFF.md](docs/GPU_HANDOFF.md) — 5080 numerical gate

## License

MIT — Tyler Zervas
