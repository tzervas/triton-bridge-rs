# triton-bridge-rs

**0.1 contract** for a Triton / PTX launch bridge. **Not** a Triton port.
**Not** Unsloth. **`bridge_ready()` is `false`.**

[![CI](https://github.com/tzervas/triton-bridge-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tzervas/triton-bridge-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![semver](https://img.shields.io/badge/semver-0.x-orange.svg)](.cz.toml)

```toml
triton-bridge = "0.1"
```

```rust
use triton_bridge::{bridge_ready, load_ptx};

assert!(!bridge_ready());
assert!(load_ptx("flash", ".version 8.0\n", Some(90)).unwrap_err().is_not_ready());
```

## Why this repo exists

Python Unsloth kernels are written in [Triton](https://github.com/triton-lang/triton).
The Rust fleet ([unsloth-rs](https://github.com/tzervas/unsloth-rs), axolotl-rs,
peft-rs) needs those **algorithms** without a permanent Python runtime.

| Job | Where | 0.1.0 status |
|-----|--------|----------------|
| Transformer **ops** | **unsloth-rs** `CustomOp*` | Device-resident on Candle storage |
| **Compile / load / launch** foreign GPU kernels | **this crate** | API exists; implementation does not |

`tritter-accel` is the **opposite** direction (Rust → Python). Do not merge.

## Honest status

| Surface | State |
|---------|--------|
| Crate compiles / crates.io-shaped | ✅ `0.1.0` contract |
| `load_ptx` / `load_cubin` / `launch` | Validates args, then [`NotReady`](https://docs.rs/triton-bridge) |
| Features `cuda`, `python` | **Reserved no-ops.** They do not load CUDA or Python. They do not flip `bridge_ready()`. |
| Real PTX launch | ❌ [issue #3](https://github.com/tzervas/triton-bridge-rs/issues/3) — needs a GPU |
| CPython / `triton.compile` | ❌ [issue #4](https://github.com/tzervas/triton-bridge-rs/issues/4) |
| Native Rust kernel DSL | ❌ [issue #5](https://github.com/tzervas/triton-bridge-rs/issues/5) |

Do not depend on this crate for throughput. unsloth-rs must not take a
**hard** dependency until Phase 1 loads a real module.

## SemVer (0.x)

Commitizen: [`.cz.toml`](.cz.toml). `major_version_zero = true` — breaking
changes bump **minor**. 1.0.0 is not automatic.

```bash
cz bump --increment MINOR   # after a feat
cz bump --increment PATCH   # after a fix
git push origin main --follow-tags
```

## Phases

See [ROADMAP.md](ROADMAP.md). Phase 1 (precompiled CUBIN + device-pointer
launch, **no** CPython) is the useful temporary bridge.

GPU work happens on the workstation 5080, not in CI. Handoff:
[docs/GPU_HANDOFF.md](docs/GPU_HANDOFF.md).

## Consumer contract

```text
unsloth-rs  ──CustomOp / NVRTC──►  CudaStorage   (default)
     │
     └──optional, after bridge_ready()──►  this crate
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PYTHON_UNSLOTH_KERNEL_MAP.md](docs/PYTHON_UNSLOTH_KERNEL_MAP.md)
- [docs/WHY_SEPARATE_REPO.md](docs/WHY_SEPARATE_REPO.md)
- [docs/REVIEW_0_1.md](docs/REVIEW_0_1.md) — adversarial review of 0.0.1
- [CONTRIBUTING.md](CONTRIBUTING.md) · [DEBT.md](DEBT.md)

## License

MIT — Tyler Zervas
