# Roadmap

## Phase 0 / 0.1.0 — contract (now)

- [x] Repo, README, architecture, kernel map
- [x] `bridge_ready() == false`
- [x] Stable load/launch types (`LoadedModule`, `LaunchSpec`, `BridgeError`)
- [x] No Python / CUDA implementation on the default feature set
- [x] Commitizen 0.x, CHANGELOG, CI, GPU handoff
- [x] Issues #1–#7

## Phase 1 — precompiled launch (next real code, **needs GPU**)

- [ ] `ModuleCache` over cudarc (or Candle `CudaDevice::get_or_load_custom_func`)
- [ ] Load PTX string / CUBIN bytes
- [ ] Launch with explicit grid/block/smem; device pointers only
- [ ] One Flash Attention payload + numerical test vs Candle softmax
- [ ] NOTICE for any Apache-2.0 Unsloth-derived PTX
- [ ] Feature `cuda` actually links a loader; still does not embed Python
- [ ] Flip `bridge_ready()` **only** when load+launch on a device pointer works

Hand off to the 5080 via [docs/GPU_HANDOFF.md](docs/GPU_HANDOFF.md).

## Phase 2 — optional compile FFI

- [ ] Prefer `scripts/compile_kernels.py` writing Phase 1 assets
- [ ] Feature `python` only if compile-at-runtime is proven necessary
- [ ] Cache key: source hash + SM + dtype + constexprs

## Phase 3 — native DSL

- [ ] Decide backend after Phase 1 has numbers
- [ ] Success = delete Phase 2
- [ ] Spin out if it grows a compiler ([WHY_SEPARATE_REPO.md](docs/WHY_SEPARATE_REPO.md))

## Never here

- LoRA trainer, QLoRA, axolotl CLI
- VSA / ternary GPU (tritter-accel)
- Claiming Unsloth product parity
- Depending on peft / qlora / axolotl / rust-ai-core / unsloth-rs
