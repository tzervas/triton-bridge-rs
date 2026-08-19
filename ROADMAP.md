# Roadmap

## Phase 0 / 0.1.0 — contract

- [x] Repo, README, architecture, kernel map
- [x] `bridge_ready() == false` on default features
- [x] Stable load/launch types (`LoadedModule`, `LaunchSpec`, `BridgeError`)
- [x] No Python / CUDA implementation on the default feature set
- [x] Commitizen 0.x, CHANGELOG, CI, GPU handoff
- [x] Issues #1–#7

## Phase 1 — precompiled launch (0.2.0 — this PR)

- [x] Driver loader via `libloading` + `libcuda` (`--features cuda`)
- [x] Load PTX string / CUBIN bytes (`cuModuleLoadData`)
- [x] Launch with explicit grid/block/smem + `KernelArg` device pointers
- [x] `FAIL_ENV` when libcuda / device missing (no silent pass)
- [x] `bridge_ready()` runtime-true only with driver + device
- [ ] One Flash Attention payload + numerical test vs Candle softmax (**5080**)
- [ ] NOTICE for any Apache-2.0 Unsloth-derived PTX
- [ ] Flip documented “ready on CI” only after that numerical gate

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
