# Roadmap

## Phase 0 — honesty stub (now)

- [x] Repo, README, architecture, kernel map
- [x] `bridge_ready() == false`
- [x] No Python dependency on the default feature set
- [x] Issues #1–#7

## Phase 1 — precompiled launch (next real code)

- [ ] `ModuleCache` over cudarc (or Candle `CudaDevice::get_or_load_custom_func`)
- [ ] Load PTX string / CUBIN bytes from disk or `&'static [u8]`
- [ ] Launch with explicit grid/block/smem; device pointers only
- [ ] One Flash Attention payload + numerical test vs Candle softmax
- [ ] NOTICE for any Apache-2.0 Unsloth-derived PTX
- [ ] Feature `cuda`; default stays CPU-stub so CI is green without a GPU

## Phase 2 — optional compile FFI

- [ ] Feature `python`: call `triton.compile` via PyO3 **or** document a
      standalone `scripts/compile_kernels.py` that writes Phase 1 assets
- [ ] Prefer the script (no in-process CPython) unless compile-at-runtime
      is proven necessary
- [ ] Cache key: source hash + SM + dtype + constexprs

## Phase 3 — native DSL

- [ ] Decide: extend unsloth-rs NVRTC CUDA C, CubeCL (only with zero-copy),
      or a new IR
- [ ] Success = delete Phase 2
- [ ] If it grows a compiler, spin out (see WHY_SEPARATE_REPO.md)

## Explicitly later / never here

- LoRA trainer, QLoRA, axolotl CLI
- VSA / ternary GPU (tritter-accel)
- Claiming Unsloth product parity
