# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) **0.x** (`major_version_zero`).
Commits: Conventional Commits via Commitizen (`.cz.toml`).

## [Unreleased]

## [0.2.0] - 2026-08-17

### Added
- Phase 1 CUDA driver loader behind `--features cuda` (`libloading` + `libcuda`).
- `KernelArg` (device pointers as `u64`, no host `*const T`).
- PTX / CUBIN shape checks (`validate_ptx` requires `.version`).
- `device_alloc` / `device_free` / `memcpy_htod` / `memcpy_dtoh` (cuda feature).
- `precompiled/identity_f32.ptx` + `tests/identity_smoke.rs`.
- `scripts/compile_unsloth_fa.py` + `docs/GROK_BUILD_CLI.md` for the 5080 job.
- `BridgeError::is_fail_env()` — no libcuda / no device is **FAIL_ENV**, not green.

### Changed
- `bridge_ready()` is a **runtime** function. True only with `cuda` + driver + device.
- `LaunchSpec` gained `args`. Breaking for struct literals (0.x minor bump).

### Honesty
- Default features still do not link CUDA (`NotReady`).
- `cuda` feature on a CPU box returns `Device { FAIL_ENV }` — never a silent pass.
- First real payload remains Flash Attention CUBIN on the 5080 (see GPU_HANDOFF).
- unsloth-rs still must not hard-depend on this crate for default math.

## [0.1.0] - 2026-08-17

### Added
- Public contract: `bridge_ready`, `load_ptx`, `load_cubin`, `launch`,
  `LaunchSpec`, `LoadedModule`, `BridgeError`.
- Argument validation (`InvalidModule`) vs `NotReady` for well-formed input.
- Leaf-rule test: Cargo.toml must not depend on peft/qlora/axolotl/core/unsloth.
- Docs: architecture, kernel map, GPU handoff (5080 / Claude), review notes.
- Commitizen 0.x config (`.cz.toml`).
- CI: fmt, clippy `-D warnings`, tests default + `--all-features`,
  `cargo package --allow-dirty` dry-run.

### Honesty
- `bridge_ready()` is **false**. Features `cuda` / `python` are reserved no-ops.
- This is not a Triton compiler and not an Unsloth product.

[Unreleased]: https://github.com/tzervas/triton-bridge-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tzervas/triton-bridge-rs/releases/tag/v0.2.0
[0.1.0]: https://github.com/tzervas/triton-bridge-rs/releases/tag/v0.1.0
