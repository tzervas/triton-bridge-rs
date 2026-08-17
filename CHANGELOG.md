# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) **0.x** (`major_version_zero`).
Commits: Conventional Commits via Commitizen (`.cz.toml`).

## [Unreleased]

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

[Unreleased]: https://github.com/tzervas/triton-bridge-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tzervas/triton-bridge-rs/releases/tag/v0.1.0
