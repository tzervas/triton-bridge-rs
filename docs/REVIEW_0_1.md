# Adversarial + constructive review (pre-0.1.0)

Reviewer: same agent that scaffolded Phase 0. Date: 2026-08-17.

## Adversarial (what was wrong)

| # | Finding | Severity | Fix in 0.1.0 |
|---|---------|----------|----------------|
| A1 | `cuda` / `python` features compiled and `*_feature_enabled()` could be true while doing nothing. Easy to misread as “CUDA works.” | High | Features documented as reserved no-ops. `bridge_ready()` **never** follows them. Tests run `--all-features` and still assert `!ready`. |
| A2 | `load_ptx(name, ptx) -> Result<(), _>` was not the Phase 1 shape. Linking it from unsloth would force a break. | High | `LoadedModule`, `LaunchSpec`, `load_cubin`, `launch`. Fields on `LoadedModule` are private. |
| A3 | `unsafe_code = forbid` with no escape hatch documented. Phase 1 driver pointers need `unsafe`. | Med | Forbid stays for 0.1.0 (no GPU code). CONTRIBUTING + Cargo.toml comment: Phase 1 opens a tiny `cuda` window, not a crate-wide allow. |
| A4 | No Commitizen / changelog / 0.x policy. Easy to tag `1.0.0` by accident. | Med | `.cz.toml` `major_version_zero = true`. |
| A5 | CI was `cargo test --all-features` only. No fmt, no clippy deny, no package dry-run. | Med | Expanded workflow. |
| A6 | `rust-version = 1.85` vs CI `stable`. Drift. | Low | CI uses `1.85` for MSRV + `stable` for clippy. |
| A7 | Honesty test only ran default features, so A1 was untested. | High | Split tests; validation vs `NotReady`. |
| A8 | Crate description / keywords said “triton” in a way that implied a port. | Low | Description + README lead with “contract, not a port.” Dropped `triton` keyword. |
| A9 | No GPU handoff. Next step would stall in a CPU sandbox. | Med | `docs/GPU_HANDOFF.md`. |
| A10 | `BridgeError` was a single `NotImplemented` string. Callers cannot `match` stably. | Med | `#[non_exhaustive]` enum with `NotReady`. |
| A11 | Empty PTX returned the same error as “not implemented.” | Low | `InvalidModule` vs `NotReady`. |
| A12 | Leaf rule was prose only. | Low | `FORBIDDEN_DEP_PREFIXES` + `include_str!` test. |
| A13 | `Cargo.lock` gitignored while a lib with zero deps. Fine, but CI shouldn’t require a lock. | Info | Still gitignored. |
| A14 | Phase 0 README claimed “scaffold only” but issues #2–#7 were already the real backlog. Risk of treating docs as implementation. | Info | README status table unchanged in meaning; 0.1.0 is explicitly a **contract release**. |

## Constructive (what to do next — not in 0.1.0)

1. **Do not crates.io-depend this from unsloth-rs until Phase 1.** A 0.1.0 crates.io crate that always returns `NotReady` is publishable; wiring it as a hard optional dep just to print `false` is noise. unsloth gets a hook module + reserved feature.
2. **First GPU job is one FA cubin**, not a compiler. Issue #3.
3. **Prefer `scripts/compile_kernels.py` over in-process CPython** (#4).
4. **Do not relax `bridge_ready` for “we loaded PTX into host memory.”** Ready means device-pointer launch.
5. Keep this crate a leaf. If a DSL grows, spin out (issue #5).

## Bugs found in 0.0.1 stub

None that execute (there is no loader). The bugs were **API / honesty / process**. All A1–A12 items above are addressed in 0.1.0 except A3’s future unsafe window (documented, not opened).
