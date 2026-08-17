# Debt

| ID | Item | Status |
|----|------|--------|
| TB-00 | Implementation is a contract stub | Honest — `bridge_ready() == false` (0.1.0) |
| TB-01 | No PTX assets | Phase 1 — #3 |
| TB-02 | No CUDA CI | GPU_HANDOFF.md; FAIL_ENV same class as unsloth-rs |
| TB-03 | Triton source license/NOTICE | File before vendoring any Unsloth PTX |
| TB-04 | libtriton C API unstable | Do not hard-dep |
| TB-05 | Must not take peft/axolotl/unsloth deps | Tested in `cargo_toml_has_no_forbidden_deps` |
| TB-06 | `cuda`/`python` features are empty | Documented no-ops; do not flip ready |
| TB-07 | `unsafe_code = forbid` vs Phase 1 pointers | Open a tiny window under `cuda` only when #3 starts |
| TB-08 | Not on crates.io until this tag is published | Optional; git tag `v0.1.0` is the distribute point |

When Phase 1 lands: “unsloth-rs still defaults to CustomOp; this crate is opt-in.”
