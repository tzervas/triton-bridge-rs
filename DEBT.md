# Debt

| ID | Item | Status |
|----|------|--------|
| TB-00 | Crate is a stub | Honest — `bridge_ready() == false` |
| TB-01 | No PTX assets | Phase 1 |
| TB-02 | No CUDA CI | Same FAIL_ENV class as unsloth-rs GPU_SETUP |
| TB-03 | Triton source license/NOTICE | File before vendoring any Unsloth PTX |
| TB-04 | libtriton C API unstable | Do not hard-dep |
| TB-05 | Must not take peft/axolotl deps | Leaf rule |

When Phase 1 lands, add a row: “unsloth-rs still defaults to CustomOp; this
crate is opt-in.”
