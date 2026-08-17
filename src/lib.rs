// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Triton / PTX launch **bridge**.
//!
//! * **0.2** implements Phase 1: CUDA driver load/launch behind `--features cuda`.
//! * Default features still do **not** link CUDA. [`bridge_ready`] is then `false`.
//! * unsloth-rs keeps Candle `CustomOp*` as the default math path. This crate
//!   is for *foreign* kernels (Flash Attention CUBIN first), not RMS/RoPE/CE.
//!
//! ```
//! use triton_bridge::{bridge_ready, load_ptx};
//! if !bridge_ready() {
//!     assert!(load_ptx("flash", ".version 8.0\n", Some(90)).is_err());
//! }
//! ```

#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![warn(missing_docs)]

mod api;
mod args;
mod catalog;
mod error;
mod ptx;

#[cfg(feature = "cuda")]
mod cuda_loader;

pub use api::{
    device_alloc, device_free, launch, load_cubin, load_ptx, memcpy_dtoh, memcpy_htod, LaunchSpec,
    LoadedModule,
};
pub use args::KernelArg;
pub use catalog::{lookup, KernelEntry, KernelHome, KERNEL_CATALOG};
pub use error::BridgeError;

/// Semantic crate version (same as `Cargo.toml`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `true` only when the `cuda` feature is on **and** the driver sees a device.
///
/// Enabling the `cuda` Cargo feature on a CPU box does **not** flip this.
/// Enabling `python` never flips this.
#[must_use]
pub fn bridge_ready() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_loader::driver_ready()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Phase 2 `python` feature is compiled in. Still does **not** mean ready.
#[must_use]
pub const fn python_feature_enabled() -> bool {
    cfg!(feature = "python")
}

/// CUDA launch feature is compiled in. Still does **not** mean a device exists.
#[must_use]
pub const fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

/// Why [`bridge_ready`] is false (stable string for logs / issues).
#[must_use]
pub fn not_ready_reason() -> &'static str {
    if cfg!(feature = "cuda") {
        "0.2 cuda feature: driver missing, no device, or cuInit failed (FAIL_ENV). See tzervas/triton-bridge-rs#3"
    } else {
        "0.2 default: no CUDA driver linked. Build with --features cuda on a machine with libcuda."
    }
}

/// Names this crate must never depend on (leaf rule, issue #7).
pub const FORBIDDEN_DEP_PREFIXES: &[&str] = &[
    "peft-rs",
    "qlora-rs",
    "axolotl-rs",
    "rust-ai-core",
    "unsloth-rs",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honesty_default_features() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
        assert!(VERSION.starts_with("0."));
        if !cuda_feature_enabled() {
            assert!(!bridge_ready());
        }
        let _ = python_feature_enabled();
    }

    #[test]
    fn error_display_mentions_bridge() {
        let s = BridgeError::NotReady.to_string();
        assert!(s.contains("not ready"), "{s}");
    }

    #[test]
    fn leaf_prefixes_are_listed() {
        assert!(FORBIDDEN_DEP_PREFIXES.contains(&"peft-rs"));
        assert!(FORBIDDEN_DEP_PREFIXES.contains(&"unsloth-rs"));
    }

    #[test]
    fn cargo_toml_has_no_forbidden_deps() {
        let toml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        for name in FORBIDDEN_DEP_PREFIXES {
            let needle = format!("{name} =");
            assert!(
                !toml.contains(&needle),
                "leaf rule: Cargo.toml must not depend on {name}"
            );
        }
    }
}
