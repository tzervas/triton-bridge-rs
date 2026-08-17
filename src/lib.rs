// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Temporary Triton / PTX launch **contract**.
//!
//! This crate is **0.1.x**: the public API is real, the implementation is not.
//! [`bridge_ready`] is `false`. [`load_ptx`], [`load_cubin`], and [`launch`]
//! validate arguments and then return [`BridgeError::NotReady`].
//!
//! unsloth-rs must keep using Candle `CustomOp*` as the default. This crate
//! is the future home of *foreign* kernel load/launch, not transformer math.
//!
//! ```
//! use triton_bridge::{bridge_ready, load_ptx};
//! assert!(!bridge_ready());
//! assert!(load_ptx("flash", ".version 8.0\n", Some(90)).is_err());
//! ```

#![warn(missing_docs)]

mod api;
mod error;

pub use api::{launch, load_cubin, load_ptx, LaunchSpec, LoadedModule};
pub use error::BridgeError;

/// Semantic crate version (same as `Cargo.toml`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `false` until Phase 1 can load PTX/CUBIN and launch on a device pointer.
///
/// Callers (unsloth-rs, axolotl) **must** branch on this. Enabling the
/// `cuda` or `python` cargo features does **not** flip this to `true`.
#[must_use]
pub const fn bridge_ready() -> bool {
    false
}

/// Phase 2 `python` feature is compiled in. Still does **not** mean ready.
#[must_use]
pub const fn python_feature_enabled() -> bool {
    cfg!(feature = "python")
}

/// CUDA launch feature is compiled in. Still does **not** mean ready.
#[must_use]
pub const fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

/// Why [`bridge_ready`] is false (stable string for logs / issues).
#[must_use]
pub const fn not_ready_reason() -> &'static str {
    "0.1 contract: no PTX/CUBIN loader, no Triton FFI. See tzervas/triton-bridge-rs#3"
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
    fn honesty_default_and_features() {
        assert!(!bridge_ready());
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
        assert!(VERSION.starts_with("0."));
        // Features may be on in `--all-features` CI. Ready must stay false.
        let _ = python_feature_enabled();
        let _ = cuda_feature_enabled();
        assert!(!bridge_ready());
    }

    #[test]
    fn error_display_not_ready() {
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
            // Naive but good enough: a dep line would be `name =`
            let needle = format!("{name} =");
            assert!(
                !toml.contains(&needle),
                "leaf rule: Cargo.toml must not depend on {name}"
            );
        }
    }
}
