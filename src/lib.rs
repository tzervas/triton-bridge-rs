// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Temporary Triton / PTX launch bridge.
//!
//! This crate is a **contract + stub**. [`bridge_ready`] is `false` until
//! Phase 1 loads a real module (see repo `ROADMAP.md`).
//!
//! unsloth-rs must keep using Candle `CustomOp*` as the default. This crate
//! is the future home of *foreign* kernel load/launch, not transformer math.

#![warn(missing_docs)]

/// Semantic crate version (keep in sync with `Cargo.toml`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `false` until Phase 1 can load PTX/CUBIN and launch on a device pointer.
///
/// Callers (unsloth-rs, axolotl) **must** branch on this instead of assuming
/// a Triton runtime exists.
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
    "Phase 0 stub: no PTX/CUBIN loader, no Triton FFI. See tzervas/triton-bridge-rs#1"
}

/// Error type for future load/launch APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// Called an API that does not exist yet.
    NotImplemented(&'static str),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented(msg) => write!(f, "triton-bridge: {msg}"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Placeholder so the public surface is obvious. Always [`BridgeError::NotImplemented`].
///
/// # Errors
///
/// Always. Phase 1 will take PTX bytes + a device ordinal.
pub fn load_ptx(_name: &str, _ptx: &str) -> Result<(), BridgeError> {
    Err(BridgeError::NotImplemented(not_ready_reason()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honesty() {
        assert!(!bridge_ready());
        assert!(!python_feature_enabled());
        assert!(!cuda_feature_enabled());
        assert!(load_ptx("x", "").is_err());
    }
}
