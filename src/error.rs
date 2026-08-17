// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Errors. Stable enough for consumers to `match` on [`BridgeError::NotReady`].

use std::fmt;

/// Load / launch failure.
///
/// `NotReady` is the only variant 0.1.0 produces for well-formed input.
/// Other variants exist so Phase 1 does not have to break `match`es.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BridgeError {
    /// Phase 1 has not landed. [`crate::bridge_ready`] is `false`.
    NotReady,
    /// Caller passed junk (empty name, empty PTX, zero grid, …).
    InvalidModule {
        /// Why the module or spec was rejected.
        reason: String,
    },
    /// CUDA / device error (unused until Phase 1).
    Device {
        /// `cudaSetDevice` ordinal.
        ordinal: u32,
        /// Driver or runtime message.
        message: String,
    },
    /// Kernel launch failed after a module was loaded (unused until Phase 1).
    Launch {
        /// Kernel symbol.
        kernel: String,
        /// Driver or runtime message.
        message: String,
    },
}

impl BridgeError {
    /// True when the crate cannot do the requested work yet (not a caller bug).
    #[must_use]
    pub const fn is_not_ready(&self) -> bool {
        matches!(self, Self::NotReady)
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReady => write!(
                f,
                "triton-bridge 0.1: not ready ({})",
                crate::not_ready_reason()
            ),
            Self::InvalidModule { reason } => write!(f, "triton-bridge: invalid module: {reason}"),
            Self::Device { ordinal, message } => {
                write!(f, "triton-bridge: device {ordinal}: {message}")
            }
            Self::Launch { kernel, message } => {
                write!(f, "triton-bridge: launch {kernel}: {message}")
            }
        }
    }
}

impl std::error::Error for BridgeError {}
