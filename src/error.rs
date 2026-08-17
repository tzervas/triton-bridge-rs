// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Errors. Stable enough for consumers to `match` on [`BridgeError::NotReady`].

use std::fmt;

/// Load / launch failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BridgeError {
    /// No driver linked, or caller used the default feature set.
    NotReady,
    /// Caller passed junk (empty name, empty PTX, zero grid, …).
    InvalidModule {
        /// Why the module or spec was rejected.
        reason: String,
    },
    /// CUDA / device error. Message starts with `FAIL_ENV` when the host
    /// has no driver or no device (honest; not a silent skip).
    Device {
        /// `cudaSetDevice` ordinal.
        ordinal: u32,
        /// Driver or runtime message.
        message: String,
    },
    /// Kernel launch failed after a module was loaded.
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

    /// True when the environment cannot run CUDA (no libcuda / no device).
    #[must_use]
    pub fn is_fail_env(&self) -> bool {
        match self {
            Self::Device { message, .. } => message.contains("FAIL_ENV"),
            _ => false,
        }
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReady => write!(
                f,
                "triton-bridge: not ready ({})",
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
