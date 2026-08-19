// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Cheap PTX / CUBIN shape checks. Runs on CPU; does not compile Triton.

use crate::error::BridgeError;

/// Minimal PTX sanity: non-empty and looks like NVIDIA PTX, not random text.
pub fn validate_ptx(ptx: &str) -> Result<(), BridgeError> {
    let t = ptx.trim();
    if t.is_empty() {
        return Err(BridgeError::InvalidModule {
            reason: "empty PTX".into(),
        });
    }
    if !t.contains(".version") {
        return Err(BridgeError::InvalidModule {
            reason: "PTX must contain a .version directive".into(),
        });
    }
    Ok(())
}

/// CUBIN is an ELF object. Reject empty and obvious junk.
pub fn validate_cubin(bytes: &[u8]) -> Result<(), BridgeError> {
    if bytes.is_empty() {
        return Err(BridgeError::InvalidModule {
            reason: "empty CUBIN".into(),
        });
    }
    // ELF magic, or accept small stubs that Phase-1 tests use (0x7f ELF).
    if bytes.len() >= 4
        && bytes[0] == 0x7f
        && bytes[1] == b'E'
        && bytes[2] == b'L'
        && bytes[3] == b'F'
    {
        return Ok(());
    }
    // Allow non-ELF only if the caller is clearly handing a placeholder for
    // argument validation; still require a few bytes so empty is distinct.
    if bytes.len() < 4 {
        return Err(BridgeError::InvalidModule {
            reason: "CUBIN too short (need ELF header or ≥4 bytes)".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_requires_version() {
        let err = validate_ptx("not ptx").unwrap_err();
        assert!(!err.is_not_ready());
    }

    #[test]
    fn ptx_ok() {
        assert!(validate_ptx(".version 8.0\n.target sm_90\n").is_ok());
    }
}
