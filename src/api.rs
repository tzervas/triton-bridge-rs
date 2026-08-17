// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Load / launch surface. 0.1.0 validates arguments and returns [`BridgeError::NotReady`].

use crate::error::BridgeError;

/// Opaque loaded module. Cannot be constructed by callers in 0.1.0.
///
/// Phase 1 will hold a CUDA module handle here. Fields stay private so that
/// adding a handle is not a breaking change.
#[derive(Debug, Clone)]
pub struct LoadedModule {
    name: String,
}

impl LoadedModule {
    /// Module name passed to [`load_ptx`] / [`load_cubin`].
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `true` only when this handle refers to a device-resident module.
    ///
    /// Always `false` in 0.1.0 (nothing is loaded).
    #[must_use]
    pub const fn is_device_resident(&self) -> bool {
        false
    }
}

/// Launch configuration. Device pointers are **not** accepted in 0.1.0
/// (that would require `unsafe` and a real module).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchSpec<'a> {
    /// Kernel symbol inside the module.
    pub kernel: &'a str,
    /// Grid dimensions (x, y, z).
    pub grid: [u32; 3],
    /// Block dimensions (x, y, z).
    pub block: [u32; 3],
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// CUDA device ordinal.
    pub device_ordinal: u32,
}

impl<'a> LaunchSpec<'a> {
    /// Build a spec. Does not validate; [`launch`] does.
    #[must_use]
    pub const fn new(kernel: &'a str) -> Self {
        Self {
            kernel,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem_bytes: 0,
            device_ordinal: 0,
        }
    }
}

fn reject_empty_name(name: &str) -> Result<(), BridgeError> {
    if name.is_empty() {
        return Err(BridgeError::InvalidModule {
            reason: "empty module name".into(),
        });
    }
    Ok(())
}

/// Load PTX text. Always [`BridgeError::NotReady`] if `name` and `ptx` are non-empty.
///
/// `sm` is the target compute capability (e.g. `90`, `120`). Stored for Phase 1;
/// ignored in 0.1.0.
///
/// # Errors
///
/// * [`BridgeError::InvalidModule`] — empty name or empty PTX
/// * [`BridgeError::NotReady`] — well-formed input, loader not implemented
pub fn load_ptx(name: &str, ptx: &str, sm: Option<u32>) -> Result<LoadedModule, BridgeError> {
    let _ = sm;
    reject_empty_name(name)?;
    if ptx.is_empty() {
        return Err(BridgeError::InvalidModule {
            reason: "empty PTX".into(),
        });
    }
    Err(BridgeError::NotReady)
}

/// Load CUBIN bytes. Same contract as [`load_ptx`].
///
/// # Errors
///
/// * [`BridgeError::InvalidModule`] — empty name or empty cubin
/// * [`BridgeError::NotReady`] — well-formed input, loader not implemented
pub fn load_cubin(name: &str, bytes: &[u8], sm: Option<u32>) -> Result<LoadedModule, BridgeError> {
    let _ = sm;
    reject_empty_name(name)?;
    if bytes.is_empty() {
        return Err(BridgeError::InvalidModule {
            reason: "empty CUBIN".into(),
        });
    }
    Err(BridgeError::NotReady)
}

/// Launch a kernel from a loaded module. 0.1.0 never succeeds.
///
/// # Errors
///
/// * [`BridgeError::InvalidModule`] — empty kernel name or zero grid/block
/// * [`BridgeError::NotReady`] — well-formed spec (no module can exist yet)
pub fn launch(module: &LoadedModule, spec: &LaunchSpec<'_>) -> Result<(), BridgeError> {
    let _ = module;
    if spec.kernel.is_empty() {
        return Err(BridgeError::InvalidModule {
            reason: "empty kernel name".into(),
        });
    }
    if spec.grid[0] == 0 || spec.block[0] == 0 {
        return Err(BridgeError::InvalidModule {
            reason: "grid[0] and block[0] must be non-zero".into(),
        });
    }
    Err(BridgeError::NotReady)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_name_is_invalid_not_not_ready() {
        let err = load_ptx("", ".version 8.0", Some(90)).unwrap_err();
        assert!(!err.is_not_ready());
        assert!(matches!(err, BridgeError::InvalidModule { .. }));
    }

    #[test]
    fn empty_ptx_is_invalid() {
        assert!(matches!(
            load_ptx("k", "", None).unwrap_err(),
            BridgeError::InvalidModule { .. }
        ));
    }

    #[test]
    fn empty_cubin_is_invalid() {
        assert!(matches!(
            load_cubin("k", &[], None).unwrap_err(),
            BridgeError::InvalidModule { .. }
        ));
    }

    #[test]
    fn well_formed_ptx_is_not_ready() {
        let err = load_ptx("fa", ".version 8.0\n.target sm_90\n", Some(90)).unwrap_err();
        assert!(err.is_not_ready());
        assert_eq!(err, BridgeError::NotReady);
    }

    #[test]
    fn well_formed_cubin_is_not_ready() {
        assert_eq!(
            load_cubin("fa", &[0x7f, b'E', b'L', b'F'], Some(90)).unwrap_err(),
            BridgeError::NotReady
        );
    }

    #[test]
    fn launch_validates_before_not_ready() {
        // Cannot construct LoadedModule from outside — test uses a dummy via
        // the fact that launch doesn't need a real one... we need *some* value.
        // 0.1.0: only this crate can build it. Use the test-only ctor.
        let module = LoadedModule { name: "fa".into() };
        assert_eq!(module.name(), "fa");
        assert!(!module.is_device_resident());

        let bad = LaunchSpec::new("");
        assert!(matches!(
            launch(&module, &bad).unwrap_err(),
            BridgeError::InvalidModule { .. }
        ));

        let mut zero = LaunchSpec::new("k");
        zero.grid = [0, 1, 1];
        assert!(matches!(
            launch(&module, &zero).unwrap_err(),
            BridgeError::InvalidModule { .. }
        ));

        let ok = LaunchSpec::new("flash_fwd");
        assert_eq!(launch(&module, &ok).unwrap_err(), BridgeError::NotReady);
    }
}
