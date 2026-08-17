// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Load / launch surface.
//!
//! * Default features: validate arguments, then [`BridgeError::NotReady`].
//! * `--features cuda`: real `cuModuleLoadData` / `cuLaunchKernel`. No device
//!   → [`BridgeError::Device`] (`FAIL_ENV`).

use crate::args::KernelArg;
use crate::error::BridgeError;
use crate::ptx::{validate_cubin, validate_ptx};

/// Opaque loaded module.
///
/// With `cuda`, `handle` is a `CUmodule` stored as `usize` (0 = none).
#[derive(Debug, Clone)]
pub struct LoadedModule {
    name: String,
    sm: Option<u32>,
    handle: usize,
}

impl LoadedModule {
    /// Module name passed to [`load_ptx`] / [`load_cubin`].
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Target SM, if the caller supplied one.
    #[must_use]
    pub const fn sm(&self) -> Option<u32> {
        self.sm
    }

    /// `true` only when this handle refers to a device-resident module.
    #[must_use]
    pub const fn is_device_resident(&self) -> bool {
        self.handle != 0
    }
}

/// Launch configuration.
#[derive(Debug, Clone, PartialEq)]
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
    /// Kernel arguments. Device pointers must already be on the GPU.
    pub args: &'a [KernelArg],
}

impl<'a> LaunchSpec<'a> {
    /// Build a spec with empty args. Does not validate; [`launch`] does.
    #[must_use]
    pub const fn new(kernel: &'a str) -> Self {
        Self {
            kernel,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem_bytes: 0,
            device_ordinal: 0,
            args: &[],
        }
    }

    /// Attach device-resident arguments.
    #[must_use]
    pub fn with_args(mut self, args: &'a [KernelArg]) -> Self {
        self.args = args;
        self
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

fn validate_spec(spec: &LaunchSpec<'_>) -> Result<(), BridgeError> {
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
    Ok(())
}

/// Load PTX text.
///
/// # Errors
///
/// * [`BridgeError::InvalidModule`] — empty / not PTX
/// * [`BridgeError::NotReady`] — default features (no driver)
/// * [`BridgeError::Device`] — `cuda` feature but no driver/device (`FAIL_ENV`)
pub fn load_ptx(name: &str, ptx: &str, sm: Option<u32>) -> Result<LoadedModule, BridgeError> {
    reject_empty_name(name)?;
    validate_ptx(ptx)?;
    load_bytes(name, ptx.as_bytes(), sm)
}

/// Load CUBIN bytes. Same contract as [`load_ptx`].
pub fn load_cubin(name: &str, bytes: &[u8], sm: Option<u32>) -> Result<LoadedModule, BridgeError> {
    reject_empty_name(name)?;
    validate_cubin(bytes)?;
    load_bytes(name, bytes, sm)
}

fn load_bytes(name: &str, data: &[u8], sm: Option<u32>) -> Result<LoadedModule, BridgeError> {
    #[cfg(feature = "cuda")]
    {
        let module = crate::cuda_loader::load_module_data(name, data)?;
        Ok(LoadedModule {
            name: name.into(),
            sm,
            handle: module as usize,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (data, sm);
        let _ = name;
        Err(BridgeError::NotReady)
    }
}

/// Launch a kernel from a loaded module.
pub fn launch(module: &LoadedModule, spec: &LaunchSpec<'_>) -> Result<(), BridgeError> {
    validate_spec(spec)?;
    #[cfg(feature = "cuda")]
    {
        if module.handle == 0 {
            return Err(BridgeError::NotReady);
        }
        crate::cuda_loader::launch_module(
            module.handle as *mut std::ffi::c_void,
            spec.kernel,
            spec.grid,
            spec.block,
            spec.shared_mem_bytes,
            spec.args,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = module;
        Err(BridgeError::NotReady)
    }
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
    fn well_formed_ptx_default_is_not_ready_or_fail_env() {
        let err = load_ptx("fa", ".version 8.0\n.target sm_90\n", Some(90)).unwrap_err();
        // Default features → NotReady. `--features cuda` without a GPU → Device.
        assert!(err.is_not_ready() || matches!(err, BridgeError::Device { .. }));
    }

    #[test]
    fn well_formed_cubin_default_is_not_ready_or_fail_env() {
        let err = load_cubin("fa", &[0x7f, b'E', b'L', b'F'], Some(90)).unwrap_err();
        assert!(err.is_not_ready() || matches!(err, BridgeError::Device { .. }));
    }

    #[test]
    fn launch_validates_before_backend() {
        let module = LoadedModule {
            name: "fa".into(),
            sm: Some(90),
            handle: 0,
        };
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
        let err = launch(&module, &ok).unwrap_err();
        assert!(err.is_not_ready() || matches!(err, BridgeError::Device { .. }));
    }
}
