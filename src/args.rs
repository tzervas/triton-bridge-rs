// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Kernel launch arguments. Device pointers are `u64` (`CUdeviceptr`).

/// One launch argument. Pointers must already live on the device.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelArg {
    /// Device pointer (`CUdeviceptr` as `u64`). Never a host `*const T`.
    DevicePtr(u64),
    /// Signed 32-bit.
    I32(i32),
    /// Unsigned 32-bit.
    U32(u32),
    /// `f32` bit pattern (avoid sending host floats through ABI by accident).
    F32Bits(u32),
    /// Signed 64-bit (sizes, strides).
    I64(i64),
}

impl KernelArg {
    /// Pack an `f32` as bits.
    #[must_use]
    pub fn f32(v: f32) -> Self {
        Self::F32Bits(v.to_bits())
    }
}
