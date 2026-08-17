// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Phase 1 smoke: load `identity_f32.ptx` and copy a vector.
//!
//! CPU / no libcuda → `NotReady` or `FAIL_ENV` (honest).
//! 5080 with `--features cuda` → launch and assert byte-identical copy.

use triton_bridge::{
    device_alloc, device_free, launch, load_ptx, memcpy_dtoh, memcpy_htod, KernelArg, LaunchSpec,
};

const PTX: &str = include_str!("../precompiled/identity_f32.ptx");

#[test]
fn identity_ptx_loads_or_fail_env() {
    match load_ptx("identity", PTX, Some(90)) {
        Ok(module) => {
            assert!(module.is_device_resident());
            let n = 64_u32;
            let host: Vec<f32> = (0_u16..64).map(|i| f32::from(i) * 0.5).collect();
            let bytes: Vec<u8> = host
                .iter()
                .flat_map(|f| f.to_bits().to_le_bytes())
                .collect();
            let din = device_alloc(bytes.len()).expect("alloc in");
            let dout = device_alloc(bytes.len()).expect("alloc out");
            memcpy_htod(din, &bytes).expect("htod");
            let args = [
                KernelArg::DevicePtr(din),
                KernelArg::DevicePtr(dout),
                KernelArg::U32(n),
            ];
            let spec = LaunchSpec::new("identity_f32").with_args(&args);
            let mut spec = spec;
            spec.grid = [1, 1, 1];
            spec.block = [n, 1, 1];
            launch(&module, &spec).expect("launch identity_f32");
            let mut out = vec![0u8; bytes.len()];
            memcpy_dtoh(&mut out, dout).expect("dtoh");
            device_free(din).ok();
            device_free(dout).ok();
            assert_eq!(out, bytes, "identity copy must be bit-exact");
        }
        Err(e) => {
            assert!(
                e.is_not_ready() || e.is_fail_env(),
                "unexpected error (not FAIL_ENV/NotReady): {e}"
            );
        }
    }
}
