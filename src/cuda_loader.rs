// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Phase 1 CUDA driver loader. Compiled only with `--features cuda`.
//!
//! Uses `libloading` against `libcuda.so.1` so we do not pin a `cudarc`
//! version. No device → [`crate::error::BridgeError::Device`] with `FAIL_ENV`
//! in the message (honest; never a silent pass).
//!
//! `bridge_ready()` is true only after `cuInit` + at least one device.

#![allow(unsafe_code)]
#![allow(
    clippy::borrow_as_ptr,
    clippy::ref_as_ptr,
    clippy::ptr_as_ptr,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::struct_field_names
)]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use libloading::Library;

use crate::args::KernelArg;
use crate::error::BridgeError;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

struct Fns {
    cu_init: unsafe extern "C" fn(c_uint) -> CUresult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CUresult,
    cu_device_get: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    cu_module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cu_launch_kernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    cu_get_error_string: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
}

/// Process-wide driver + primary context. Serialized on a mutex.
struct DriverState {
    _lib: Library,
    fns: Fns,
    ctx: CUcontext,
    device_count: i32,
}

unsafe impl Send for DriverState {}

static DRIVER: OnceLock<Mutex<Option<Result<DriverState, String>>>> = OnceLock::new();

fn err_string(fns: &Fns, code: CUresult) -> String {
    unsafe {
        let mut ptr: *const c_char = std::ptr::null();
        if (fns.cu_get_error_string)(code, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
            return std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        }
    }
    format!("CUresult={code}")
}

fn open_lib() -> Result<Library, String> {
    let names = [
        "libcuda.so.1",
        "libcuda.so",
        "nvcuda.dll",
        "/usr/lib/wsl/lib/libcuda.so.1",
    ];
    let mut last = "no candidate".to_string();
    for n in names {
        match unsafe { Library::new(n) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last = format!("{n}: {e}"),
        }
    }
    Err(format!("FAIL_ENV: libcuda not found ({last})"))
}

fn load_fns(lib: &Library) -> Result<Fns, String> {
    unsafe {
        Ok(Fns {
            cu_init: *lib.get(b"cuInit\0").map_err(|e| e.to_string())?,
            cu_device_get_count: *lib.get(b"cuDeviceGetCount\0").map_err(|e| e.to_string())?,
            cu_device_get: *lib.get(b"cuDeviceGet\0").map_err(|e| e.to_string())?,
            cu_ctx_create: *lib
                .get(b"cuCtxCreate_v2\0")
                .or_else(|_| lib.get(b"cuCtxCreate\0"))
                .map_err(|e| e.to_string())?,
            cu_module_load_data: *lib.get(b"cuModuleLoadData\0").map_err(|e| e.to_string())?,
            cu_module_get_function: *lib
                .get(b"cuModuleGetFunction\0")
                .map_err(|e| e.to_string())?,
            cu_launch_kernel: *lib.get(b"cuLaunchKernel\0").map_err(|e| e.to_string())?,
            cu_get_error_string: *lib.get(b"cuGetErrorString\0").map_err(|e| e.to_string())?,
            cu_mem_alloc: *lib
                .get(b"cuMemAlloc_v2\0")
                .or_else(|_| lib.get(b"cuMemAlloc\0"))
                .map_err(|e| e.to_string())?,
            cu_mem_free: *lib
                .get(b"cuMemFree_v2\0")
                .or_else(|_| lib.get(b"cuMemFree\0"))
                .map_err(|e| e.to_string())?,
            cu_memcpy_htod: *lib
                .get(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| lib.get(b"cuMemcpyHtoD\0"))
                .map_err(|e| e.to_string())?,
            cu_memcpy_dtoh: *lib
                .get(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| lib.get(b"cuMemcpyDtoH\0"))
                .map_err(|e| e.to_string())?,
            cu_ctx_synchronize: *lib.get(b"cuCtxSynchronize\0").map_err(|e| e.to_string())?,
        })
    }
}

fn init_state() -> Result<DriverState, String> {
    let lib = open_lib()?;
    let fns = load_fns(&lib)?;
    unsafe {
        let r = (fns.cu_init)(0);
        if r != CUDA_SUCCESS {
            return Err(format!("FAIL_ENV: cuInit: {}", err_string(&fns, r)));
        }
        let mut count: c_int = 0;
        let r = (fns.cu_device_get_count)(&mut count);
        if r != CUDA_SUCCESS {
            return Err(format!(
                "FAIL_ENV: cuDeviceGetCount: {}",
                err_string(&fns, r)
            ));
        }
        if count < 1 {
            return Err("FAIL_ENV: no CUDA device (cuDeviceGetCount=0)".into());
        }
        let mut dev: CUdevice = 0;
        let r = (fns.cu_device_get)(&mut dev, 0);
        if r != CUDA_SUCCESS {
            return Err(format!("FAIL_ENV: cuDeviceGet: {}", err_string(&fns, r)));
        }
        let mut ctx: CUcontext = std::ptr::null_mut();
        let r = (fns.cu_ctx_create)(&mut ctx, 0, dev);
        if r != CUDA_SUCCESS {
            return Err(format!("FAIL_ENV: cuCtxCreate: {}", err_string(&fns, r)));
        }
        Ok(DriverState {
            _lib: lib,
            fns,
            ctx,
            device_count: count,
        })
    }
}

fn with_driver<T>(
    f: impl FnOnce(&DriverState) -> Result<T, BridgeError>,
) -> Result<T, BridgeError> {
    let slot = DRIVER.get_or_init(|| Mutex::new(None));
    let mut guard = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.is_none() {
        *guard = Some(init_state());
    }
    match guard.as_ref().expect("just set") {
        Ok(state) => f(state),
        Err(msg) => Err(BridgeError::Device {
            ordinal: 0,
            message: msg.clone(),
        }),
    }
}

/// `true` when the driver initialized and at least one device exists.
pub fn driver_ready() -> bool {
    with_driver(|s| Ok(s.device_count > 0 && !s.ctx.is_null())).unwrap_or(false)
}

/// Load PTX or CUBIN bytes via `cuModuleLoadData`.
pub fn load_module_data(name: &str, data: &[u8]) -> Result<CUmodule, BridgeError> {
    let mut buf = Vec::with_capacity(data.len() + 1);
    buf.extend_from_slice(data);
    buf.push(0);
    with_driver(|state| unsafe {
        let mut module: CUmodule = std::ptr::null_mut();
        let r = (state.fns.cu_module_load_data)(&mut module, buf.as_ptr().cast());
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Device {
                ordinal: 0,
                message: format!("cuModuleLoadData({name}): {}", err_string(&state.fns, r)),
            });
        }
        Ok(module)
    })
}

/// Launch `kernel` from an already-loaded module.
pub fn launch_module(
    module: CUmodule,
    kernel: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem_bytes: u32,
    args: &[KernelArg],
) -> Result<(), BridgeError> {
    let cname = CString::new(kernel).map_err(|_| BridgeError::InvalidModule {
        reason: "kernel name contains NUL".into(),
    })?;
    with_driver(|state| unsafe {
        let mut func: CUfunction = std::ptr::null_mut();
        let r = (state.fns.cu_module_get_function)(&raw mut func, module, cname.as_ptr());
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Launch {
                kernel: kernel.into(),
                message: err_string(&state.fns, r),
            });
        }

        let mut i32s: Vec<i32> = Vec::new();
        let mut u32s: Vec<u32> = Vec::new();
        let mut i64s: Vec<i64> = Vec::new();
        let mut ptrs: Vec<CUdeviceptr> = Vec::new();
        let mut raw: Vec<*mut c_void> = Vec::with_capacity(args.len());

        for a in args {
            match *a {
                KernelArg::DevicePtr(p) => {
                    ptrs.push(p);
                    raw.push(std::ptr::from_mut(ptrs.last_mut().unwrap()).cast());
                }
                KernelArg::I32(v) => {
                    i32s.push(v);
                    raw.push(std::ptr::from_mut(i32s.last_mut().unwrap()).cast());
                }
                KernelArg::U32(v) | KernelArg::F32Bits(v) => {
                    u32s.push(v);
                    raw.push(std::ptr::from_mut(u32s.last_mut().unwrap()).cast());
                }
                KernelArg::I64(v) => {
                    i64s.push(v);
                    raw.push(std::ptr::from_mut(i64s.last_mut().unwrap()).cast());
                }
            }
        }

        let r = (state.fns.cu_launch_kernel)(
            func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem_bytes,
            std::ptr::null_mut(),
            raw.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Launch {
                kernel: kernel.into(),
                message: err_string(&state.fns, r),
            });
        }
        let r = (state.fns.cu_ctx_synchronize)();
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Launch {
                kernel: kernel.into(),
                message: format!("cuCtxSynchronize: {}", err_string(&state.fns, r)),
            });
        }
        Ok(())
    })
}

/// Allocate `bytes` on the current context. Returns `CUdeviceptr` as `u64`.
pub fn mem_alloc(bytes: usize) -> Result<u64, BridgeError> {
    if bytes == 0 {
        return Err(BridgeError::InvalidModule {
            reason: "device_alloc: zero bytes".into(),
        });
    }
    with_driver(|state| unsafe {
        let mut ptr: CUdeviceptr = 0;
        let r = (state.fns.cu_mem_alloc)(&raw mut ptr, bytes);
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Device {
                ordinal: 0,
                message: format!("cuMemAlloc({bytes}): {}", err_string(&state.fns, r)),
            });
        }
        Ok(ptr)
    })
}

/// Free a device pointer from [`mem_alloc`].
pub fn mem_free(ptr: u64) -> Result<(), BridgeError> {
    if ptr == 0 {
        return Ok(());
    }
    with_driver(|state| unsafe {
        let r = (state.fns.cu_mem_free)(ptr);
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Device {
                ordinal: 0,
                message: format!("cuMemFree: {}", err_string(&state.fns, r)),
            });
        }
        Ok(())
    })
}

/// Host → device copy.
pub fn memcpy_htod(dst: u64, src: &[u8]) -> Result<(), BridgeError> {
    with_driver(|state| unsafe {
        let r = (state.fns.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len());
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Device {
                ordinal: 0,
                message: format!("cuMemcpyHtoD: {}", err_string(&state.fns, r)),
            });
        }
        Ok(())
    })
}

/// Device → host copy.
pub fn memcpy_dtoh(dst: &mut [u8], src: u64) -> Result<(), BridgeError> {
    with_driver(|state| unsafe {
        let r = (state.fns.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len());
        if r != CUDA_SUCCESS {
            return Err(BridgeError::Device {
                ordinal: 0,
                message: format!("cuMemcpyDtoH: {}", err_string(&state.fns, r)),
            });
        }
        Ok(())
    })
}
