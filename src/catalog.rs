// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Python Unsloth Triton kernel names → who owns the Rust port.
//!
//! This is a catalog, not a compiler. Math we own lives in unsloth-rs `CustomOp`.
//! This crate only loads *foreign* PTX/CUBIN (Flash Attention first).

/// Where a Python Unsloth kernel should land.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelHome {
    /// Candle `CustomOp` in unsloth-rs — do **not** FFI Triton for these.
    UnslothCustomOp,
    /// First real payload for this crate (tiled FA).
    TritonBridgePhase1,
    /// peft-rs consumes unsloth; do not fork kernels there.
    PeftConsume,
    /// axolotl-rs data plane (packing), not a kernel crate.
    AxolotlData,
    /// Explicitly out of scope.
    OutOfScope,
}

/// One mapped kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelEntry {
    /// Python file under `unsloth/kernels/`.
    pub python: &'static str,
    /// Rust home.
    pub home: KernelHome,
    /// One-line role.
    pub role: &'static str,
}

/// Stable catalog. Keep in lockstep with `docs/PYTHON_UNSLOTH_KERNEL_MAP.md`.
pub const KERNEL_CATALOG: &[KernelEntry] = &[
    KernelEntry {
        python: "rms_layernorm.py",
        home: KernelHome::UnslothCustomOp,
        role: "RMSNorm fwd/bwd",
    },
    KernelEntry {
        python: "rope_embedding.py",
        home: KernelHome::UnslothCustomOp,
        role: "RoPE apply",
    },
    KernelEntry {
        python: "swiglu.py",
        home: KernelHome::UnslothCustomOp,
        role: "silu(gate)*up",
    },
    KernelEntry {
        python: "cross_entropy_loss.py",
        home: KernelHome::UnslothCustomOp,
        role: "chunked CE",
    },
    KernelEntry {
        python: "flash_attention_2.py",
        home: KernelHome::TritonBridgePhase1,
        role: "tiled attention (SRAM)",
    },
    KernelEntry {
        python: "fast_lora.py",
        home: KernelHome::PeftConsume,
        role: "fused LoRA add",
    },
    KernelEntry {
        python: "flex_attention.py",
        home: KernelHome::AxolotlData,
        role: "packing / ragged",
    },
    KernelEntry {
        python: "moe.py",
        home: KernelHome::OutOfScope,
        role: "routing",
    },
];

/// Look up a Python kernel filename (suffix match).
#[must_use]
pub fn lookup(python_name: &str) -> Option<&'static KernelEntry> {
    KERNEL_CATALOG
        .iter()
        .find(|e| e.python == python_name || python_name.ends_with(e.python))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_is_ours() {
        let e = lookup("flash_attention_2.py").unwrap();
        assert_eq!(e.home, KernelHome::TritonBridgePhase1);
    }

    #[test]
    fn rms_stays_in_unsloth() {
        assert_eq!(
            lookup("rms_layernorm.py").unwrap().home,
            KernelHome::UnslothCustomOp
        );
    }
}
