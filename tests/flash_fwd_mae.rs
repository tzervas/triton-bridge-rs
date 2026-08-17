// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Job C numerical gate: Unsloth FA PTX vs Candle-compatible softmax.
//!
//! * Missing payload / no driver → `FAIL_ENV` (honest; not a green MAE).
//! * Launch uses [`KernelArg::DevicePtr`] only — no host Q/K/V in the kernel.
//! * This crate does **not** depend on unsloth-rs.

use std::fs;
use std::path::PathBuf;

use triton_bridge::{
    device_alloc, device_free, launch, load_ptx, memcpy_dtoh, memcpy_htod, KernelArg, LaunchSpec,
};

const MAE_BUDGET: f64 = 1e-5;
const HEAD_DIM: usize = 64;
const SEQS: [usize; 2] = [128, 512];

fn payload_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("precompiled/sm90_flash_fwd.ptx")
}

/// Candle-compatible SDPA: `softmax(Q K^T * scale) V` in f32 (multiply scale).
fn candle_softmax_attn(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * dim];
    let mut scores = vec![0.0f32; seq];
    for query_i in 0..seq {
        let qrow = &query[query_i * dim..query_i * dim + dim];
        let mut max_s = f32::NEG_INFINITY;
        for (key_j, score) in scores.iter_mut().enumerate() {
            let krow = &key[key_j * dim..key_j * dim + dim];
            let mut dot = 0.0f32;
            for pos in 0..dim {
                dot += qrow[pos] * krow[pos];
            }
            let scaled = dot * scale;
            *score = scaled;
            if scaled > max_s {
                max_s = scaled;
            }
        }
        let mut sum = 0.0f32;
        for score in &mut scores {
            let exp = (*score - max_s).exp();
            *score = exp;
            sum += exp;
        }
        let inv = if sum > 0.0 { sum.recip() } else { 0.0 };
        let dest = &mut out[query_i * dim..query_i * dim + dim];
        for (key_j, score) in scores.iter().enumerate() {
            let weight = *score * inv;
            let vrow = &value[key_j * dim..key_j * dim + dim];
            for pos in 0..dim {
                dest[pos] += weight * vrow[pos];
            }
        }
    }
    out
}

fn mae(got: &[f32], expect: &[f32]) -> f64 {
    assert_eq!(got.len(), expect.len());
    let mut acc = 0.0f64;
    for (lhs, rhs) in got.iter().zip(expect.iter()) {
        acc += f64::from(*lhs - *rhs).abs();
    }
    let n_elem = u32::try_from(got.len()).expect("mae len fits u32");
    acc / f64::from(n_elem)
}

fn le_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect()
}

fn from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic host Q/K/V. Copied to the device; kernel never sees host ptrs.
fn host_qkv(seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_elem = seq * dim;
    let mut query = vec![0.0f32; n_elem];
    let mut key = vec![0.0f32; n_elem];
    let mut value = vec![0.0f32; n_elem];
    for (idx, (q_slot, k_slot, v_slot)) in query
        .iter_mut()
        .zip(key.iter_mut())
        .zip(value.iter_mut())
        .map(|((q_slot, k_slot), v_slot)| (q_slot, k_slot, v_slot))
        .enumerate()
    {
        let idx_u = u16::try_from(idx).expect("qkv index fits u16");
        let phase = f32::from(idx_u) * 0.017 - 0.31;
        *q_slot = phase.sin() * 0.25;
        *k_slot = phase.cos() * 0.25;
        let rem = u16::try_from(idx % 17).expect("mod 17 fits u16");
        *v_slot = f32::from(rem) * 0.05 - 0.4;
    }
    (query, key, value)
}

#[test]
fn flash_fwd_vs_candle_softmax_or_fail_env() {
    let path = payload_path();
    if !path.is_file() {
        eprintln!(
            "FAIL_ENV: precompiled/sm90_flash_fwd.ptx missing — Unsloth FA was not \
             compiled (no invented PTX). device_pointers=false"
        );
        return;
    }

    let ptx = fs::read_to_string(&path).expect("read FA PTX");
    match load_ptx("sm90_flash_fwd", &ptx, Some(90)) {
        Ok(module) => {
            assert!(
                module.is_device_resident(),
                "loaded FA module must be device-resident"
            );
            // Without a compile-time ABI sidecar we do not guess Unsloth's
            // kernel argument order. Launch only when the meta names an entry
            // *and* we can pass Q/K/V/Out as device pointers.
            let meta_path = path.with_extension("ptx.meta.json");
            let meta = fs::read_to_string(&meta_path).unwrap_or_default();
            let entry = meta
                .lines()
                .find_map(|line| {
                    let trimmed = line.trim().trim_end_matches(',');
                    trimmed
                        .strip_prefix("\"entry\":")
                        .map(|v| v.trim().trim_matches('"').to_string())
                })
                .filter(|name| !name.is_empty() && name != "null");
            let Some(kernel) = entry else {
                eprintln!(
                    "FAIL_ENV: precompiled/sm90_flash_fwd.ptx loaded but no kernel \
                     entry in sidecar metadata; refusing to guess ABI. \
                     device_pointers=false"
                );
                return;
            };

            for seq in SEQS {
                let dim = HEAD_DIM;
                let dim_u = u32::try_from(dim).expect("head dim fits u32");
                let seq_u = u32::try_from(seq).expect("seq fits u32");
                let dim_f = f32::from(u16::try_from(dim).expect("head dim fits u16"));
                let scale = dim_f.sqrt().recip();
                let (query, key, value) = host_qkv(seq, dim);
                let expect = candle_softmax_attn(&query, &key, &value, seq, dim, scale);
                let bytes = seq * dim * 4;

                let dq = device_alloc(bytes).expect("alloc Q");
                let dk = device_alloc(bytes).expect("alloc K");
                let dv = device_alloc(bytes).expect("alloc V");
                let dout = device_alloc(bytes).expect("alloc Out");
                memcpy_htod(dq, &le_bytes(&query)).expect("htod Q");
                memcpy_htod(dk, &le_bytes(&key)).expect("htod K");
                memcpy_htod(dv, &le_bytes(&value)).expect("htod V");

                // Device pointers only — no to_vec1 of Q/K/V into the kernel.
                let args = [
                    KernelArg::DevicePtr(dq),
                    KernelArg::DevicePtr(dk),
                    KernelArg::DevicePtr(dv),
                    KernelArg::DevicePtr(dout),
                    KernelArg::f32(scale),
                    KernelArg::U32(seq_u),
                    KernelArg::U32(dim_u),
                ];
                let mut spec = LaunchSpec::new(&kernel).with_args(&args);
                spec.grid = [u32::try_from(seq.div_ceil(64)).unwrap_or(1), 1, 1];
                spec.block = [64, 1, 1];

                if let Err(e) = launch(&module, &spec) {
                    device_free(dq).ok();
                    device_free(dk).ok();
                    device_free(dv).ok();
                    device_free(dout).ok();
                    panic!("FA launch failed (not FAIL_ENV): {e}");
                }

                let mut raw = vec![0u8; bytes];
                memcpy_dtoh(&mut raw, dout).expect("dtoh Out");
                device_free(dq).ok();
                device_free(dk).ok();
                device_free(dv).ok();
                device_free(dout).ok();

                let got = from_le_bytes(&raw);
                let err = mae(&got, &expect);
                assert!(
                    err < MAE_BUDGET,
                    "seq={seq} MAE {err} >= {MAE_BUDGET} (Candle softmax, f32)"
                );
                eprintln!("seq={seq} MAE={err} (budget {MAE_BUDGET})");
            }
        }
        Err(e) => {
            assert!(
                e.is_not_ready() || e.is_fail_env(),
                "unexpected error (not FAIL_ENV/NotReady): {e}"
            );
            eprintln!("FAIL_ENV/NotReady loading FA PTX: {e}");
        }
    }
}
