//! Parity tests for the trivial trace-builder primitives. Each test runs
//! the GPU kernel and compares element-wise against a CPU reference.
//!
//! Cuda-gated (the math-cuda crate is GPU-only). Skipped without an actual
//! GPU runner present.

use math_cuda::device::backend;
use math_cuda::trace_primitives::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn rand_u64s(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen::<u64>()).collect()
}

// ---------------------------------------------------------------------------
// 1. pad_to_pow2_u64
// ---------------------------------------------------------------------------
fn pad_to_pow2_cpu(src: &[u64], dst_len: usize, sentinel: u64) -> Vec<u64> {
    let mut out = vec![sentinel; dst_len];
    let copy_len = src.len().min(dst_len);
    out[..copy_len].copy_from_slice(&src[..copy_len]);
    out
}

#[test]
fn pad_to_pow2_parity() {
    let be = backend().unwrap();
    let stream = be.next_stream();
    for (src_len, dst_len, sentinel) in [(0usize, 16usize, 0u64), (5, 8, 0), (100, 128, 42), (1023, 1024, 7), (1024, 2048, 999)] {
        let src = rand_u64s(src_len, 100 + src_len as u64);
        let cpu = pad_to_pow2_cpu(&src, dst_len, sentinel);
        let src_dev = stream.clone_htod(&src).unwrap();
        let gpu_dev = pad_to_pow2_u64_dev(&src_dev, src_len, sentinel, dst_len, &stream).unwrap();
        let gpu = stream.clone_dtoh(&gpu_dev).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(cpu, gpu, "pad_to_pow2 src_len={src_len} dst_len={dst_len}");
    }
}

// ---------------------------------------------------------------------------
// 2. decompose_u64_to_bytes
// ---------------------------------------------------------------------------
fn decompose_to_bytes_cpu(src: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(src.len() * 8);
    for v in src {
        for k in 0..8 {
            out.push((v >> (8 * k)) & 0xff);
        }
    }
    out
}

#[test]
fn decompose_to_bytes_parity() {
    let be = backend().unwrap();
    let stream = be.next_stream();
    for n in [0usize, 1, 17, 256, 1024] {
        let src = rand_u64s(n, 200 + n as u64);
        let cpu = decompose_to_bytes_cpu(&src);
        let src_dev = stream.clone_htod(&src).unwrap();
        let gpu_dev = decompose_u64_to_bytes_dev(&src_dev, n, &stream).unwrap();
        let gpu = stream.clone_dtoh(&gpu_dev).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(cpu, gpu, "decompose_to_bytes n={n}");
    }
}

// ---------------------------------------------------------------------------
// 3. decompose_u64_to_halfwords
// ---------------------------------------------------------------------------
fn decompose_to_halfwords_cpu(src: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for v in src {
        for k in 0..4 {
            out.push((v >> (16 * k)) & 0xffff);
        }
    }
    out
}

#[test]
fn decompose_to_halfwords_parity() {
    let be = backend().unwrap();
    let stream = be.next_stream();
    for n in [0usize, 1, 17, 256, 1024] {
        let src = rand_u64s(n, 300 + n as u64);
        let cpu = decompose_to_halfwords_cpu(&src);
        let src_dev = stream.clone_htod(&src).unwrap();
        let gpu_dev = decompose_u64_to_halfwords_dev(&src_dev, n, &stream).unwrap();
        let gpu = stream.clone_dtoh(&gpu_dev).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(cpu, gpu, "decompose_to_halfwords n={n}");
    }
}

// ---------------------------------------------------------------------------
// 4. fill_sequential_u64
// ---------------------------------------------------------------------------
#[test]
fn fill_sequential_parity() {
    let be = backend().unwrap();
    let stream = be.next_stream();
    for (start, stride, n) in [(0u64, 1u64, 0usize), (0, 1, 17), (10, 4, 1000), (1_000_000, 7, 4096)] {
        let cpu: Vec<u64> = (0..n as u64).map(|i| start + i * stride).collect();
        let gpu_dev = fill_sequential_u64_dev(start, stride, n, &stream).unwrap();
        let gpu = stream.clone_dtoh(&gpu_dev).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(cpu, gpu, "fill_sequential start={start} stride={stride} n={n}");
    }
}

// ---------------------------------------------------------------------------
// 5. range_check_column_u64
// ---------------------------------------------------------------------------
#[test]
fn range_check_column_parity() {
    let be = backend().unwrap();
    let stream = be.next_stream();
    for n in [0usize, 1, 256, 65536] {
        let cpu: Vec<u64> = (0..n as u64).collect();
        let gpu_dev = range_check_column_u64_dev(n, &stream).unwrap();
        let gpu = stream.clone_dtoh(&gpu_dev).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(cpu, gpu, "range_check n={n}");
    }
}

// ---------------------------------------------------------------------------
// 6. extract_bits_u64
// ---------------------------------------------------------------------------
#[test]
fn extract_bits_parity() {
    let be = backend().unwrap();
    let stream = be.next_stream();
    for (shift, width) in [(0u32, 1u32), (3, 5), (16, 16), (32, 32), (0, 64), (4, 60)] {
        let n = 1024usize;
        let src = rand_u64s(n, 400 + (shift as u64) * 64 + (width as u64));
        let mask = if width >= 64 { !0u64 } else { (1u64 << width) - 1 };
        let cpu: Vec<u64> = src.iter().map(|v| (v >> shift) & mask).collect();
        let src_dev = stream.clone_htod(&src).unwrap();
        let gpu_dev = extract_bits_u64_dev(&src_dev, n, shift, width, &stream).unwrap();
        let gpu = stream.clone_dtoh(&gpu_dev).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(cpu, gpu, "extract_bits shift={shift} width={width}");
    }
}

// ---------------------------------------------------------------------------
// 7. multiplicity_count_by_index
// ---------------------------------------------------------------------------
#[test]
fn multiplicity_count_by_index_parity() {
    use std::collections::HashMap;
    let be = backend().unwrap();
    let stream = be.next_stream();
    for (n, max_key) in [(0usize, 16usize), (100, 16), (10_000, 256), (100_000, 1024)] {
        let mut rng = ChaCha8Rng::seed_from_u64(500 + n as u64 + max_key as u64);
        let keys: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() % max_key as u64).collect();
        let counts_len = max_key;

        // CPU reference.
        let mut cpu = vec![0u64; counts_len];
        let mut hm: HashMap<u64, u64> = HashMap::new();
        for &k in &keys {
            *hm.entry(k).or_insert(0) += 1;
        }
        for (k, c) in hm {
            cpu[k as usize] = c;
        }

        // GPU.
        let keys_dev = if n == 0 {
            stream.alloc_zeros::<u64>(0).unwrap()
        } else {
            stream.clone_htod(&keys).unwrap()
        };
        let mut counts_dev = stream.alloc_zeros::<u64>(counts_len).unwrap();
        multiplicity_count_by_index_dev(&keys_dev, n, &mut counts_dev, &stream).unwrap();
        let gpu = stream.clone_dtoh(&counts_dev).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(cpu, gpu, "multiplicity n={n} max_key={max_key}");
    }
}
