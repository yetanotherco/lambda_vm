//! Raw-bit parity of the LDE entry points against the untouched single-column
//! `coset_lde_base` pipeline (zero-filled buffer, bit-reverse at `lde_size`,
//! then the NTT body), which runs the same butterflies in the same order.
//!
//! The batched, ext3 and row-major paths bit-reverse only the `n`-prefix and
//! load the zero-padded expansion inside the forward NTT's first 8 levels, and
//! skip zeroing the padding tail; the row-major path also bit-reverses a
//! device-resident input straight into the LDE buffer. None of that may change
//! a single output bit, so every comparison here is on raw (non-canonical)
//! u64s. The shapes straddle the fused-load thresholds: below the 8-level
//! kernel (lde < 256), exactly one block, one prefix slot per block (blowup
//! 256), past it (blowup 512), blowup 1 (no padding at all), and both sides
//! of the switch from in-place waves to the single copied-tail launch.

use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math_cuda::device::backend;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `(log_n, blowup)` pairs covering both sides of every threshold.
const SHAPES: &[(u32, usize)] = &[
    (4, 2),   // lde 32: per-level only, zero-filled path
    (6, 2),   // lde 128: still below the fused kernel
    (7, 2),   // lde 256: one block, spread path, single wave
    (8, 2),   // two blocks, two waves
    (5, 8),   // lde 256 with log_spread 3
    (1, 128), // n = 2, log_spread 7
    (1, 256), // one prefix slot per block (log_spread 8)
    (1, 512), // no slot in most blocks: zero-filled fallback
    (10, 1),  // blowup 1: nothing to pad
    (10, 2),
    (9, 4),
    (11, 8),
    (13, 2),
    (12, 16), // 256 blocks: the largest tail-only shape
    (16, 2),  // 512 blocks: one in-place wave, then the tail
    (15, 8),  // log_spread 3 with waves
    (17, 4),  // several waves
];

fn coset_weights(n: usize, g: u64) -> Vec<u64> {
    let inv_n = GoldilocksField::inv(&(n as u64)).unwrap();
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &g);
    }
    w
}

/// Random raw u64s, deliberately including non-canonical values.
fn random_u64s(rng: &mut ChaCha8Rng, len: usize) -> Vec<u64> {
    (0..len).map(|_| rng.r#gen::<u64>()).collect()
}

/// The reference: the old single-column pipeline.
fn reference_lde(col: &[u64], blowup: usize, weights: &[u64]) -> Vec<u64> {
    math_cuda::lde::coset_lde_base(col, blowup, weights).expect("reference coset_lde_base")
}

fn seed(log_n: u32, blowup: usize, cols: usize, salt: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(((log_n as u64) << 32) ^ ((blowup as u64) << 16) ^ cols as u64 ^ salt)
}

#[test]
fn batched_base_entry_points_match_reference_bits() {
    for &(log_n, blowup) in SHAPES {
        for m in [1usize, 3] {
            let n = 1usize << log_n;
            let lde = n * blowup;
            let mut rng = seed(log_n, blowup, m, 1);
            let cols: Vec<Vec<u64>> = (0..m).map(|_| random_u64s(&mut rng, n)).collect();
            let slices: Vec<&[u64]> = cols.iter().map(|c| c.as_slice()).collect();
            let weights = coset_weights(n, 7);
            let expected: Vec<Vec<u64>> = cols
                .iter()
                .map(|c| reference_lde(c, blowup, &weights))
                .collect();
            let ctx = format!("log_n={log_n} blowup={blowup} m={m}");

            let batch = math_cuda::lde::coset_lde_batch_base(&slices, blowup, &weights).unwrap();
            assert_eq!(batch, expected, "coset_lde_batch_base {ctx}");

            let mut owned = vec![vec![0u64; lde]; m];
            let mut outs: Vec<&mut [u64]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
            math_cuda::lde::coset_lde_batch_base_into(&slices, blowup, &weights, &mut outs)
                .unwrap();
            assert_eq!(owned, expected, "coset_lde_batch_base_into {ctx}");

            if lde >= 2 {
                let mut owned = vec![vec![0u64; lde]; m];
                let mut outs: Vec<&mut [u64]> =
                    owned.iter_mut().map(|v| v.as_mut_slice()).collect();
                let mut leaves = vec![0u8; (lde / 2) * 32];
                math_cuda::lde::coset_lde_batch_base_into_with_leaf_hash(
                    &slices,
                    blowup,
                    &weights,
                    &mut outs,
                    &mut leaves,
                )
                .unwrap();
                assert_eq!(
                    owned, expected,
                    "coset_lde_batch_base_into_with_leaf_hash {ctx}"
                );
            }
        }
    }
}

/// Interleave three base components into ext3 `[a0, b0, c0, a1, ...]`.
fn interleave3(parts: &[Vec<u64>; 3]) -> Vec<u64> {
    let len = parts[0].len();
    let mut out = Vec::with_capacity(3 * len);
    for i in 0..len {
        out.extend(parts.iter().map(|p| p[i]));
    }
    out
}

#[test]
fn batched_ext3_matches_reference_bits() {
    for &(log_n, blowup) in SHAPES {
        let n = 1usize << log_n;
        let lde = n * blowup;
        let m = 2;
        let mut rng = seed(log_n, blowup, m, 2);
        let parts: Vec<[Vec<u64>; 3]> = (0..m)
            .map(|_| std::array::from_fn(|_| random_u64s(&mut rng, n)))
            .collect();
        let cols: Vec<Vec<u64>> = parts.iter().map(interleave3).collect();
        let slices: Vec<&[u64]> = cols.iter().map(|c| c.as_slice()).collect();
        let weights = coset_weights(n, 7);
        let expected: Vec<Vec<u64>> = parts
            .iter()
            .map(|p| {
                interleave3(&std::array::from_fn(|k| {
                    reference_lde(&p[k], blowup, &weights)
                }))
            })
            .collect();

        let mut owned = vec![vec![0u64; 3 * lde]; m];
        let mut outs: Vec<&mut [u64]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::coset_lde_batch_ext3_into(&slices, n, blowup, &weights, &mut outs).unwrap();
        assert_eq!(
            owned, expected,
            "coset_lde_batch_ext3_into log_n={log_n} blowup={blowup}"
        );
    }
}

/// Slabs handed in with GARBAGE in every padding tail: the slab LDE must
/// never read it (spread load) or must zero it itself (small shapes).
#[test]
fn ext3_slabs_keep_ignores_the_padding_tail() {
    for &(log_n, blowup) in SHAPES {
        let n = 1usize << log_n;
        let lde = n * blowup;
        let m = 2;
        let mut rng = seed(log_n, blowup, m, 3);
        let parts: Vec<[Vec<u64>; 3]> = (0..m)
            .map(|_| std::array::from_fn(|_| random_u64s(&mut rng, n)))
            .collect();
        let weights = coset_weights(n, 7);

        let mut slabs = Vec::with_capacity(3 * m * lde);
        for p in &parts {
            for comp in p {
                slabs.extend_from_slice(comp);
                slabs.extend(random_u64s(&mut rng, lde - n).into_iter().map(|x| x | 1));
            }
        }
        let expected: Vec<Vec<u64>> = parts
            .iter()
            .map(|p| {
                interleave3(&std::array::from_fn(|k| {
                    reference_lde(&p[k], blowup, &weights)
                }))
            })
            .collect();

        let be = backend().expect("cuda backend");
        let stream = be.next_stream();
        let buf = stream.clone_htod(&slabs).unwrap();
        let mut owned = vec![vec![0u64; 3 * lde]; m];
        let mut outs: Vec<&mut [u64]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::coset_lde_batch_ext3_slabs_keep(
            &stream,
            buf,
            m,
            n,
            blowup,
            &weights,
            Some(&mut outs),
        )
        .unwrap();
        assert_eq!(
            owned, expected,
            "coset_lde_batch_ext3_slabs_keep log_n={log_n} blowup={blowup}"
        );
    }
}

/// Coefficients in, coset evaluations out: the reference scales on the host
/// (bit-identical Goldilocks mul) and runs the untouched `ntt::forward` on the
/// zero-padded coefficients.
#[test]
fn evaluate_poly_coset_ext3_matches_reference_bits() {
    for &(log_n, blowup) in SHAPES {
        let n = 1usize << log_n;
        let lde = n * blowup;
        let m = 2;
        let mut rng = seed(log_n, blowup, m, 4);
        let parts: Vec<[Vec<u64>; 3]> = (0..m)
            .map(|_| std::array::from_fn(|_| random_u64s(&mut rng, n)))
            .collect();
        let coefs: Vec<Vec<u64>> = parts.iter().map(interleave3).collect();
        let slices: Vec<&[u64]> = coefs.iter().map(|c| c.as_slice()).collect();
        let weights = random_u64s(&mut rng, n);
        let reference = |comp: &Vec<u64>| {
            let mut padded: Vec<u64> = comp
                .iter()
                .zip(&weights)
                .map(|(c, w)| GoldilocksField::mul(c, w))
                .collect();
            padded.resize(lde, 0);
            math_cuda::ntt::forward(&padded).expect("reference forward ntt")
        };
        let expected: Vec<Vec<u64>> = parts
            .iter()
            .map(|p| interleave3(&std::array::from_fn(|k| reference(&p[k]))))
            .collect();

        let mut owned = vec![vec![0u64; 3 * lde]; m];
        let mut outs: Vec<&mut [u64]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::evaluate_poly_coset_batch_ext3_into(
            &slices, n, blowup, &weights, &mut outs,
        )
        .unwrap();
        assert_eq!(
            owned, expected,
            "evaluate_poly_coset_batch_ext3_into log_n={log_n} blowup={blowup}"
        );
    }
}

/// Row-major reference: column `c` of the `n x m` row-major input through the
/// old pipeline, laid back out row-major.
fn reference_row_major(
    row_major: &[u64],
    n: usize,
    m: usize,
    blowup: usize,
    w: &[u64],
) -> Vec<u64> {
    let lde = n * blowup;
    let mut out = vec![0u64; lde * m];
    for c in 0..m {
        let col: Vec<u64> = (0..n).map(|r| row_major[r * m + c]).collect();
        for (r, v) in reference_lde(&col, blowup, w).into_iter().enumerate() {
            out[r * m + c] = v;
        }
    }
    out
}

fn download(
    buf: &math_cuda::CudaSlice<u64>,
    ready: impl FnOnce(&math_cuda::CudaStream),
) -> Vec<u64> {
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    ready(&stream);
    let host = stream.clone_dtoh(buf).unwrap();
    stream.synchronize().unwrap();
    host
}

/// Host input and device input (bit-reversed straight out of `predev`) must
/// give the reference bits, the same root, the same column-major handle and
/// the same trace-domain snapshot. Column counts exercise partial 8-column
/// tiles of the row-major kernels.
#[test]
fn row_major_host_and_device_input_match_reference_bits() {
    for &(log_n, blowup) in SHAPES {
        if n_times(log_n, blowup) < 2 {
            continue;
        }
        for m in [1usize, 3, 13] {
            let n = 1usize << log_n;
            let lde = n * blowup;
            let mut rng = seed(log_n, blowup, m, 5);
            let row_major = random_u64s(&mut rng, n * m);
            let weights = coset_weights(n, 7);
            let expected = reference_row_major(&row_major, n, m, blowup, &weights);
            let ctx = format!("log_n={log_n} blowup={blowup} m={m}");

            let (host_handle, host_lde) =
                math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
                    &row_major, None, n, m, blowup, &weights, true,
                )
                .unwrap();
            assert_eq!(host_lde, expected, "row-major host input {ctx}");

            let predev = upload_synced(&row_major);
            let (dev_handle, dev_lde) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
                &row_major,
                Some(&predev),
                n,
                m,
                blowup,
                &weights,
                true,
            )
            .unwrap();
            assert_eq!(dev_lde, expected, "row-major device input {ctx}");
            assert_eq!(
                dev_handle.tree.as_ref().unwrap().root,
                host_handle.tree.as_ref().unwrap().root,
                "row-major root, device vs host input {ctx}"
            );

            let col_major: Vec<u64> = (0..m)
                .flat_map(|c| (0..lde).map(move |r| (r, c)))
                .map(|(r, c)| expected[r * m + c])
                .collect();
            let trace_col_major: Vec<u64> = (0..m)
                .flat_map(|c| (0..n).map(move |r| (r, c)))
                .map(|(r, c)| row_major[r * m + c])
                .collect();
            for (what, handle) in [("host", &host_handle), ("device", &dev_handle)] {
                let buf = download(&handle.buf, |s| handle.wait_ready_on(s).unwrap());
                assert_eq!(buf, col_major, "{what}-input column-major handle {ctx}");
                let trace = download(handle.trace_dev.as_ref().unwrap(), |s| {
                    handle.wait_ready_on(s).unwrap()
                });
                assert_eq!(trace, trace_col_major, "{what}-input trace snapshot {ctx}");
            }
        }
    }
}

/// A device input as the prover hands one over: uploaded on its own stream
/// and host-synchronized before it escapes. The backend disables cudarc's
/// cross-stream event tracking (see `Backend::init`), so the LDE stream does
/// not wait on the upload by itself.
fn upload_synced(data: &[u64]) -> math_cuda::CudaSlice<u64> {
    let stream = backend().expect("cuda backend").next_stream();
    let dev = stream.clone_htod(data).unwrap();
    stream.synchronize().unwrap();
    dev
}

fn n_times(log_n: u32, blowup: usize) -> usize {
    (1usize << log_n) * blowup
}

#[test]
fn row_major_split_trees_device_input_matches_host_input() {
    for &(log_n, blowup) in SHAPES {
        if n_times(log_n, blowup) < 2 {
            continue;
        }
        let (n, m, split) = (1usize << log_n, 5usize, 2usize);
        let mut rng = seed(log_n, blowup, m, 6);
        let row_major = random_u64s(&mut rng, n * m);
        let weights = coset_weights(n, 7);
        let expected = reference_row_major(&row_major, n, m, blowup, &weights);
        let ctx = format!("log_n={log_n} blowup={blowup}");

        let (host_nodes, host_handle, host_lde) = math_cuda::lde::coset_lde_row_major_split_trees(
            &row_major, None, n, m, blowup, &weights, split, true, true,
        )
        .unwrap();
        assert_eq!(host_lde, expected, "split trees host input {ctx}");

        let predev = upload_synced(&row_major);
        let (dev_nodes, dev_handle, dev_lde) = math_cuda::lde::coset_lde_row_major_split_trees(
            &row_major,
            Some(&predev),
            n,
            m,
            blowup,
            &weights,
            split,
            true,
            true,
        )
        .unwrap();
        assert_eq!(dev_lde, expected, "split trees device input {ctx}");
        assert_eq!(dev_nodes, host_nodes, "precomputed tree {ctx}");
        assert_eq!(
            dev_handle.tree.as_ref().unwrap().root,
            host_handle.tree.as_ref().unwrap().root,
            "multiplicity root {ctx}"
        );
    }
}

#[test]
fn row_major_ext3_host_and_device_input_match_reference_bits() {
    for &(log_n, blowup) in SHAPES {
        if n_times(log_n, blowup) < 2 {
            continue;
        }
        let (n, m) = (1usize << log_n, 3usize);
        // An ext3 row-major table is a base row-major table with 3m lanes.
        let mut rng = seed(log_n, blowup, m, 7);
        let row_major = random_u64s(&mut rng, n * m * 3);
        let weights = coset_weights(n, 7);
        let expected = reference_row_major(&row_major, n, 3 * m, blowup, &weights);
        let ctx = format!("log_n={log_n} blowup={blowup}");

        let (host_handle, host_lde) =
            math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep(
                &row_major, n, m, blowup, &weights, true,
            )
            .unwrap();
        assert_eq!(host_lde, expected, "ext3 row-major host input {ctx}");

        let dev_in = upload_synced(&row_major);
        let (dev_handle, dev_lde) =
            math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep_dev(
                &dev_in, n, m, blowup, &weights, true,
            )
            .unwrap();
        assert_eq!(dev_lde, expected, "ext3 row-major device input {ctx}");
        assert_eq!(
            dev_handle.tree.as_ref().unwrap().root,
            host_handle.tree.as_ref().unwrap().root,
            "ext3 row-major root {ctx}"
        );
    }
}
