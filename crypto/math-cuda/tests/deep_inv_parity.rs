//! The row-wise DEEP/OOD denominator paths against the legacy paths they
//! replace by default (`math_cuda::deep_inv`).
//!
//! Each computes the same field elements as its legacy path, so each test runs
//! both on the same inputs and compares canonically — the device representation
//! is non-canonical, and a different operation order may land a value on `v` or
//! `v + p`, which nothing downstream can observe:
//!
//! - `invert_denoms_rowwise_dev` against the six-kernel scan
//!   (`compute_and_invert_denoms_ext3_scan_dev`) and the definition,
//!   `d · d⁻¹ = 1`, on a sample of rows;
//! - `deep_composition_ext3_fused_keep` against the buffered
//!   `deep_composition_ext3_fully_resident_keep` fed the scan's inverses, and
//!   against a host port of the DEEP sum;
//! - `barycentric_*_chunked_with_dev_inv_denoms` against the one-block-per-
//!   column `barycentric_*_one_block_with_dev_inv_denoms`, at a nonzero offset
//!   into a multi-point buffer.
//!
//! Every test names both of its paths, so none reads the process setting
//! (`LAMBDA_VM_DEEP_INV_LEGACY`); `tests/deep_inv_setting.rs` does.
//!
//! Needs a GPU, like every test in this directory.

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math::field::traits::IsPrimeField;
use math_cuda::device::backend;
use math_cuda::inverse::{
    DenomSign, compute_and_invert_denoms_ext3_scan_dev, invert_denoms_rowwise_dev,
};
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext3>;

fn rand_fp3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
    ])
}

fn raw3(e: &Fp3) -> [u64; 3] {
    [
        *e.value()[0].value(),
        *e.value()[1].value(),
        *e.value()[2].value(),
    ]
}

fn flat3(xs: &[Fp3]) -> Vec<u64> {
    xs.iter().flat_map(raw3).collect()
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(Gl::canonical).collect()
}

fn from_raw3(x: &[u64]) -> Fp3 {
    Fp3::new([Fp::from_raw(x[0]), Fp::from_raw(x[1]), Fp::from_raw(x[2])])
}

// ---------------------------------------------------------------------------
// Row-wise denominator inversion
// ---------------------------------------------------------------------------

/// Row-wise == scan == host, canonically, for `k` points over `n` domain points
/// whose limbs include non-canonical values.
fn rowwise_matches_scan(n: usize, k: usize, sign: DenomSign, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let z: Vec<Fp3> = (0..k).map(|_| rand_fp3(&mut rng)).collect();
    let z_raw = flat3(&z);

    let be = backend().unwrap();
    let stream = be.next_stream();
    let x_dev = stream.clone_htod(&x).unwrap();
    let scan =
        compute_and_invert_denoms_ext3_scan_dev(&x_dev, &z_raw, n, k, sign, &stream).unwrap();
    let rowwise = invert_denoms_rowwise_dev(&x_dev, &z_raw, n, k, sign, &stream).unwrap();
    let scan: Vec<u64> = stream.clone_dtoh(&scan).unwrap();
    let rowwise: Vec<u64> = stream.clone_dtoh(&rowwise).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(rowwise.len(), 3 * n * k);
    assert_eq!(
        canon(&rowwise),
        canon(&scan),
        "row-wise != scan at n={n} k={k} seed={seed}"
    );

    // And the definition, on a sample of rows: d · d^{-1} = 1.
    for i in (0..n).step_by((n / 97).max(1)) {
        for (kk, zk) in z.iter().enumerate() {
            let xi = Fp::from_raw(x[i]).to_extension::<Ext3>();
            let d = match sign {
                DenomSign::ZMinusX => zk - &xi,
                DenomSign::XMinusZ => &xi - zk,
            };
            let at = (kk * n + i) * 3;
            assert_eq!(
                d * from_raw3(&rowwise[at..at + 3]),
                Fp3::one(),
                "not an inverse at row {i}, point {kk}"
            );
        }
    }
}

#[test]
fn rowwise_inversion_matches_the_scan() {
    // Every instantiation, both signs; the R3 (trace, k points), R3 parts
    // (k = 1) and R4 (k = 1 + points) shapes; a size that is not a multiple of
    // the block.
    for k in 1..=math_cuda::inverse::ROWWISE_MAX_K {
        rowwise_matches_scan(1000 + k, k, DenomSign::ZMinusX, 10 + k as u64);
        rowwise_matches_scan(777, k, DenomSign::XMinusZ, 20 + k as u64);
    }
    rowwise_matches_scan(1 << 16, 2, DenomSign::ZMinusX, 31);
    rowwise_matches_scan(1 << 16, 1, DenomSign::ZMinusX, 32);
    rowwise_matches_scan(1 << 18, 3, DenomSign::XMinusZ, 33);
}

/// A zero denominator (a domain point equal to an opening point) zeroes the
/// row's inverses in release — the scan's release behaviour for its batch —
/// and leaves every other row exact.
#[cfg(not(any(debug_assertions, feature = "test-faults")))]
#[test]
fn rowwise_zero_denominator_zeroes_only_its_row() {
    let n = 300;
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut x: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    // Point 1 lies in the base field, and row 17 sits on it.
    let z = [
        rand_fp3(&mut rng),
        Fp3::new([Fp::from(42u64), Fp::zero(), Fp::zero()]),
    ];
    x[17] = 42;
    let be = backend().unwrap();
    let stream = be.next_stream();
    let x_dev = stream.clone_htod(&x).unwrap();
    let out =
        invert_denoms_rowwise_dev(&x_dev, &flat3(&z), n, 2, DenomSign::XMinusZ, &stream).unwrap();
    let out: Vec<u64> = stream.clone_dtoh(&out).unwrap();
    stream.synchronize().unwrap();
    for k in 0..2 {
        let at = (k * n + 17) * 3;
        assert_eq!(canon(&out[at..at + 3]), vec![0, 0, 0], "row 17 point {k}");
    }
    for i in [0usize, 16, 18, n - 1] {
        let at = i * 3;
        let d = Fp::from_raw(x[i]).to_extension::<Ext3>() - z[0];
        assert_eq!(d * from_raw3(&out[at..at + 3]), Fp3::one(), "row {i}");
    }
}

/// The same input trips the debug guard, as the scan's zero-total guard does.
#[cfg(any(debug_assertions, feature = "test-faults"))]
#[test]
#[should_panic(expected = "a zero denominator has no inverse")]
fn rowwise_zero_denominator_trips_the_guard() {
    let n = 300;
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut x: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let z = [
        rand_fp3(&mut rng),
        Fp3::new([Fp::from(42u64), Fp::zero(), Fp::zero()]),
    ];
    x[17] = 42;
    let be = backend().unwrap();
    let stream = be.next_stream();
    let x_dev = stream.clone_htod(&x).unwrap();
    let _ = invert_denoms_rowwise_dev(&x_dev, &flat3(&z), n, 2, DenomSign::XMinusZ, &stream);
}

// ---------------------------------------------------------------------------
// Fused DEEP
// ---------------------------------------------------------------------------

struct DeepCase {
    main: Vec<Vec<u64>>,
    aux: Vec<Vec<Fp3>>,
    parts: Vec<Vec<Fp3>>,
    coset: Vec<u64>,
    z: Vec<Fp3>,
    h_ood: Vec<Fp3>,
    trace_ood: Vec<Vec<Fp3>>,
    gammas_h: Vec<Fp3>,
    gammas_tr: Vec<Vec<Fp3>>,
}

fn deep_case(
    lde: usize,
    num_main: usize,
    num_aux: usize,
    parts: usize,
    k: usize,
    seed: u64,
) -> DeepCase {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let total = num_main + num_aux;
    DeepCase {
        main: (0..num_main)
            .map(|_| (0..lde).map(|_| rng.r#gen::<u64>()).collect())
            .collect(),
        aux: (0..num_aux)
            .map(|_| (0..lde).map(|_| rand_fp3(&mut rng)).collect())
            .collect(),
        parts: (0..parts)
            .map(|_| (0..lde).map(|_| rand_fp3(&mut rng)).collect())
            .collect(),
        coset: (0..lde).map(|_| rng.r#gen::<u64>()).collect(),
        z: (0..1 + k).map(|_| rand_fp3(&mut rng)).collect(),
        h_ood: (0..parts).map(|_| rand_fp3(&mut rng)).collect(),
        trace_ood: (0..total)
            .map(|_| (0..k).map(|_| rand_fp3(&mut rng)).collect())
            .collect(),
        gammas_h: (0..parts).map(|_| rand_fp3(&mut rng)).collect(),
        gammas_tr: (0..total)
            .map(|_| (0..k).map(|_| rand_fp3(&mut rng)).collect())
            .collect(),
    }
}

fn ext3_slabs(cols: &[Vec<Fp3>], lde: usize) -> Vec<u64> {
    let mut out = vec![0u64; cols.len().max(1) * 3 * lde];
    for (c, col) in cols.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            let [a, b, cc] = raw3(v);
            out[(c * 3) * lde + r] = a;
            out[(c * 3 + 1) * lde + r] = b;
            out[(c * 3 + 2) * lde + r] = cc;
        }
    }
    out
}

/// Host DEEP at row `i` (natural order), the formula both kernels implement.
fn host_deep_row(case: &DeepCase, i: usize) -> Fp3 {
    let x = Fp::from_raw(case.coset[i]).to_extension::<Ext3>();
    let inv: Vec<Fp3> = case.z.iter().map(|z| (&x - z).inv().unwrap()).collect();
    let mut acc = Fp3::zero();
    for (j, part) in case.parts.iter().enumerate() {
        acc += case.gammas_h[j] * (part[i] - case.h_ood[j]) * inv[0];
    }
    let k = case.z.len() - 1;
    for (j, col) in case.main.iter().enumerate() {
        let t = Fp::from_raw(col[i]).to_extension::<Ext3>();
        for kk in 0..k {
            acc += case.gammas_tr[j][kk] * (&t - &case.trace_ood[j][kk]) * inv[1 + kk];
        }
    }
    for (j, col) in case.aux.iter().enumerate() {
        let jj = case.main.len() + j;
        for kk in 0..k {
            acc += case.gammas_tr[jj][kk] * (col[i] - case.trace_ood[jj][kk]) * inv[1 + kk];
        }
    }
    acc
}

fn fused_deep_matches_buffered(
    lde: usize,
    num_main: usize,
    num_aux: usize,
    parts: usize,
    k: usize,
    seed: u64,
) {
    let case = deep_case(lde, num_main, num_aux, parts, k, seed);
    let be = backend().unwrap();
    let stream = be.next_stream();

    let mut main_flat = vec![0u64; num_main.max(1) * lde];
    for (c, col) in case.main.iter().enumerate() {
        main_flat[c * lde..(c + 1) * lde].copy_from_slice(col);
    }
    let main = GpuLdeBase {
        ready: None,
        buf: Arc::new(stream.clone_htod(&main_flat).unwrap()),
        m: num_main,
        lde_size: lde,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };
    let aux = (num_aux > 0).then(|| GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.clone_htod(&ext3_slabs(&case.aux, lde)).unwrap()),
        m: num_aux,
        lde_size: lde,
        tree: None,
    });
    let parts_dev = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.clone_htod(&ext3_slabs(&case.parts, lde)).unwrap()),
        m: parts,
        lde_size: lde,
        tree: None,
    };
    let coset_dev = stream.clone_htod(&case.coset).unwrap();
    let z_raw = flat3(&case.z);
    let h_ood = flat3(&case.h_ood);
    let trace_ood: Vec<u64> = case.trace_ood.iter().flat_map(|c| flat3(c)).collect();
    let gammas_h = flat3(&case.gammas_h);
    let gammas_tr: Vec<u64> = case.gammas_tr.iter().flat_map(|c| flat3(c)).collect();
    stream.synchronize().unwrap();

    let fused = math_cuda::deep::deep_composition_ext3_fused_keep(
        &stream,
        &main,
        aux.as_ref(),
        &parts_dev,
        &coset_dev,
        &z_raw,
        &h_ood,
        &trace_ood,
        &gammas_h,
        &gammas_tr,
        parts,
        num_main,
        num_aux,
        k,
        1,
        lde,
    )
    .unwrap()
    .download()
    .unwrap();

    // The buffered path: the scan's inverses, then the kernel that reads them.
    // (It uploads the empty trace tables as they are, which the driver refuses
    // at k = 0 — no table proves with zero points; the host sum below covers it.)
    if k > 0 {
        let inv = compute_and_invert_denoms_ext3_scan_dev(
            &coset_dev,
            &z_raw,
            lde,
            1 + k,
            DenomSign::XMinusZ,
            &stream,
        )
        .unwrap();
        let buffered = math_cuda::deep::deep_composition_ext3_fully_resident_keep(
            &stream,
            &main,
            aux.as_ref(),
            &parts_dev,
            &inv,
            &h_ood,
            &trace_ood,
            &gammas_h,
            &gammas_tr,
            parts,
            num_main,
            num_aux,
            k,
            1,
            lde,
        )
        .unwrap()
        .download()
        .unwrap();
        assert_eq!(
            canon(&fused),
            canon(&buffered),
            "fused != buffered DEEP at lde={lde} main={num_main} aux={num_aux} parts={parts} k={k}"
        );
    }

    // The host sum on a sample of rows (the codeword is in FRI order:
    // position p holds natural row bitrev(p)).
    let log = lde.trailing_zeros();
    for p in (0..lde).step_by((lde / 61).max(1)) {
        let i = p.reverse_bits() >> (usize::BITS - log);
        assert_eq!(
            from_raw3(&fused[p * 3..p * 3 + 3]),
            host_deep_row(&case, i),
            "fused DEEP != host at row {i}"
        );
    }
}

#[test]
fn fused_deep_matches_the_buffered_kernel() {
    // Every instantiation (0..=3 eval points), with and without aux, one and
    // two composition parts.
    fused_deep_matches_buffered(1 << 10, 5, 3, 2, 2, 1);
    fused_deep_matches_buffered(1 << 10, 4, 0, 1, 0, 2);
    fused_deep_matches_buffered(1 << 11, 3, 2, 2, 1, 3);
    fused_deep_matches_buffered(1 << 9, 6, 4, 2, 3, 4);
    fused_deep_matches_buffered(1 << 16, 12, 5, 2, 2, 5);
}

/// More points than the fused kernels take is a clean decline, not a launch.
#[test]
fn fused_deep_declines_past_its_cap() {
    let lde = 1 << 8;
    let k = math_cuda::deep::FUSED_MAX_EVAL_POINTS + 1;
    let be = backend().unwrap();
    let stream = be.next_stream();
    let main = GpuLdeBase {
        ready: None,
        buf: Arc::new(stream.alloc_zeros::<u64>(lde).unwrap()),
        m: 1,
        lde_size: lde,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };
    let parts_dev = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.alloc_zeros::<u64>(3 * lde).unwrap()),
        m: 1,
        lde_size: lde,
        tree: None,
    };
    let coset = stream.alloc_zeros::<u64>(lde).unwrap();
    let r = math_cuda::deep::deep_composition_ext3_fused_keep(
        &stream,
        &main,
        None,
        &parts_dev,
        &coset,
        &vec![0; 3 * (1 + k)],
        &[0; 3],
        &vec![0; 3 * k],
        &[0; 3],
        &vec![0; 3 * k],
        1,
        1,
        0,
        k,
        1,
        lde,
    );
    assert!(r.is_err(), "{k} points must decline");
}

// ---------------------------------------------------------------------------
// Chunked single-point OOD sums
// ---------------------------------------------------------------------------

fn chunked_bary_matches(log_n: u32, blowup: usize, cols: usize, seed: u64) {
    use math_cuda::barycentric::{
        barycentric_base_chunked_with_dev_inv_denoms,
        barycentric_base_one_block_with_dev_inv_denoms,
        barycentric_ext3_chunked_with_dev_inv_denoms,
        barycentric_ext3_one_block_with_dev_inv_denoms,
    };
    let n = 1usize << log_n;
    let lde = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let base: Vec<u64> = (0..cols * lde).map(|_| rng.r#gen::<u64>()).collect();
    let ext: Vec<u64> = (0..cols * 3 * lde).map(|_| rng.r#gen::<u64>()).collect();
    let points: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    // Three points' inverses back to back; the sums read the middle one.
    let inv: Vec<u64> = (0..3 * 3 * n).map(|_| rng.r#gen::<u64>()).collect();
    let offset = 3 * n;

    let be = backend().unwrap();
    let stream = be.next_stream();
    let main = GpuLdeBase {
        ready: None,
        buf: Arc::new(stream.clone_htod(&base).unwrap()),
        m: cols,
        lde_size: lde,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };
    let aux = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.clone_htod(&ext).unwrap()),
        m: cols,
        lde_size: lde,
        tree: None,
    };
    let points_dev = stream.clone_htod(&points).unwrap();
    let inv_dev = stream.clone_htod(&inv).unwrap();
    stream.synchronize().unwrap();

    let one_block = barycentric_base_one_block_with_dev_inv_denoms(
        &stream,
        &main,
        blowup,
        &points_dev,
        &inv_dev,
        offset,
        n,
    )
    .unwrap();
    let chunked = barycentric_base_chunked_with_dev_inv_denoms(
        &stream,
        &main,
        blowup,
        &points_dev,
        &inv_dev,
        offset,
        n,
    )
    .unwrap();
    assert_eq!(
        canon(&chunked),
        canon(&one_block),
        "base sums, 2^{log_n} x {cols}"
    );

    let one_block = barycentric_ext3_one_block_with_dev_inv_denoms(
        &stream,
        &aux,
        blowup,
        &points_dev,
        &inv_dev,
        offset,
        n,
    )
    .unwrap();
    let chunked = barycentric_ext3_chunked_with_dev_inv_denoms(
        &stream,
        &aux,
        blowup,
        &points_dev,
        &inv_dev,
        offset,
        n,
    )
    .unwrap();
    assert_eq!(
        canon(&chunked),
        canon(&one_block),
        "ext3 sums, 2^{log_n} x {cols}"
    );
}

#[test]
fn chunked_single_point_sums_match_one_block_per_column() {
    // The parts OOD's 1–2 columns at a size that chunks 64-fold and past the
    // old cap, a rows-bound size, and a single-chunk one.
    chunked_bary_matches(20, 2, 1, 1);
    chunked_bary_matches(20, 4, 2, 2);
    chunked_bary_matches(16, 2, 3, 3);
    chunked_bary_matches(12, 2, 2, 4);
}
