//! Informal timing comparison for single-column and multi-column LDE.
//! Ignored by default; run with `cargo test ... -- --ignored --nocapture`.

use std::time::Instant;

use math::fft::bowers_fft::LayerTwiddles;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

type Fp = FieldElement<GoldilocksField>;

fn coset_weights(n: usize, g: u64) -> Vec<u64> {
    let inv_n = *FieldElement::<GoldilocksField>::from(n as u64)
        .inv()
        .unwrap()
        .value();
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &g);
    }
    w
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_2_to_18_blowup_4() {
    let log_n = 18;
    let blowup = 4;
    let n = 1usize << log_n;
    let lde = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let input: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let weights = coset_weights(n, 7);

    let _ = math_cuda::lde::coset_lde_base(&input, blowup, &weights).unwrap();

    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n as u64).unwrap();
    let fwd_tw = LayerTwiddles::<GoldilocksField>::new(lde.trailing_zeros() as u64).unwrap();
    let weights_fp: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();

    const TRIALS: u32 = 10;

    let t0 = Instant::now();
    for _ in 0..TRIALS {
        let _ = math_cuda::lde::coset_lde_base(&input, blowup, &weights).unwrap();
    }
    let gpu_ns = t0.elapsed().as_nanos() / TRIALS as u128;

    let t0 = Instant::now();
    for _ in 0..TRIALS {
        let mut buf: Vec<Fp> = input.iter().map(|&x| Fp::from_raw(x)).collect();
        Polynomial::coset_lde_full_expand::<GoldilocksField>(
            &mut buf,
            blowup,
            &weights_fp,
            &inv_tw,
            &fwd_tw,
        )
        .unwrap();
        std::hint::black_box(&buf);
    }
    let cpu_ns = t0.elapsed().as_nanos() / TRIALS as u128;

    let ratio = cpu_ns as f64 / gpu_ns as f64;
    println!(
        "single-column LDE 2^{log_n} blowup={blowup}: cpu={cpu_ns}ns gpu={gpu_ns}ns ratio={ratio:.2}x",
    );
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_2_to_16_blowup_4() {
    let log_n = 16;
    let blowup = 4;
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let input: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let weights = coset_weights(n, 7);

    let _ = math_cuda::lde::coset_lde_base(&input, blowup, &weights).unwrap();

    const TRIALS: u32 = 20;

    let t0 = Instant::now();
    for _ in 0..TRIALS {
        let _ = math_cuda::lde::coset_lde_base(&input, blowup, &weights).unwrap();
    }
    let gpu_ns = t0.elapsed().as_nanos() / TRIALS as u128;
    println!("single-column LDE 2^{log_n} blowup={blowup}: gpu={gpu_ns}ns");
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_multi_column_parallel() {
    // Simulates a multi-column workload processed via rayon: many columns
    // dispatched concurrently to stress the stream pool. log_n = 16 keeps
    // memory footprint manageable.
    let log_n = 16u32;
    let blowup = 4usize;
    let n = 1usize << log_n;
    let lde = n * blowup;
    let num_cols = 64;

    // Warm up.
    let _ = math_cuda::lde::coset_lde_base(&vec![0u64; n], blowup, &coset_weights(n, 7)).unwrap();

    // Build input data.
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();
    let weights = coset_weights(n, 7);
    let weights_fp: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();
    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n as u64).unwrap();
    let fwd_tw = LayerTwiddles::<GoldilocksField>::new(lde.trailing_zeros() as u64).unwrap();

    // GPU: rayon parallel across columns, each column picks a stream.
    let t0 = Instant::now();
    let _gpu_results: Vec<Vec<u64>> = columns
        .par_iter()
        .map(|col| math_cuda::lde::coset_lde_base(col, blowup, &weights).unwrap())
        .collect();
    let gpu_ns = t0.elapsed().as_nanos();

    // CPU: same rayon parallel pattern as the prover's `expand_columns_to_lde`.
    let mut cpu_bufs: Vec<Vec<Fp>> = columns
        .iter()
        .map(|c| c.iter().map(|&x| Fp::from_raw(x)).collect())
        .collect();
    let t0 = Instant::now();
    cpu_bufs.par_iter_mut().for_each(|buf| {
        Polynomial::coset_lde_full_expand::<GoldilocksField>(
            buf,
            blowup,
            &weights_fp,
            &inv_tw,
            &fwd_tw,
        )
        .unwrap();
    });
    let cpu_ns = t0.elapsed().as_nanos();

    let ratio = cpu_ns as f64 / gpu_ns as f64;
    println!(
        "{num_cols}-column LDE 2^{log_n} blowup={blowup}: cpu={cpu_ns}ns gpu={gpu_ns}ns ratio={ratio:.2}x  (cores={})",
        rayon::current_num_threads(),
    );
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_batched_prover_scale() {
    // Realistic large-table shape: ~1M rows, blowup 4, a few dozen columns.
    // Exercises batched LDE at prover-scale sizes.
    let log_n = 20u32; // 1M rows
    let blowup = 4usize;
    let n = 1usize << log_n;
    let num_cols = 20;

    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();
    let weights = coset_weights(n, 7);
    let weights_fp: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();
    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n as u64).unwrap();
    let fwd_tw =
        LayerTwiddles::<GoldilocksField>::new((n * blowup).trailing_zeros() as u64).unwrap();

    let warm_slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    for _ in 0..8 {
        let _ = math_cuda::lde::coset_lde_batch_base(&warm_slices, blowup, &weights).unwrap();
    }

    let slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    let mut gpu_samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = math_cuda::lde::coset_lde_batch_base(&slices, blowup, &weights).unwrap();
        gpu_samples.push(t0.elapsed().as_nanos());
    }
    gpu_samples.sort();
    let gpu_ns = gpu_samples[gpu_samples.len() / 2]; // median

    let mut cpu_samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let mut cpu_bufs: Vec<Vec<Fp>> = columns
            .iter()
            .map(|c| c.iter().map(|&x| Fp::from_raw(x)).collect())
            .collect();
        let t0 = Instant::now();
        cpu_bufs.par_iter_mut().for_each(|buf| {
            Polynomial::coset_lde_full_expand::<GoldilocksField>(
                buf,
                blowup,
                &weights_fp,
                &inv_tw,
                &fwd_tw,
            )
            .unwrap();
        });
        cpu_samples.push(t0.elapsed().as_nanos());
    }
    cpu_samples.sort();
    let cpu_ns = cpu_samples[cpu_samples.len() / 2]; // median

    let ratio = cpu_ns as f64 / gpu_ns as f64;
    println!(
        "prover-scale batched {num_cols} cols, log_n={log_n}, blowup={blowup}: cpu={cpu_ns}ns gpu={gpu_ns}ns ratio={ratio:.2}x (median of 10)",
    );
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_batched_vs_rayon_cpu() {
    let log_n = 16u32;
    let blowup = 4usize;
    let n = 1usize << log_n;
    let num_cols = 64;

    let mut rng = ChaCha8Rng::seed_from_u64(21);
    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();
    let weights = coset_weights(n, 7);

    // Warm up every stream slot so subsequent iterations don't pay the
    // one-time pinned staging alloc cost.
    let warm_slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    for _ in 0..64 {
        let _ = math_cuda::lde::coset_lde_batch_base(&warm_slices, blowup, &weights).unwrap();
    }
    let weights_fp: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();
    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n as u64).unwrap();
    let fwd_tw =
        LayerTwiddles::<GoldilocksField>::new((n * blowup).trailing_zeros() as u64).unwrap();

    // GPU batched — first run may include lazy device init; do a few to stabilise.
    let slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    let mut gpu_ns = u128::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = math_cuda::lde::coset_lde_batch_base(&slices, blowup, &weights).unwrap();
        gpu_ns = gpu_ns.min(t0.elapsed().as_nanos());
    }

    // CPU rayon (same pattern as prover).
    let mut cpu_bufs: Vec<Vec<Fp>> = columns
        .iter()
        .map(|c| c.iter().map(|&x| Fp::from_raw(x)).collect())
        .collect();
    let t0 = Instant::now();
    cpu_bufs.par_iter_mut().for_each(|buf| {
        Polynomial::coset_lde_full_expand::<GoldilocksField>(
            buf,
            blowup,
            &weights_fp,
            &inv_tw,
            &fwd_tw,
        )
        .unwrap();
    });
    let cpu_ns = t0.elapsed().as_nanos();

    let ratio = cpu_ns as f64 / gpu_ns as f64;
    println!(
        "batched {num_cols} cols, log_n={log_n}, blowup={blowup}: cpu={cpu_ns}ns gpu={gpu_ns}ns ratio={ratio:.2}x  (cores={})",
        rayon::current_num_threads(),
    );
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_multi_column_serialized_gpu() {
    use std::sync::Mutex;

    let log_n = 16u32;
    let blowup = 4usize;
    let n = 1usize << log_n;
    let num_cols = 64;

    let _ = math_cuda::lde::coset_lde_base(&vec![0u64; n], blowup, &coset_weights(n, 7)).unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();
    let weights = coset_weights(n, 7);

    // Single global Mutex so only one thread at a time calls GPU.
    let gpu_lock = Mutex::new(());
    let t0 = Instant::now();
    let _: Vec<Vec<u64>> = columns
        .par_iter()
        .map(|col| {
            let _guard = gpu_lock.lock().unwrap();
            math_cuda::lde::coset_lde_base(col, blowup, &weights).unwrap()
        })
        .collect();
    let gpu_ns = t0.elapsed().as_nanos();
    println!("GPU mutex-serialised from rayon: {gpu_ns}ns for {num_cols} cols");
}

#[test]
#[ignore = "informal perf probe; run with --ignored"]
fn bench_lde_multi_column_gpu_limited_threads() {
    // Same as multi_column_parallel but forces rayon to use only 8 threads
    // (matching the GPU stream pool rough capacity). Tests whether oversubscribed
    // rayon + many streams is the bottleneck.
    let gpu_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    let log_n = 16u32;
    let blowup = 4usize;
    let n = 1usize << log_n;
    let num_cols = 64;

    let _ = math_cuda::lde::coset_lde_base(&vec![0u64; n], blowup, &coset_weights(n, 7)).unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();
    let weights = coset_weights(n, 7);

    let t0 = Instant::now();
    let _gpu_results: Vec<Vec<u64>> = gpu_pool.install(|| {
        columns
            .par_iter()
            .map(|col| math_cuda::lde::coset_lde_base(col, blowup, &weights).unwrap())
            .collect()
    });
    let gpu_ns = t0.elapsed().as_nanos();

    let t0 = Instant::now();
    let _serial_gpu_results: Vec<Vec<u64>> = columns
        .iter()
        .map(|col| math_cuda::lde::coset_lde_base(col, blowup, &weights).unwrap())
        .collect();
    let gpu_serial_ns = t0.elapsed().as_nanos();

    println!(
        "GPU-only 8-thread: gpu-parallel={gpu_ns}ns gpu-serial={gpu_serial_ns}ns speedup={:.2}x",
        gpu_serial_ns as f64 / gpu_ns as f64,
    );
}
