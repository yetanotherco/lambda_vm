//! Streaming and chunking must preserve every Keccak lookup multiplicity.
use crate::tables::{
    bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType},
    keccak::KeccakOperation,
    trace_builder::{collect_bitwise_from_keccak, for_each_keccak_bitwise_lookup},
};
use executor::vm::instruction::execution::keccak_f1600;

fn operations(n: usize) -> Vec<KeccakOperation> {
    let mut seed = 17u64;
    (0..n)
        .map(|i| {
            let input = core::array::from_fn(|lane| match i % 4 {
                0 => 0,
                1 => u64::MAX,
                2 => 0xaaaa_5555_aaaa_5555u64.rotate_left(lane as u32),
                _ => {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                }
            });
            let mut output = input;
            keccak_f1600(&mut output);
            KeccakOperation {
                timestamp: i as u64,
                // Exercise carries in the 25 aligned state pointers, and repeat
                // some inputs/addresses to test accumulation of duplicate lookups.
                state_addr: [0x1000, 0xfff8, 0xffff_fff8, 0x1000][i % 4],
                input,
                output,
            }
        })
        .collect()
}

fn buffered(ops: &[KeccakOperation], histogram: &mut BitwiseHistogram) {
    // Reproduce the previous materialize-then-count consumer. The operation
    // generator is unchanged by this refactor; existing round/bus tests check
    // its arithmetic, while this reference checks how its output is counted.
    let mut records = Vec::new();
    for_each_keccak_bitwise_lookup(ops, |op| records.push(op));
    histogram.add_ops(&records);
}

#[test]
fn keccak_streaming_preserves_buffered_multiplicities() {
    let ops = operations(9);
    let mut expected = BitwiseHistogram::new();
    let mut actual = BitwiseHistogram::new();
    let sentinel = BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, 17, 29);
    expected.bump(sentinel);
    actual.bump(sentinel);
    collect_bitwise_from_keccak(&[], &mut actual);
    assert!(actual == expected, "empty input must preserve prior counts");
    for end in [1, 3, 9] {
        buffered(&ops[..end], &mut expected);
        collect_bitwise_from_keccak(&ops[..end], &mut actual);
        assert!(actual == expected, "lookup multiplicities differ at {end}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn keccak_parallel_chunks_preserve_all_multiplicities() {
    use rayon::prelude::*;
    let ops = operations(11);
    let mut expected = BitwiseHistogram::new();
    buffered(&ops, &mut expected);
    // One worker, an uneven split and more workers than operations. This
    // exercises the same permutation boundaries as the trace-build scheduler.
    for (workers, input) in [(1, &ops[..]), (3, &ops[..]), (4, &ops[..2]), (4, &ops[..0])] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let chunk = input.len().div_ceil(workers).max(1);
        let merged = pool
            .install(|| {
                input
                    .par_chunks(chunk)
                    .map(|slice| {
                        let mut h = BitwiseHistogram::new();
                        collect_bitwise_from_keccak(slice, &mut h);
                        h
                    })
                    .reduce_with(|mut a, b| {
                        a.merge(&b);
                        a
                    })
            })
            .unwrap_or_else(BitwiseHistogram::new);
        if input.len() == ops.len() {
            assert!(
                merged == expected,
                "parallel counts differ with {workers} workers"
            );
        } else {
            let mut reference = BitwiseHistogram::new();
            buffered(input, &mut reference);
            assert!(merged == reference, "short/empty parallel input differs");
        }
    }
}

#[test]
#[ignore = "collector microbenchmark; run with --release --ignored --nocapture"]
fn bench_keccak_lookup_collectors() {
    use std::{hint::black_box, time::Instant};
    let n = std::env::var("KECCAK_LOOKUP_BENCH_N")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1024);
    let ops = operations(n);
    println!(
        "permutations={n} removed_vector_payload_bytes={}",
        n * 24_777 * std::mem::size_of::<BitwiseOperation>()
    );
    // Alternate order across runs to reduce warm-cache/order bias. Include
    // allocation and first-touch of the same 80 MiB histogram in both paths.
    for run in 0..6 {
        let order = if run % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for streaming in order {
            let start = Instant::now();
            let mut h = BitwiseHistogram::new();
            if streaming {
                collect_bitwise_from_keccak(black_box(&ops), &mut h);
            } else {
                buffered(black_box(&ops), &mut h);
            }
            black_box(&h);
            let elapsed = start.elapsed();
            println!(
                "run={run} streaming={streaming} elapsed_ms={:.3}",
                elapsed.as_secs_f64() * 1000.
            );
        }
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let workers = 4;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        // Both schedules use the same number of histogram buckets and the
        // same reduction. The old schedule assigns all Keccak work to one;
        // the new schedule distributes whole permutations among the buckets.
        for run in 0..6 {
            let order = if run % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for split in order {
                let start = Instant::now();
                let merged = pool.install(|| {
                    (0..workers)
                        .into_par_iter()
                        .map(|i| {
                            let mut h = BitwiseHistogram::new();
                            if split {
                                let chunk = ops.len().div_ceil(workers).max(1);
                                let lo = (i * chunk).min(ops.len());
                                let hi = (lo + chunk).min(ops.len());
                                collect_bitwise_from_keccak(&ops[lo..hi], &mut h);
                            } else if i == 0 {
                                buffered(&ops, &mut h);
                            }
                            h
                        })
                        .reduce_with(|mut a, b| {
                            a.merge(&b);
                            a
                        })
                        .unwrap()
                });
                black_box(&merged);
                let elapsed = start.elapsed();
                println!(
                    "run={run} workers={workers} split_streaming={split} elapsed_ms={:.3}",
                    elapsed.as_secs_f64() * 1000.
                );
            }
        }
    }
}
