//! A host timing of phase 5's packed fills run at once, as a prove runs them:
//! the large block's tables of four kinds, each on a thread of its own, all
//! started together each rep (ECDAS 4 × 2^17 rows, KECCAK 2 × 2^17, COPY 8 ×
//! 2^20, EQ 9 × 2^20: 23 threads, unpinned, as the shared pool's are). Each rep
//! prints every kind's summed fill time and the set's wall. Run alone: a
//! timing.

use super::ecdas::{self, CompactEcdasOp, EcdasForm, EcdasOperation};
use super::gpack::TraceForm;
use super::{copy, eq, keccak};
use std::sync::{Arc, Barrier};
use std::time::Instant;

fn ecdas_step(i: usize) -> ::ecsm::EcdasStep {
    let bytes =
        |f: usize| -> [u8; 32] { std::array::from_fn(|b| ((i * 31 + f * 7 + b) % 251) as u8) };
    let wide =
        |f: usize| -> [u8; 33] { std::array::from_fn(|b| ((i * 17 + f * 5 + b) % 241) as u8) };
    let carries = |k: usize| -> [i64; 64] {
        std::array::from_fn(|j| ((i + k) as i64 * 37 + j as i64 * 11) % 4000 - 2000)
    };
    ::ecsm::EcdasStep {
        x_a: bytes(1),
        y_a: bytes(2),
        x_g: bytes(3),
        y_g: bytes(4),
        round: (i % 256) as u8,
        op: (i % 2) as u8,
        next_op: ((i + 1) % 2) as u8,
        lambda: bytes(5),
        x_r: bytes(6),
        y_r: bytes(7),
        q0: wide(8),
        q1: wide(9),
        q2: wide(10),
        c0: carries(0),
        c1: carries(1),
        c2: carries(2),
    }
}

#[test]
#[ignore = "timing: phase 5's packed fills at once, run alone"]
fn concurrent_packed_fills_timing() {
    let reps: usize = std::env::var("CONCURRENT_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let (e_rows, k_rows, c_rows, q_rows) = (1usize << 17, 1usize << 17, 1usize << 20, 1usize << 20);
    let ecdas_ops: Arc<Vec<CompactEcdasOp>> = Arc::new(
        (0..e_rows - 16)
            .map(|i| {
                CompactEcdasOp::from_op(&EcdasOperation {
                    timestamp: 4 * i as u64 + 3,
                    step: ecdas_step(i),
                })
            })
            .collect(),
    );
    let input: Vec<u8> = (0..136 * (k_rows - 16))
        .map(|i| (i * 131 + 7) as u8)
        .collect();
    let keccak_ops = Arc::new(keccak::sponge_ops(1000, 0x1000_0000, 0x3000_0000, &input));
    let copy_ops = Arc::new(super::trace_builder::copy_ops_for_test(
        copy::Functionality::Copy,
        0x1000_0000,
        0x3000_0000,
        8 * (c_rows as u64 - 16),
    ));
    let eq_ops: Arc<Vec<eq::EqOperation>> = Arc::new(
        (0..q_rows - 16)
            .map(|i| {
                let a = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                eq::EqOperation::new(a, if i % 3 == 0 { a } else { a ^ (i as u64) }, i % 5 == 0)
            })
            .collect(),
    );
    assert!(keccak_ops.len() <= k_rows && copy_ops.len() <= c_rows);
    let kinds: Vec<(&'static str, usize)> =
        vec![("ecdas", 4), ("keccak", 2), ("copy", 8), ("eq", 9)];
    let threads: usize = kinds.iter().map(|k| k.1).sum();
    let start = Arc::new(Barrier::new(threads + 1));
    let done = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::new();
    for (kind, n) in kinds.iter().copied() {
        for _ in 0..n {
            let (start, done) = (start.clone(), done.clone());
            let (eo, ko, co, qo) = (
                ecdas_ops.clone(),
                keccak_ops.clone(),
                copy_ops.clone(),
                eq_ops.clone(),
            );
            handles.push(std::thread::spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("a pool");
                let mut ms = Vec::with_capacity(reps);
                for _ in 0..reps {
                    start.wait();
                    let t = Instant::now();
                    pool.install(|| match kind {
                        "ecdas" => drop(std::hint::black_box(ecdas::generate_ecdas_rows_in(
                            &eo,
                            e_rows,
                            TraceForm::Narrow,
                            EcdasForm::Halves,
                        ))),
                        "keccak" => drop(std::hint::black_box(keccak::generate_keccak_rows_as(
                            &ko,
                            k_rows,
                            TraceForm::Narrow,
                        ))),
                        "copy" => drop(std::hint::black_box(copy::generate_copy_rows_as(
                            &co,
                            c_rows,
                            TraceForm::Narrow,
                        ))),
                        _ => drop(std::hint::black_box(eq::generate_eq_trace_as(
                            &qo,
                            TraceForm::Narrow,
                        ))),
                    });
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                    done.wait();
                }
                (kind, ms)
            }));
        }
    }
    let mut walls = Vec::with_capacity(reps);
    for _ in 0..reps {
        start.wait();
        let t = Instant::now();
        done.wait();
        walls.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let results: Vec<(&str, Vec<f64>)> = handles
        .into_iter()
        .map(|h| h.join().expect("a fill thread"))
        .collect();
    for r in 0..reps {
        let sum = |k: &str| {
            results
                .iter()
                .filter(|(n, _)| *n == k)
                .map(|(_, ms)| ms[r])
                .sum::<f64>()
        };
        println!(
            "CONCURRENT FILL BENCH: stagger {} · skip close {} · rep {} · wall {:.1} ms · ecdas {:.1} · keccak {:.1} · copy {:.1} · eq {:.1} ms (summed over each kind's tables)",
            stark::narrow::narrow_stagger_columns(),
            stark::narrow::bench::skip_close(),
            r + 1,
            walls[r],
            sum("ecdas"),
            sum("keccak"),
            sum("copy"),
            sum("eq"),
        );
    }
}
