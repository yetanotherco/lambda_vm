//! A host timing of the ECDAS halves rows' generator alone (the phase-5 fill
//! the block's ECDAS table takes, `try_generate_ecdas_rows_in` under the halves
//! form, the packed writer): 251 849 steps (the large block's), one thread,
//! the median of the reps. Run alone: a timing.

use super::ecdas::{self, CompactEcdasOp, EcdasForm, EcdasOperation};
use super::gpack::TraceForm;

fn step(i: usize) -> ::ecsm::EcdasStep {
    let bytes = |f: usize| -> [u8; 32] { std::array::from_fn(|b| ((i * 31 + f * 7 + b) % 251) as u8) };
    let wide = |f: usize| -> [u8; 33] { std::array::from_fn(|b| ((i * 17 + f * 5 + b) % 241) as u8) };
    let carries = |k: usize| -> [i64; 64] { std::array::from_fn(|j| ((i + k) as i64 * 37 + j as i64 * 11) % 4000 - 2000) };
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
#[ignore = "timing: the ECDAS halves generator alone, run alone"]
fn ecdas_halves_generator_timing() {
    let n = 251_849usize;
    let rows: Vec<CompactEcdasOp> = (0..n)
        .map(|i| CompactEcdasOp::from_op(&EcdasOperation { timestamp: 4 * i as u64 + 3, step: step(i) }))
        .collect();
    let num_rows = n.next_power_of_two();
    let reps: usize = std::env::var("ECDAS_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(9);
    let threads: usize = std::env::var("ECDAS_BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("a pool");
    let mut ms = Vec::new();
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let table = pool.install(|| ecdas::try_generate_ecdas_rows_in(&rows, num_rows, TraceForm::Narrow, EcdasForm::Halves));
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&table);
    }
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "ECDAS HALVES BENCH: {n} steps · {threads} threads · {reps} reps · median {:.1} ms · min {:.1} ms · all {:?}",
        sorted[reps / 2],
        sorted[0],
        ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
    );
}
