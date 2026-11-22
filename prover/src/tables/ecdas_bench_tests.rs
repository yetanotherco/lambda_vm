//! A host timing of the ECDAS halves rows' generator alone (the phase-5 fill
//! the block's ECDAS table takes, `try_generate_ecdas_rows_in` under the halves
//! form, the packed writer): 251 849 steps (the large block's), one thread,
//! the median of the reps. Run alone: a timing.

use super::ecdas::{self, CompactEcdasOp, EcdasForm, EcdasOperation};
use super::gpack::TraceForm;

fn step(i: usize) -> ::ecsm::EcdasStep {
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

/// `arena.<MALLCTL_ARENAS_ALL>.purge`: every arena's unused pages back to the
/// system, so the next allocation faults fresh ones.
fn purge_all() {
    // SAFETY: a NUL-terminated name; a control that neither reads nor writes
    // takes null pointers and zero lengths.
    let rc = unsafe {
        tikv_jemalloc_sys::mallctl(
            c"arena.4096.purge".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(rc, 0, "arena.4096.purge");
}

#[test]
#[ignore = "timing: the ECDAS halves generator alone, run alone"]
fn ecdas_halves_generator_timing() {
    let n = 251_849usize;
    let rows: Vec<CompactEcdasOp> = (0..n)
        .map(|i| {
            CompactEcdasOp::from_op(&EcdasOperation {
                timestamp: 4 * i as u64 + 3,
                step: step(i),
            })
        })
        .collect();
    let num_rows = n.next_power_of_two();
    let reps: usize = std::env::var("ECDAS_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let threads: usize = std::env::var("ECDAS_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool");
    // ECDAS_BENCH_MEMORY: reuse (each rep's table freed, the next fill in
    // its memory), purge (every arena purged before each rep, so the fill
    // faults fresh pages), keep (every table held: fresh extents).
    let memory = std::env::var("ECDAS_BENCH_MEMORY").unwrap_or_else(|_| "reuse".to_string());
    assert!(
        ["reuse", "purge", "keep"].contains(&memory.as_str()),
        "ECDAS_BENCH_MEMORY {memory}"
    );
    // ECDAS_BENCH_WARM: untimed fills before rep 1 · ECDAS_BENCH_SLEEP_MS: a
    // pause before each rep · ECDAS_BENCH_NEW_POOL=1: each rep on a new
    // pool's new thread.
    let env_n = |k: &str| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };
    let (warm, sleep_ms, new_pool) = (
        env_n("ECDAS_BENCH_WARM"),
        env_n("ECDAS_BENCH_SLEEP_MS"),
        env_n("ECDAS_BENCH_NEW_POOL") == 1,
    );
    for _ in 0..warm {
        let table = pool.install(|| {
            ecdas::try_generate_ecdas_rows_in(&rows, num_rows, TraceForm::Narrow, EcdasForm::Halves)
        });
        std::hint::black_box(&table);
    }
    // ECDAS_BENCH_PLAN=widepack: each rep the 64-bit table, then packed.
    let plan = std::env::var("ECDAS_BENCH_PLAN").unwrap_or_else(|_| "narrow".to_string());
    assert!(
        ["narrow", "narrowu", "widepack"].contains(&plan.as_str()),
        "ECDAS_BENCH_PLAN {plan}"
    );
    let mut kept = Vec::new();
    let mut ms = Vec::new();
    for _ in 0..reps {
        if memory == "purge" {
            purge_all();
        }
        if sleep_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
        }
        let fresh_pool = new_pool.then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool")
        });
        let pool = fresh_pool.as_ref().unwrap_or(&pool);
        let t = std::time::Instant::now();
        // The synthetic steps break the relations, so the checked fill notes a
        // problem a row; the unchecked plans (narrowu, widepack) do not.
        let table = pool.install(|| match plan.as_str() {
            // The fill's first-time plan (no width hint yet): the 64-bit
            // table, then packed.
            "widepack" => Ok(super::gpack::packed(
                ecdas::generate_ecdas_rows_in(&rows, num_rows, TraceForm::Wide, EcdasForm::Halves),
                &ecdas::WIDTHS_HALVES,
            )),
            "narrowu" => Ok(ecdas::generate_ecdas_rows_in(
                &rows,
                num_rows,
                TraceForm::Narrow,
                EcdasForm::Halves,
            )),
            _ => ecdas::try_generate_ecdas_rows_in(
                &rows,
                num_rows,
                TraceForm::Narrow,
                EcdasForm::Halves,
            ),
        });
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&table);
        if memory == "keep" {
            kept.push(table);
        }
    }
    drop(kept);
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "ECDAS HALVES BENCH: {n} steps · {threads} threads · tile {} · memory {memory} · warm {warm} · sleep {sleep_ms} ms · new pool {new_pool} · plan {plan} · {reps} reps · median {:.1} ms · min {:.1} ms · all {:?}",
        stark::narrow::narrow_tile_rows(),
        sorted[reps / 2],
        sorted[0],
        ms.iter()
            .map(|v| (v * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
}

/// The COPY and EQ packed fills alone, on one thread, `PACKED_BENCH_REPS`
/// reps each (9), the median: COPY 2^20 rows (an 8 MiB memcpy, a row each 8
/// bytes) and EQ 2^20 rows (synthetic operands). The first rep of each builds
/// wide then packs (no width hint yet); the rest write packed. Run alone: a
/// timing.
#[test]
#[ignore = "timing: the COPY and EQ packed fills alone, run alone"]
fn copy_and_eq_packed_fills_timing() {
    use super::{copy, eq};
    let reps: usize = std::env::var("PACKED_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    let rows = 1usize << 20;
    let copy_ops = super::trace_builder::copy_ops_for_test(
        copy::Functionality::Copy,
        0x1000_0000,
        0x3000_0000,
        8 * (rows as u64 - 16),
    );
    assert!(copy_ops.len() <= rows, "{} COPY rows", copy_ops.len());
    let eq_ops: Vec<eq::EqOperation> = (0..rows - 16)
        .map(|i| {
            let a = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            eq::EqOperation::new(a, if i % 3 == 0 { a } else { a ^ (i as u64) }, i % 5 == 0)
        })
        .collect();
    let time = |name: &str, f: &(dyn Fn() + Sync)| {
        let mut ms = Vec::new();
        for _ in 0..reps {
            let t = std::time::Instant::now();
            pool.install(f);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut sorted = ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "PACKED BENCH: {name} · tile {} · {reps} reps · median {:.1} ms · min {:.1} ms · all {:?}",
            stark::narrow::narrow_tile_rows(),
            sorted[reps / 2],
            sorted[0],
            ms.iter()
                .map(|v| (v * 10.0).round() / 10.0)
                .collect::<Vec<_>>()
        );
    };
    time("copy", &|| {
        std::hint::black_box(copy::generate_copy_rows_as(
            &copy_ops,
            rows,
            TraceForm::Narrow,
        ));
    });
    time("eq", &|| {
        std::hint::black_box(eq::generate_eq_trace_as(&eq_ops, TraceForm::Narrow));
    });
}
