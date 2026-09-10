//! WHIR against FRI on the same programs, same machine, CPU only.
//!
//! Ignored: these are measurements, not assertions. Run one with
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib \
//!     multilinear_bench_tests::shapes -- --ignored --nocapture
//! ```
//!
//! `RAYON_NUM_THREADS=1` on both sides is the algorithmic comparison — neither
//! prover gets credit for being better parallelised. All cores is the number a
//! user sees. The gap between them is the parallelisation backlog, which for
//! the multilinear path is most of it: only the Merkle build and the NTT are
//! parallel today.

use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use multilinear::whir_chain::GrindBits;
use stark::proof::options::GoldilocksCubicProofOptions;

use crate::multilinear_prove;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;

/// Blowup 4, 128 bits, 20 bits of grinding — the parameters the multilinear
/// path derives its own from, so the two are being asked for the same security.
fn options() -> stark::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_params(4, 128, 20).expect("valid options")
}

fn elf_bytes(name: &str) -> Vec<u8> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/program_artifacts");
    for dir in ["rust", "asm"] {
        let path = root.join(dir).join(format!("{name}.elf"));
        if let Ok(bytes) = std::fs::read(&path) {
            return bytes;
        }
    }
    panic!("no ELF named {name}");
}

/// The programs to sweep, smallest first: `(elf, private input)`.
///
/// `ethrex` with the 10-transfer block is the repo's reference workload. It runs
/// here **monolithic**, because the multilinear path has no continuations yet,
/// so it is bigger than the epochs the continuation bench proves.
const PROGRAMS: &[(&str, &str)] = &[
    ("sub", ""),
    ("all_instructions_64", ""),
    ("keccak", ""),
    ("ethrex", "ethrex_empty_block"),
    ("ethrex", "ethrex_simple_tx"),
    ("ethrex", "ethrex_bench_4"),
    ("ethrex", "ethrex_10_transfers"),
];

/// A private-input fixture from `executor/tests`, empty for a program that
/// takes none.
fn input_bytes(name: &str) -> Vec<u8> {
    if name.is_empty() {
        return Vec::new();
    }
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/tests")
        .join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|_| panic!("read {}", path.display()))
}

/// What each program costs in trace, without proving anything.
///
/// The multilinear prover's peak is driven by the **stacked** width
/// `num_vars + log2(columns)`: one codeword of `2^(n_stack + log_blowup)` base
/// elements per table, plus the extension-field fold on top of it. Printed here
/// so a program that cannot fit is ruled out before an hour is spent finding
/// out.
#[test]
#[ignore]
fn shapes() {
    println!(
        "\n{:<22} {:>7} {:>8} {:>7} {:>9} {:>12}",
        "program", "tables", "rows", "cols", "n_stack", "cells"
    );
    for &(name, input) in PROGRAMS {
        let label = if input.is_empty() {
            name.to_string()
        } else {
            input.to_string()
        };
        let bytes = elf_bytes(name);
        let inputs = input_bytes(input);
        let elf = match Elf::load(&bytes) {
            Ok(elf) => elf,
            Err(e) => {
                println!("{label:<22} load failed: {e:?}");
                continue;
            }
        };
        let logs = match Executor::new(&elf, inputs.clone()).and_then(Executor::run) {
            Ok(result) => result.logs,
            Err(e) => {
                println!("{label:<22} run failed: {e:?}");
                continue;
            }
        };
        let mut traces =
            match Traces::from_elf_and_logs(&elf, &logs, &MaxRowsConfig::default(), &inputs) {
                Ok(traces) => traces,
                Err(e) => {
                    println!("{label:<22} trace failed: {e:?}");
                    continue;
                }
            };
        let table_counts = traces.table_counts();
        let airs = crate::VmAirs::new(
            &elf,
            &options(),
            false,
            &traces.page_configs,
            &table_counts,
            None,
            true,
            None,
            None,
            None,
        );
        let pairs = airs.air_trace_pairs(&mut traces);

        // The widest stack sets the query count and the biggest single
        // allocation; the **cell total** is what normalises one prover against
        // another running different shapes, which is how the reference system
        // was compared — cells per second.
        let (tables, mut rows, mut cols, mut n_stack) = (pairs.len(), 0usize, 0usize, 0usize);
        let mut cells = 0u64;
        for (_, trace, _) in &pairs {
            let columns = trace.columns_main();
            let height = columns.first().map_or(0, Vec::len);
            let vars = height.trailing_zeros() as usize;
            let stack = vars + columns.len().next_power_of_two().trailing_zeros() as usize;
            cells += (height * columns.len()) as u64;
            if stack > n_stack {
                (rows, cols, n_stack) = (height, columns.len(), stack);
            }
        }
        println!(
            "{label:<22} {tables:>7} {rows:>8} {cols:>7} {n_stack:>9} {:>10.3}e9",
            cells as f64 / 1e9,
        );
    }
}

/// Proves one program and prints wall-clock and proof size.
///
/// `LAMBDA_VM_BENCH_ELF` picks the program, `LAMBDA_VM_BENCH_INPUT` its private
/// input fixture, `LAMBDA_VM_BENCH_BACKEND` which prover runs (`fri`, `whir`,
/// or both). One backend per process is what makes peak RSS attributable, so
/// the memory numbers come from
///
/// ```text
/// LAMBDA_VM_BENCH_BACKEND=whir /usr/bin/time -l cargo test --release ...
/// ```
#[test]
#[ignore]
fn whir_against_fri() {
    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let backend = std::env::var("LAMBDA_VM_BENCH_BACKEND").unwrap_or_else(|_| "both".into());
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let max_rows = MaxRowsConfig::default();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    let label = if input.is_empty() { &name } else { &input };
    println!("\n{label} — CPU, RAYON_NUM_THREADS={threads}, backend={backend}");

    let mib = |n: usize| n as f64 / (1024.0 * 1024.0);
    let mut fri = None;
    let mut whir = None;

    if backend != "whir" {
        let start = Instant::now();
        let proof = crate::prove_with_options_and_inputs(&bytes, &inputs, &opts, &max_rows)
            .expect("univariate prove");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .len();
        let start = Instant::now();
        let ok = crate::verify_with_options(&proof, &bytes, &opts, None, None).expect("verify");
        assert!(ok, "the univariate proof must verify");
        fri = Some((prove, start.elapsed(), size));
    }

    if backend != "fri" {
        let start = Instant::now();
        let proof =
            multilinear_prove::prove_with_options_and_inputs(&bytes, &inputs, &opts, &max_rows)
                .expect("multilinear prove");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .len();
        let start = Instant::now();
        let ok = multilinear_prove::verify_with_options(&proof, &bytes, &opts).expect("verify");
        assert!(ok, "the multilinear proof must verify");
        whir = Some((prove, start.elapsed(), size));
    }

    println!(
        "{:<12} {:>10} {:>10} {:>12}",
        "backend", "prove", "verify", "proof"
    );
    for (tag, run) in [("FRI", &fri), ("WHIR", &whir)] {
        if let Some((prove, verify, size)) = run {
            println!(
                "{tag:<12} {:>9.2}s {:>9.2}s {:>10.2} MiB",
                prove.as_secs_f64(),
                verify.as_secs_f64(),
                mib(*size)
            );
        }
    }
    if let (Some(f), Some(w)) = (fri, whir) {
        println!(
            "{:<12} {:>9.2}x {:>9.2}x {:>10.2}x",
            "WHIR/FRI",
            w.0.as_secs_f64() / f.0.as_secs_f64(),
            w.1.as_secs_f64() / f.1.as_secs_f64(),
            w.2 as f64 / f.2 as f64,
        );
    }
}

/// Where the multilinear prover's time goes, phase by phase.
///
/// Replays the same pipeline [`multilinear_prove::prove_with_options_and_inputs`]
/// runs, with a clock between the steps. The split that matters is **commit**
/// (NTT + Merkle, already parallel) against **argue** (sumcheck, GKR, LogUp,
/// stacking — all serial today): the second is the parallelisation backlog, and
/// its share is what says whether closing it is worth the work.
#[test]
#[ignore]
fn phases() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};

    use crate::test_utils::{E, F};

    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let label = if input.is_empty() { &name } else { &input };
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());

    let total = Instant::now();
    let start = Instant::now();
    let elf = Elf::load(&bytes).expect("load");
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .expect("run")
        .logs;
    let execute = start.elapsed();

    let start = Instant::now();
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, &MaxRowsConfig::default(), &inputs).expect("traces");
    let trace_build = start.elapsed();

    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &opts,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            let columns = trace.columns_main();
            (columns.len(), columns[0].len().trailing_zeros() as usize)
        })
        .collect();
    let mut config = multilinear_prove::chain_config(&shapes);
    // `LAMBDA_VM_BENCH_NO_GRIND` zeroes the proof of work while leaving the query
    // count alone. Not a valid proof — it drops the bits grinding buys — but it
    // is the only way to read the grinding cost off the same run, since asking
    // `with_security` for fewer bits would hand them back as extra queries.
    if std::env::var_os("LAMBDA_VM_BENCH_NO_GRIND").is_some() {
        config.grind = GrindBits::default();
    }

    let start = Instant::now();
    let layouts: Vec<TableLayout<'_, F, E>> = pairs
        .iter()
        .zip(&shapes)
        .map(|((air, _, _), &(width, num_vars))| {
            TableLayout::<F, E>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                width,
                num_vars,
                Uniforms::default(),
            )
            .expect("layout")
        })
        .collect();
    let layout = start.elapsed();

    // `commit` is now the one commitment the whole proof shares, so this phase
    // is where the stacking shows up.
    let start = Instant::now();
    let tables: Vec<CommittedTable<'_, F, E>> = layouts
        .into_iter()
        .zip(&pairs)
        .map(|(layout, (_, trace, _))| {
            let columns = trace.columns_main();
            CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
                .expect("materialize")
        })
        .collect();
    let count = tables.len();
    let committed = CommittedTables::commit(tables, &config).expect("commit");
    let commit = start.elapsed();

    let start = Instant::now();
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    let proof =
        multilinear_table::multi_prove(&committed, &config, &mut transcript).expect("prove");
    let argue = start.elapsed();
    let total = total.elapsed();

    println!(
        "\n{label} — CPU, RAYON_NUM_THREADS={threads}, {count} tables, {} queries, {} commitment(s)",
        config.num_queries,
        proof.roots.len(),
    );
    println!("{:<14} {:>9} {:>7}", "phase", "seconds", "share");
    for (tag, took) in [
        ("execute", execute),
        ("trace build", trace_build),
        ("layout", layout),
        ("commit", commit),
        ("argue", argue),
    ] {
        println!(
            "{tag:<14} {:>9.2} {:>6.1}%",
            took.as_secs_f64(),
            100.0 * took.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!("{:<14} {:>9.2}", "total", total.as_secs_f64());
    assert_eq!(proof.tables.len(), count);
}

/// What a proof is made of, part by part.
///
/// The multilinear proof grows with the **number of tables**, and the suspicion
/// is that the per-table WHIR opening is why: every table commits on its own,
/// so every table pays its own queries. This says how much of the bytes that
/// actually is, which is what decides whether stacking the tables together is
/// worth the work.
#[test]
#[ignore]
fn proof_composition() {
    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let proof = multilinear_prove::prove_with_options_and_inputs(
        &bytes,
        &inputs,
        &options(),
        &MaxRowsConfig::default(),
    )
    .expect("prove");

    let size = |v: &[u8]| v.len() as f64 / (1024.0 * 1024.0);
    let ser = |x: &dyn Fn() -> Vec<u8>| x();
    let total = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("whole")
        .len();

    // Each part on its own, summed across tables. The opening is not among
    // them: there is one for the whole proof, not one per table.
    let mut gkr = 0usize;
    let mut sumcheck = 0usize;
    let mut reduce = 0usize;
    let mut factor_values = 0usize;
    for t in &proof.proof.tables {
        gkr += rkyv::to_bytes::<rkyv::rancor::Error>(&t.gkr)
            .expect("gkr")
            .len();
        sumcheck += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.sumcheck)
            .expect("sumcheck")
            .len();
        reduce += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.reduce)
            .expect("reduce")
            .len();
        factor_values += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.factor_values)
            .expect("factor_values")
            .len();
    }
    let columns = rkyv::to_bytes::<rkyv::rancor::Error>(&proof.proof.columns)
        .expect("columns")
        .len();
    let _ = ser;

    let label = if input.is_empty() { &name } else { &input };
    println!(
        "\n{label} — {} tables, {} commitment(s)",
        proof.proof.tables.len(),
        proof.proof.roots.len(),
    );

    println!(
        "{:<18} {:>10} {:>8} {:>12}",
        "part", "MiB", "share", "per table"
    );
    for (tag, n) in [
        ("WHIR opening", columns),
        ("GKR", gkr),
        ("sumcheck", sumcheck),
        ("claim reduce", reduce),
        ("factor values", factor_values),
    ] {
        println!(
            "{tag:<18} {:>10.2} {:>7.1}% {:>11.3}",
            size(&vec![0u8; n]),
            100.0 * n as f64 / total as f64,
            size(&vec![0u8; n]) / proof.proof.tables.len() as f64,
        );
    }
    println!("{:<18} {:>10.2}", "whole proof", size(&vec![0u8; total]));
}

/// How big the constraint DAG is per table — the thing `IrShape::combine`
/// walks once per hypercube index.
#[test]
#[ignore]
fn constraint_program_sizes() {
    use stark::traits::AIR;
    let bytes = elf_bytes("ethrex");
    let inputs = input_bytes("ethrex_10_transfers");
    let elf = Elf::load(&bytes).unwrap();
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .unwrap()
        .logs;
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, &MaxRowsConfig::default(), &inputs).unwrap();
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &options(),
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    let mut sizes: Vec<(String, usize, usize)> = pairs
        .iter()
        .map(|(air, trace, _)| {
            (
                air.name().to_string(),
                air.constraint_program().nodes.len(),
                trace.columns_main()[0].len(),
            )
        })
        .collect();
    sizes.sort_by_key(|(_, n, _)| std::cmp::Reverse(*n));
    sizes.dedup_by(|a, b| a.0 == b.0);
    println!("\n{:<22} {:>10} {:>10}", "table", "DAG nodes", "rows");
    for (name, nodes, rows) in sizes.iter().take(8) {
        println!("{name:<22} {nodes:>10} {rows:>10}");
    }
}
