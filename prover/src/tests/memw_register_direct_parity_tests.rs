//! Byte-parity gate for the direct-to-column MEMW_R trace fill.
//!
//! The default trace-gen path builds the MEMW_R (register fast-path) columns DIRECTLY
//! from compact `RegRow`s, without materializing an intermediate `Vec<MemwOperation>`
//! for register accesses. Under `LAMBDA_VM_LEGACY_TRACEGEN` the old path (materialize
//! `Vec<MemwOperation>` + fill from it) is used.
//!
//! SOUNDNESS-CRITICAL: the two paths MUST produce a byte-identical MEMW_R table (every
//! column, every chunk, padding included), and MUST leave the general/aligned MEMW
//! buckets unchanged (as a multiset — those tables ride a permutation-invariant bus).
//! These tests assert exactly that on real ethrex workloads (1-tx and 10-tx).
//!
//! These are `#[ignore]`d (they execute + trace-build ethrex twice, which is slow) and
//! run explicitly, e.g. `cargo test -p lambda-vm-prover --release
//! memw_register_direct_parity -- --ignored --nocapture`.

use std::path::PathBuf;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;

/// Shared crate-wide lock: the `LAMBDA_VM_LEGACY_TRACEGEN` env var is process-global, so this
/// MUST be the same mutex the other parity-test file uses (a per-file lock would not serialize
/// them across files under a parallel `--ignored` run).
use crate::tests::TRACEGEN_ENV_LOCK as ENV_LOCK;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn build_traces(elf: &Elf, logs: &[executor::vm::logs::Log], private_input: &[u8]) -> Traces {
    Traces::from_elf_and_logs(
        elf,
        logs,
        &MaxRowsConfig::default(),
        private_input,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build")
}

/// Build traces once with legacy MEMW_R fill and once with the direct fill, holding the
/// env lock so the flag is stable across each build.
fn build_both(
    elf: &Elf,
    logs: &[executor::vm::logs::Log],
    private_input: &[u8],
) -> (Traces, Traces) {
    let _guard = ENV_LOCK.lock().unwrap();

    // SAFETY: single-threaded within the lock; no other test reads the var concurrently.
    unsafe { std::env::set_var("LAMBDA_VM_LEGACY_TRACEGEN", "1") };
    let legacy = build_traces(elf, logs, private_input);

    unsafe { std::env::remove_var("LAMBDA_VM_LEGACY_TRACEGEN") };
    let direct = build_traces(elf, logs, private_input);

    (legacy, direct)
}

/// Multiset of a table's rows (row-major field-element bytes per row), for
/// permutation-invariant (bus) comparison.
fn row_multiset(
    tables: &[stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >],
) -> std::collections::BTreeMap<Vec<u64>, usize> {
    use math::field::element::FieldElement;
    let mut ms = std::collections::BTreeMap::new();
    for t in tables {
        let cols = t.num_cols();
        let data = t.main_table.row_major_data();
        for row in data.chunks(cols) {
            let key: Vec<u64> = row
                .iter()
                .map(|fe: &FieldElement<crate::tables::types::GoldilocksField>| *fe.value())
                .collect();
            *ms.entry(key).or_insert(0) += 1;
        }
    }
    ms
}

fn assert_parity(name: &str, elf: &Elf, logs: &[executor::vm::logs::Log], private_input: &[u8]) {
    let (legacy, direct) = build_both(elf, logs, private_input);

    // ---- MEMW_R: byte-identical, chunk-for-chunk, including padding ----
    assert_eq!(
        legacy.memw_registers.len(),
        direct.memw_registers.len(),
        "[{name}] MEMW_R chunk count differs"
    );
    let mut total_rows = 0usize;
    for (i, (l, d)) in legacy
        .memw_registers
        .iter()
        .zip(direct.memw_registers.iter())
        .enumerate()
    {
        assert_eq!(
            l.num_rows(),
            d.num_rows(),
            "[{name}] MEMW_R chunk {i} row count differs"
        );
        assert_eq!(
            l.num_cols(),
            d.num_cols(),
            "[{name}] MEMW_R chunk {i} col count differs"
        );
        assert_eq!(
            l.main_table.row_major_data(),
            d.main_table.row_major_data(),
            "[{name}] MEMW_R chunk {i} data differs (NOT byte-identical)"
        );
        total_rows += l.num_rows();
    }
    assert!(
        total_rows > 0,
        "[{name}] MEMW_R has no rows — test is vacuous"
    );

    // ---- General + aligned MEMW buckets: unchanged as a multiset ----
    assert_eq!(
        row_multiset(&legacy.memws),
        row_multiset(&direct.memws),
        "[{name}] general MEMW bucket multiset changed"
    );
    assert_eq!(
        row_multiset(&legacy.memw_aligneds),
        row_multiset(&direct.memw_aligneds),
        "[{name}] aligned MEMW bucket multiset changed"
    );

    eprintln!(
        "[{name}] MEMW_R byte-parity OK ({} chunks, {total_rows} rows); \
         general/aligned buckets multiset-identical",
        legacy.memw_registers.len()
    );
}

fn run_ethrex(bin: &str) -> (Elf, Vec<executor::vm::logs::Log>, Vec<u8>) {
    let root = workspace_root();
    let elf_bytes = std::fs::read(root.join("executor/program_artifacts/rust/ethrex.elf"))
        .expect("need ethrex.elf");
    let input = std::fs::read(root.join("executor/tests").join(bin))
        .unwrap_or_else(|_| panic!("need {bin}"));
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("execute");
    (elf, result.logs, input)
}

#[test]
#[ignore = "slow: executes + trace-builds ethrex twice"]
fn memw_register_direct_parity_ethrex_1tx() {
    let (elf, logs, input) = run_ethrex("ethrex_simple_tx.bin");
    assert_parity("ethrex-1tx", &elf, &logs, &input);
}

#[test]
#[ignore = "slow: executes + trace-builds ethrex twice"]
fn memw_register_direct_parity_ethrex_10tx() {
    let (elf, logs, input) = run_ethrex("ethrex_10_transfers.bin");
    assert_parity("ethrex-10tx", &elf, &logs, &input);
}
