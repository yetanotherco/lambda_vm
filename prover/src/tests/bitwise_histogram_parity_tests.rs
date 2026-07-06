//! Byte-parity gate for the "histogram-on-the-fly" BITWISE multiplicity fill.
//!
//! The default trace-gen path accumulates BITWISE lookup multiplicities directly into a
//! dense `BitwiseHistogram` (per-thread, tree-reduced) instead of materializing the giant
//! `Vec<BitwiseOperation>` (~140 M ops at 10-tx) whose only consumer was the multiplicity
//! count. Under `LAMBDA_VM_LEGACY_TRACEGEN` the old path (materialize the Vec, then
//! `update_multiplicities`) is used.
//!
//! SOUNDNESS-CRITICAL: the two paths MUST produce a byte-identical BITWISE table — every
//! one of the 21 columns (11 preprocessed + 10 multiplicity) over all 2^20 rows. Because
//! the multiplicities are summed (a commutative monoid) the equality holds regardless of
//! accumulation order. These tests assert exact equality on real ethrex workloads.
//!
//! `#[ignore]`d (they execute + trace-build ethrex twice, which is slow); run explicitly:
//!   `cargo test -p lambda-vm-prover --release bitwise_histogram_parity -- --ignored --nocapture`

use std::path::PathBuf;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;

/// The `LAMBDA_VM_LEGACY_TRACEGEN` env var is process-global, so ALL tests that toggle it —
/// across every file — must share ONE mutex, otherwise a concurrent file's build can read the
/// flag mid-flip. Use the crate-wide lock, not a per-file one.
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

/// Build once with the legacy materialize-then-count path and once with the histogram
/// path, holding the env lock so the flag is stable across each build.
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
    let histogram = build_traces(elf, logs, private_input);

    (legacy, histogram)
}

fn assert_parity(name: &str, elf: &Elf, logs: &[executor::vm::logs::Log], private_input: &[u8]) {
    let (legacy, histogram) = build_both(elf, logs, private_input);

    assert_eq!(
        legacy.bitwise.num_rows(),
        histogram.bitwise.num_rows(),
        "[{name}] BITWISE row count differs"
    );
    assert_eq!(
        legacy.bitwise.num_cols(),
        histogram.bitwise.num_cols(),
        "[{name}] BITWISE col count differs"
    );

    // Full byte-parity: all 21 columns, all 2^20 rows, row-major.
    assert_eq!(
        legacy.bitwise.main_table.row_major_data(),
        histogram.bitwise.main_table.row_major_data(),
        "[{name}] BITWISE table data differs (NOT byte-identical)"
    );

    // Guard against a vacuous pass: at least one multiplicity column must be non-zero.
    use math::field::element::FieldElement;
    let any_mult = histogram
        .bitwise
        .main_table
        .row_major_data()
        .iter()
        .any(|fe: &FieldElement<crate::tables::types::GoldilocksField>| *fe.value() != 0);
    assert!(any_mult, "[{name}] BITWISE table all-zero — test is vacuous");

    eprintln!(
        "[{name}] BITWISE byte-parity OK ({} rows x {} cols)",
        legacy.bitwise.num_rows(),
        legacy.bitwise.num_cols()
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
fn bitwise_histogram_parity_ethrex_1tx() {
    let (elf, logs, input) = run_ethrex("ethrex_simple_tx.bin");
    assert_parity("ethrex-1tx", &elf, &logs, &input);
}

#[test]
#[ignore = "slow: executes + trace-builds ethrex twice"]
fn bitwise_histogram_parity_ethrex_10tx() {
    let (elf, logs, input) = run_ethrex("ethrex_10_transfers.bin");
    assert_parity("ethrex-10tx", &elf, &logs, &input);
}
