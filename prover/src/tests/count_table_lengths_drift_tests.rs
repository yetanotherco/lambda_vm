//! Asserts `count_table_lengths` matches `Traces::from_elf_and_logs` row counts.

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{Traces, count_table_lengths};
use crate::test_utils::run_asm_elf;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;

fn assert_count_table_lengths_matches(elf: &Elf, logs: &[Log]) {
    let max_rows = MaxRowsConfig::default();

    let predicted =
        count_table_lengths(elf, logs, &max_rows, &[]).expect("count_table_lengths succeeds");
    let traces =
        Traces::from_elf_and_logs_minimal(elf, logs, &max_rows, &[]).expect("trace build succeeds");

    let sum_heights = |tables: &[stark::trace::TraceTable<_, _>]| -> u64 {
        tables.iter().map(|t| t.main_table.height as u64).sum()
    };

    // Exact-match tables: predicted row count equals built trace.
    assert_eq!(predicted.cpu_padded_rows, sum_heights(&traces.cpus), "cpu");
    assert_eq!(
        predicted.memw_padded_rows,
        sum_heights(&traces.memws),
        "memw"
    );
    assert_eq!(
        predicted.memw_aligned_padded_rows,
        sum_heights(&traces.memw_aligneds),
        "memw_aligned"
    );
    assert_eq!(
        predicted.memw_register_padded_rows,
        sum_heights(&traces.memw_registers),
        "memw_register"
    );
    assert_eq!(
        predicted.load_padded_rows,
        sum_heights(&traces.loads),
        "load"
    );
    assert_eq!(
        predicted.shift_padded_rows,
        sum_heights(&traces.shifts),
        "shift"
    );
    assert_eq!(
        predicted.commit_padded_rows, traces.commit.main_table.height as u64,
        "commit"
    );
    assert_eq!(
        predicted.memmove_padded_rows, traces.memmove.main_table.height as u64,
        "memmove"
    );
    assert_eq!(
        predicted.decode_rows, traces.decode.main_table.height as u64,
        "decode"
    );

    // Upper-bound tables: predicted is `>=` actual (LT/MUL/DVRM/BRANCH dedup ops).
    assert!(
        predicted.lt_padded_rows >= sum_heights(&traces.lts),
        "lt: predicted={} actual={}",
        predicted.lt_padded_rows,
        sum_heights(&traces.lts)
    );
    assert!(
        predicted.mul_padded_rows >= sum_heights(&traces.muls),
        "mul: predicted={} actual={}",
        predicted.mul_padded_rows,
        sum_heights(&traces.muls)
    );
    assert!(
        predicted.dvrm_padded_rows >= sum_heights(&traces.dvrms),
        "dvrm: predicted={} actual={}",
        predicted.dvrm_padded_rows,
        sum_heights(&traces.dvrms)
    );
    assert!(
        predicted.branch_padded_rows >= sum_heights(&traces.branches),
        "branch: predicted={} actual={}",
        predicted.branch_padded_rows,
        sum_heights(&traces.branches)
    );

    // Auxiliary scalars.
    assert_eq!(predicted.cycle_count, logs.len() as u64, "cycle_count");
    assert_eq!(
        predicted.unique_page_count,
        traces.pages.len() as u64,
        "unique_page_count"
    );

    // Mirrors hardcoded `halt_rows = 1` in `auto_storage::table_specs`.
    assert_eq!(traces.halt.main_table.height, 1, "halt_rows");
}

#[test]
fn count_table_lengths_matches_traces() {
    let (elf, logs, _) = run_asm_elf("fib_iterative_372k");
    assert_count_table_lengths_matches(&elf, &logs);
}

/// Runs one Rust DMA guest and asserts the sizing pass matches the built traces.
/// Each ecall has two hand-maintained replays — `collect_dma_*_ops` for
/// generation and `replay_dma_*_for_sizing` for counting — and they must agree,
/// so the fixtures cover a single chunk plus the multi-chunk / overlapping /
/// near-`MAX_DATA_ROWS` cases of `dma_memcpy_cases`, and the same schedule
/// driven through the memset table.
fn assert_dma_fixture_counts(elf_name: &str, syscall_number: u64) {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join(format!("executor/program_artifacts/rust/{elf_name}")))
            .unwrap_or_else(|_| panic!("{elf_name} not found — build its make target"));
    let elf = Elf::load(&elf_bytes).expect("valid DMA guest ELF");
    let result = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("DMA guest execution");

    assert!(
        result.logs.iter().any(|log| {
            log.src1_val == syscall_number
                && matches!(
                    result.instructions.get(&log.current_pc),
                    Some(Instruction::EcallEbreak)
                )
        }),
        "fixture must contain a DMA ecall"
    );
    assert_count_table_lengths_matches(&elf, &result.logs);
}

#[test]
fn count_table_lengths_matches_nonempty_dma_trace() {
    use executor::vm::instruction::execution::{
        DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_SYSCALL_NUMBER,
    };

    assert_dma_fixture_counts("dma_memcpy_min.elf", DMA_MEMCPY_SYSCALL_NUMBER);
    assert_dma_fixture_counts("dma_memcpy_cases.elf", DMA_MEMCPY_SYSCALL_NUMBER);
    assert_dma_fixture_counts("dma_memset_min.elf", DMA_MEMSET_SYSCALL_NUMBER);
    assert_dma_fixture_counts("dma_memset_cases.elf", DMA_MEMSET_SYSCALL_NUMBER);
}

/// The `hint` ecall routes three register reads (`a0`/`a1`/`a2`) and four output
/// writes through the memory argument, plus two LT range-checks (selector, in_addr).
/// `count_table_lengths` must replay all of that exactly, or `memw_register` (an
/// exact-match table) drifts. Uses a real hint guest so the counts are non-trivial.
#[test]
fn count_table_lengths_matches_nonempty_hint_trace() {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/hint_min.elf"))
            .expect("hint_min.elf not found — run `make compile-programs-rust`");
    let elf = Elf::load(&elf_bytes).expect("valid hint guest ELF");
    let result = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("hint guest execution");

    assert!(
        result.logs.iter().any(|log| {
            log.src1_val == executor::vm::instruction::execution::HINT_SYSCALL_NUMBER
        }),
        "fixture must contain a hint ecall"
    );
    assert_count_table_lengths_matches(&elf, &result.logs);
}
