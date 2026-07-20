//! Asserts `count_table_lengths` matches `Traces::from_elf_and_logs` row counts.

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{Traces, count_table_lengths};
use crate::test_utils::run_asm_elf;

#[test]
fn count_table_lengths_matches_traces() {
    let (elf, logs, _) = run_asm_elf("fib_iterative_372k");
    let max_rows = MaxRowsConfig::default();

    let predicted =
        count_table_lengths(&elf, &logs, &max_rows, &[]).expect("count_table_lengths succeeds");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &max_rows, &[])
        .expect("trace build succeeds");

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

/// Same drift check for a FEXT-exercising program: the accelerator tables must be
/// accounted for so `auto_storage`'s peak-RAM estimate does not undercount.
#[test]
fn count_table_lengths_matches_fext_traces() {
    let (elf, logs, _) = run_asm_elf("fext_bench");
    let max_rows = MaxRowsConfig::default();

    let predicted =
        count_table_lengths(&elf, &logs, &max_rows, &[]).expect("count_table_lengths succeeds");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &max_rows, &[])
        .expect("trace build succeeds");

    let sum_heights = |tables: &[stark::trace::TraceTable<_, _>]| -> u64 {
        tables.iter().map(|t| t.main_table.height as u64).sum()
    };

    // Sanity: the program actually exercises the accelerator.
    assert!(
        traces.fext_fma.main_table.height > 4,
        "fext_bench should exercise FEXT_FMA (height={})",
        traces.fext_fma.main_table.height
    );

    // FEXT tables: one row per ecall / touched cell, no dedup → exact match.
    assert_eq!(
        predicted.fext_load_padded_rows, traces.fext_load.main_table.height as u64,
        "fext_load"
    );
    assert_eq!(
        predicted.fext_fma_padded_rows, traces.fext_fma.main_table.height as u64,
        "fext_fma"
    );
    assert_eq!(
        predicted.fext_store_padded_rows, traces.fext_store.main_table.height as u64,
        "fext_store"
    );
    assert_eq!(
        predicted.fext_page_padded_rows, traces.fext_page.main_table.height as u64,
        "fext_page"
    );

    // LT stays an upper bound (dedup) and now includes the fext-induced
    // `old_ts < ts` sends, so it must still cover the actual LT rows.
    assert!(
        predicted.lt_padded_rows >= sum_heights(&traces.lts),
        "lt: predicted={} actual={}",
        predicted.lt_padded_rows,
        sum_heights(&traces.lts)
    );
}
