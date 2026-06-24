//! Test-only helpers for building `Traces` with a trimmed bitwise table.
//!
//! These helpers are extracted here so that production source files stay free
//! of test-only code while still allowing the ~70 call sites in the test tree
//! to use `Traces::from_logs_minimal` / `Traces::from_elf_and_logs_minimal`
//! unchanged (inherent-impl in the same crate).

use executor::elf::Elf;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::trace::TraceTable;

use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::bitwise::cols;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

/// Removes rows where all multiplicity columns are zero (TEST ONLY).
///
/// This function is for tests only. The reduced table is NOT a valid
/// preprocessed table because:
/// 1. Row indices no longer match the (x, y, z) encoding
/// 2. The verifier cannot verify against a preprocessed commitment
/// 3. A malicious prover could claim incorrect bitwise results
///
/// This is acceptable for tests because we're testing:
/// - Bus interaction balancing (sends = receives)
/// - Constraint satisfaction
/// - LogUp protocol correctness
#[cfg(test)]
pub(crate) fn trim_zero_rows(
    trace: TraceTable<GoldilocksField, GoldilocksExtension>,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = trace.main_table.height;

    // Find rows with any non-zero multiplicity
    let kept_rows: Vec<usize> = (0..num_rows)
        .filter(|&row| {
            let row_data = trace.main_table.get_row(row);
            // Check all multiplicity columns, including rows used only by a
            // BYTE_ALU lookup.
            (cols::MU_MSB8..=cols::MU_BYTE_ALU_XOR).any(|col| row_data[col] != FE::zero())
        })
        .collect();

    if kept_rows.is_empty() {
        // No lookups - return minimal table with 16 rows of zeros
        let data = vec![FE::zero(); 16 * cols::NUM_COLUMNS];
        return TraceTable::new_main(data, cols::NUM_COLUMNS, 1);
    }

    // Determine new table size (next power of 2, minimum 16)
    let new_size = kept_rows.len().next_power_of_two().max(16);

    // Allocate new trace data
    let mut new_data = vec![FE::zero(); new_size * cols::NUM_COLUMNS];

    // Copy kept rows to new table
    for (new_row, &old_row) in kept_rows.iter().enumerate() {
        let old_row_data = trace.main_table.get_row(old_row);
        let base = new_row * cols::NUM_COLUMNS;
        for (col, &val) in old_row_data.iter().enumerate() {
            new_data[base + col] = val;
        }
    }

    TraceTable::new_main(new_data, cols::NUM_COLUMNS, 1)
}

#[cfg(test)]
impl Traces {
    /// Generates all traces with a trimmed bitwise table (TEST ONLY).
    ///
    /// Like [`Traces::from_logs`] but trims the bitwise table down to only
    /// rows with non-zero multiplicities. This makes the table much smaller for
    /// tests that don't exercise many distinct byte values.
    ///
    /// # Safety / Unsoundness
    ///
    /// The trimmed bitwise table is NOT a valid preprocessed table because:
    /// 1. The bitwise table is NOT preprocessed - the verifier checks the prover's
    ///    commitment instead of a hardcoded trusted commitment
    /// 2. A malicious prover could provide incorrect bitwise results and the
    ///    verifier would accept them (e.g., claim 5 AND 3 = 7)
    /// 3. The table structure differs from production (row indices don't match)
    ///
    /// This is acceptable for tests because we're testing:
    /// - Bus interaction balancing (sends = receives)
    /// - Constraint satisfaction
    /// - LogUp protocol correctness
    ///
    /// The full preprocessed bitwise verification is tested separately in the
    /// comprehensive `test_prove_elfs_all_instructions_64_full` test.
    pub fn from_logs_trimmed(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
        max_rows: &MaxRowsConfig,
    ) -> Result<Self, Error> {
        // Generate full traces (including full 2^20 bitwise table with multiplicities)
        let mut traces = Self::from_logs(logs, instructions, max_rows)?;

        // Trim the bitwise table to only rows with non-zero multiplicities
        traces.bitwise = trim_zero_rows(traces.bitwise);

        Ok(traces)
    }

    /// Generates all traces with a minimal bitwise table (TEST ONLY).
    ///
    /// Alias for `from_logs_trimmed` for backwards compatibility.
    pub fn from_logs_minimal(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
        max_rows: &MaxRowsConfig,
    ) -> Result<Self, Error> {
        Self::from_logs_trimmed(logs, instructions, max_rows)
    }

    /// Like [`from_elf_and_logs`] but trims the bitwise table (TEST ONLY).
    ///
    /// Produces PAGE and REGISTER tables (requires ELF) while keeping the
    /// bitwise table small. Same unsoundness caveats as [`from_logs_trimmed`].
    pub fn from_elf_and_logs_minimal(
        elf: &Elf,
        logs: &[Log],
        max_rows: &MaxRowsConfig,
        private_input: &[u8],
    ) -> Result<Self, Error> {
        let mut traces = Self::from_elf_and_logs(
            elf,
            logs,
            max_rows,
            private_input,
            #[cfg(feature = "disk-spill")]
            StorageMode::Ram,
        )?;
        traces.bitwise = trim_zero_rows(traces.bitwise);
        Ok(traces)
    }
}
