//! The `FIELD_VM_DECODE` chip: the program, as preprocessed columns, plus the
//! per-proof multiplicity.
//!
//! Spec: `spec/src/field_vm_decode.toml`. The four packed `ExtField` columns
//! of the spec are eleven base columns here (the `X²` part of the flags
//! column is always zero, so it is not materialized).
//!
//! Padding rows repeat the halt entry, so they add no instruction to the program.

use stark::config::Commitment;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use super::air::decode_tuple;
use super::isa::{HALT_PC, Program};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

pub const NUM_PRECOMPUTED_COLS: usize = 11;
pub const MU: usize = NUM_PRECOMPUTED_COLS;
pub const NUM_COLUMNS: usize = NUM_PRECOMPUTED_COLS + 1;

pub fn num_rows(program: &Program, min_rows: usize) -> usize {
    program.len().next_power_of_two().max(min_rows)
}

/// `mult[pc]` executions of each address; padding rows get zero.
pub fn generate_trace(
    program: &Program,
    mult: &[u64],
    min_rows: usize,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let rows = num_rows(program, min_rows);
    let mut trace = TraceTable::new_main(zeroed_fe_vec(rows * NUM_COLUMNS), NUM_COLUMNS, 1);
    let t = &mut trace.main_table;
    let halt = decode_tuple(HALT_PC, &program.instrs[HALT_PC as usize]);
    for row in 0..rows {
        let tuple = match program.instrs.get(row) {
            Some(instr) => decode_tuple(row as u64, instr),
            None => halt,
        };
        for (c, v) in tuple.into_iter().enumerate() {
            t.set(row, c, v);
        }
        t.set(row, MU, FE::from(mult.get(row).copied().unwrap_or(0)));
    }
    trace
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::receiver(
        BusId::FieldVmDecode,
        Multiplicity::Column(MU),
        (0..NUM_PRECOMPUTED_COLS)
            .map(|c| BusValue::Packed {
                start_column: c,
                packing: Packing::Direct,
            })
            .collect(),
    )]
}

/// The program id: the Merkle root over the LDE of the preprocessed columns,
/// committed the same way the prover commits the trace.
pub fn program_commitment(
    program: &Program,
    options: &ProofOptions,
    min_rows: usize,
) -> Commitment {
    let trace = generate_trace(program, &[], min_rows);
    let rows = trace.num_rows();
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .map(|c| (0..rows).map(|r| *trace.main_table.get(r, c)).collect())
        .collect();
    crate::lfm::commit::commit_columns(&columns, options)
}
