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

/// At most this many DECODE tables.
pub const MAX_CHUNKS: usize = 4;

/// `(first pc, instructions, rows)` per DECODE table. One power-of-two table
/// when it wastes at most a quarter of its rows, otherwise the largest power of
/// two that fits and the rest.
pub fn chunks(program: &Program, min_rows: usize) -> Vec<(usize, usize, usize)> {
    let mut plan = Vec::new();
    let (mut start, mut left) = (0, program.len());
    loop {
        let full = left.next_power_of_two().max(min_rows);
        let half = full / 2;
        if super::air::within_waste(left, full)
            || plan.len() + 1 == MAX_CHUNKS
            || half < min_rows
            || half >= left
        {
            plan.push((start, left, full));
            return plan;
        }
        plan.push((start, half, half));
        start += half;
        left -= half;
    }
}

/// One trace per chunk; `mult[pc]` executions of each address. Padding rows
/// repeat the halt entry, with zero multiplicity.
pub fn generate_traces(
    program: &Program,
    mult: &[u64],
    min_rows: usize,
) -> Vec<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let halt = decode_tuple(HALT_PC, &program.instrs[HALT_PC as usize]);
    chunks(program, min_rows)
        .into_iter()
        .map(|(start, len, rows)| {
            let mut trace = TraceTable::new_main(zeroed_fe_vec(rows * NUM_COLUMNS), NUM_COLUMNS, 1);
            let t = &mut trace.main_table;
            for row in 0..rows {
                let pc = start + row;
                let tuple = if row < len {
                    decode_tuple(pc as u64, &program.instrs[pc])
                } else {
                    halt
                };
                for (c, v) in tuple.into_iter().enumerate() {
                    t.set(row, c, v);
                }
                if row < len {
                    t.set(row, MU, FE::from(mult.get(pc).copied().unwrap_or(0)));
                }
            }
            trace
        })
        .collect()
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

/// The program id: per DECODE table, the Merkle root over the LDE of its
/// preprocessed columns, committed the way the prover commits the trace.
pub fn program_commitment(
    program: &Program,
    options: &ProofOptions,
    min_rows: usize,
) -> Vec<Commitment> {
    generate_traces(program, &[], min_rows)
        .iter()
        .map(|trace| {
            let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
                .map(|c| {
                    (0..trace.num_rows())
                        .map(|r| *trace.main_table.get(r, c))
                        .collect()
                })
                .collect();
            crate::lfm::commit::commit_columns(&columns, options)
        })
        .collect()
}
