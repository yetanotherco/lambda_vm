//! PUBLIC_OUTPUT table for verifier-bound COMMIT output bytes.
//!
//! This table receives `(index, value)` lookups from the COMMIT table on the
//! `Commit` bus. The first two columns are preprocessed from `VmProof.public_output`,
//! so the verifier independently reconstructs the same commitment.

use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

pub mod cols {
    /// Preprocessed byte index.
    pub const INDEX: usize = 0;
    /// Preprocessed committed byte value.
    pub const VALUE: usize = 1;
    /// Main multiplicity column: 1 for real rows, 0 for padding.
    pub const MU: usize = 2;
    /// Total number of columns.
    pub const NUM_COLUMNS: usize = 3;
}

/// Number of preprocessed columns `(INDEX, VALUE)`.
pub const NUM_PREPROCESSED_COLS: usize = 2;

fn num_rows_for_output(public_output: &[u8]) -> usize {
    if public_output.is_empty() {
        1
    } else {
        public_output.len().next_power_of_two()
    }
}

/// Generate the PUBLIC_OUTPUT trace from committed bytes.
pub fn generate_public_output_trace(
    public_output: &[u8],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = num_rows_for_output(public_output);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, value) in public_output.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::INDEX] = FE::from(row_idx as u64);
        data[base + cols::VALUE] = FE::from(*value as u64);
        data[base + cols::MU] = FE::one();
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Compute the Merkle commitment for the preprocessed `(INDEX, VALUE)` columns.
pub fn compute_precomputed_commitment(public_output: &[u8], options: &ProofOptions) -> Commitment {
    let num_rows = num_rows_for_output(public_output);
    let mut index_col = vec![FE::zero(); num_rows];
    let mut value_col = vec![FE::zero(); num_rows];

    for (i, value) in public_output.iter().enumerate() {
        index_col[i] = FE::from(i as u64);
        value_col[i] = FE::from(*value as u64);
    }

    let columns = [index_col, value_col];

    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for public output column")
        })
        .collect();

    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for public output polynomial")
        })
        .collect();

    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    let lde_rows = columns2rows(lde_columns);
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for public output LDE");
    tree.root
}

/// Returns the preprocessed commitment for the PUBLIC_OUTPUT table.
pub fn preprocessed_commitment(public_output: &[u8], options: &ProofOptions) -> Commitment {
    compute_precomputed_commitment(public_output, options)
}

/// Creates the PUBLIC_OUTPUT bus receiver.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::receiver(
        BusId::Commit,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::INDEX,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE,
                packing: Packing::Direct,
            },
        ],
    )]
}

#[cfg(test)]
mod tests {
    use stark::proof::options::ProofOptions;

    use super::*;

    #[test]
    fn test_generate_public_output_trace() {
        let trace = generate_public_output_trace(b"abc");
        assert_eq!(trace.num_rows(), 4);

        let row0 = trace.main_table.get_row(0);
        assert_eq!(row0[cols::INDEX], FE::zero());
        assert_eq!(row0[cols::VALUE], FE::from(b'a' as u64));
        assert_eq!(row0[cols::MU], FE::one());

        let row3 = trace.main_table.get_row(3);
        assert_eq!(row3[cols::MU], FE::zero());
    }

    #[test]
    fn test_generate_public_output_trace_empty() {
        let trace = generate_public_output_trace(&[]);
        assert_eq!(trace.num_rows(), 1);
        let row0 = trace.main_table.get_row(0);
        assert_eq!(row0[cols::INDEX], FE::zero());
        assert_eq!(row0[cols::VALUE], FE::zero());
        assert_eq!(row0[cols::MU], FE::zero());
    }

    #[test]
    fn test_preprocessed_commitment_is_deterministic() {
        let opts = ProofOptions::default_test_options();
        let lhs = preprocessed_commitment(b"hello", &opts);
        let rhs = preprocessed_commitment(b"hello", &opts);
        assert_eq!(lhs, rhs);
    }
}
