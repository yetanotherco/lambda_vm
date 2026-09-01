//! Tests for the BITWISE precomputed table.

use crate::tables::bitwise::{
    NUM_PRECOMPUTED_COLS, NUM_ROWS, bus_interactions, cols, generate_bitwise_row,
    generate_bitwise_trace, is_preprocessed, preprocessed_commitment, row_index,
};
use crate::tables::types::{BusId, FE};
use crate::test_utils::multi_prove_ram;
use math::field::element::FieldElement;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::Multiplicity;
use stark::proof::options::ProofOptions;

#[test]
fn test_row_index() {
    // First row: x=0, y=0, z=0
    assert_eq!(row_index(0, 0, 0), 0);

    // x=1, y=0, z=0
    assert_eq!(row_index(1, 0, 0), 1);

    // x=0, y=1, z=0
    assert_eq!(row_index(0, 1, 0), 256);

    // x=0, y=0, z=1
    assert_eq!(row_index(0, 0, 1), 256 * 256);

    // Last row: x=255, y=255, z=15
    assert_eq!(row_index(255, 255, 15), NUM_ROWS - 1);
}

#[test]
fn test_generate_bitwise_trace() {
    let trace = generate_bitwise_trace();

    // Check dimensions
    assert_eq!(trace.num_rows(), NUM_ROWS);

    // Check a few specific values
    // Row for x=5, y=3, z=0
    let row = row_index(5, 3, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::X], FE::from(5u64));
    assert_eq!(row_data[cols::Y], FE::from(3u64));
    assert_eq!(row_data[cols::Z], FE::from(0u64));
    assert_eq!(row_data[cols::AND], FE::from(1u64)); // 5 & 3 = 1
    assert_eq!(row_data[cols::OR], FE::from(7u64)); // 5 | 3 = 7
    assert_eq!(row_data[cols::XOR], FE::from(6u64)); // 5 ^ 3 = 6

    // Check MSB8 for x=128 (MSB set)
    let row = row_index(128, 0, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::MSB8], FE::from(1u64));

    // Check MSB8 for x=127 (MSB not set)
    let row = row_index(127, 0, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::MSB8], FE::from(0u64));

    // Check MSB16 for halfword = 32768 (0x8000)
    // 32768 = 0 + 256 * 128
    let row = row_index(0, 128, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::MSB16], FE::from(1u64));

    // Check shift: x=1, y=0, z=4 -> SLL = 16, SLLC = 0
    let row = row_index(1, 0, 4);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::SLL], FE::from(16u64)); // 1 << 4 = 16
    assert_eq!(row_data[cols::SLLC], FE::from(0u64)); // no carry

    // Check shift with carry: x=0, y=128, z=1 -> halfword=32768, SLL=0, SLLC=1
    let row = row_index(0, 128, 1);
    let row_data = trace.main_table.get_row(row);
    // 32768 << 1 = 65536 & 0xFFFF = 0
    assert_eq!(row_data[cols::SLL], FE::from(0u64));
    // 32768 >> (16-1) = 32768 >> 15 = 1
    assert_eq!(row_data[cols::SLLC], FE::from(1u64));
}

#[test]
fn test_zero_check() {
    let trace = generate_bitwise_trace();

    // ZERO should be 1 only when both X and Y are 0
    let row = row_index(0, 0, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::ZERO], FE::from(1u64));

    let row = row_index(1, 0, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::ZERO], FE::from(0u64));

    let row = row_index(0, 1, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::ZERO], FE::from(0u64));
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();
    // 7 non-BYTE_ALU lookups + 3 BYTE_ALU receivers (opsel AND/OR/XOR).
    assert_eq!(interactions.len(), 10);
}

#[test]
fn test_byte_alu_receivers() {
    let byte_alu: Vec<_> = bus_interactions()
        .into_iter()
        .filter(|i| i.bus_id == u64::from(BusId::ByteAlu))
        .collect();

    // One receiver per opsel (AND/OR/XOR), each carrying [opsel, X, Y, out].
    assert_eq!(byte_alu.len(), 3);
    for interaction in &byte_alu {
        assert!(!interaction.is_sender, "BYTE_ALU lookups are receivers");
        assert_eq!(interaction.values.len(), 4, "[opsel, X, Y, out]");
    }

    // Each opsel uses its own multiplicity column, reusing the precomputed
    // AND/OR/XOR result columns.
    let mut mu_columns: Vec<usize> = byte_alu
        .iter()
        .map(|i| match i.multiplicity {
            Multiplicity::Column(c) => c,
            _ => panic!("BYTE_ALU multiplicity must be a column"),
        })
        .collect();
    mu_columns.sort_unstable();
    assert_eq!(
        mu_columns,
        vec![
            cols::MU_BYTE_ALU_AND,
            cols::MU_BYTE_ALU_OR,
            cols::MU_BYTE_ALU_XOR
        ]
    );
}

#[test]
fn test_first_row() {
    // First row: x=0, y=0, z=0
    let trace = generate_bitwise_trace();
    let row = row_index(0, 0, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::X], FE::from(0u64));
    assert_eq!(row_data[cols::Y], FE::from(0u64));
    assert_eq!(row_data[cols::Z], FE::from(0u64));
    assert_eq!(row_data[cols::AND], FE::from(0u64)); // 0 & 0 = 0
    assert_eq!(row_data[cols::OR], FE::from(0u64)); // 0 | 0 = 0
    assert_eq!(row_data[cols::XOR], FE::from(0u64)); // 0 ^ 0 = 0
    assert_eq!(row_data[cols::MSB8], FE::from(0u64)); // MSB of 0 = 0
    assert_eq!(row_data[cols::MSB16], FE::from(0u64)); // MSB of 0 = 0
    assert_eq!(row_data[cols::ZERO], FE::from(1u64)); // 0 and 0 are both zero
    assert_eq!(row_data[cols::SLL], FE::from(0u64)); // 0 << 0 = 0
    assert_eq!(row_data[cols::SLLC], FE::from(0u64)); // 0 >> 16 = 0
}

#[test]
fn test_last_row() {
    // Last row: x=255, y=255, z=15
    let trace = generate_bitwise_trace();
    let row = row_index(255, 255, 15);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::X], FE::from(255u64));
    assert_eq!(row_data[cols::Y], FE::from(255u64));
    assert_eq!(row_data[cols::Z], FE::from(15u64));
    assert_eq!(row_data[cols::AND], FE::from(255u64)); // 255 & 255 = 255
    assert_eq!(row_data[cols::OR], FE::from(255u64)); // 255 | 255 = 255
    assert_eq!(row_data[cols::XOR], FE::from(0u64)); // 255 ^ 255 = 0
    assert_eq!(row_data[cols::MSB8], FE::from(1u64)); // MSB of 255 = 1
    // halfword = 255 + 256*255 = 65535 = 0xFFFF, MSB is bit 15 = 1
    assert_eq!(row_data[cols::MSB16], FE::from(1u64));
    assert_eq!(row_data[cols::ZERO], FE::from(0u64)); // not zero
    // SLL: (65535 << 15) & 0xFFFF = 0x8000 = 32768
    assert_eq!(row_data[cols::SLL], FE::from(32768u64));
    // SLLC: 65535 >> (16 - 15) = 65535 >> 1 = 32767
    assert_eq!(row_data[cols::SLLC], FE::from(32767u64));
}

#[test]
fn test_boundary_msb16() {
    let trace = generate_bitwise_trace();

    // halfword = 32767 (0x7FFF): MSB16 should be 0
    // 32767 = 255 + 256*127, so x=255, y=127
    let row = row_index(255, 127, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::MSB16], FE::from(0u64));

    // halfword = 32768 (0x8000): MSB16 should be 1
    // 32768 = 0 + 256*128, so x=0, y=128
    let row = row_index(0, 128, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::MSB16], FE::from(1u64));
}

#[test]
fn test_shift_boundaries() {
    let trace = generate_bitwise_trace();

    // Test SLL and SLLC at z=0 (no shift)
    // halfword = 1 (x=1, y=0)
    let row = row_index(1, 0, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::SLL], FE::from(1u64)); // 1 << 0 = 1
    assert_eq!(row_data[cols::SLLC], FE::from(0u64)); // 1 >> 16 = 0

    // Test SLL and SLLC at z=15 (max shift)
    // halfword = 1 (x=1, y=0)
    let row = row_index(1, 0, 15);
    let row_data = trace.main_table.get_row(row);
    // SLL: (1 << 15) & 0xFFFF = 0x8000 = 32768
    assert_eq!(row_data[cols::SLL], FE::from(32768u64));
    // SLLC: 1 >> (16 - 15) = 1 >> 1 = 0
    assert_eq!(row_data[cols::SLLC], FE::from(0u64));

    // Test with halfword = 0x8001 = 32769 (x=1, y=128)
    let row = row_index(1, 128, 1);
    let row_data = trace.main_table.get_row(row);
    // halfword = 1 + 256*128 = 32769 = 0x8001
    // SLL: (32769 << 1) & 0xFFFF = 65538 & 0xFFFF = 2
    assert_eq!(row_data[cols::SLL], FE::from(2u64));
    // SLLC: 32769 >> (16 - 1) = 32769 >> 15 = 1
    assert_eq!(row_data[cols::SLLC], FE::from(1u64));
}

#[test]
fn test_all_bitwise_operations() {
    let trace = generate_bitwise_trace();

    // Test with x=0xAA, y=0x55 (alternating bits)
    let row = row_index(0xAA, 0x55, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::AND], FE::from(0u64)); // 0xAA & 0x55 = 0
    assert_eq!(row_data[cols::OR], FE::from(0xFFu64)); // 0xAA | 0x55 = 0xFF
    assert_eq!(row_data[cols::XOR], FE::from(0xFFu64)); // 0xAA ^ 0x55 = 0xFF
    assert_eq!(row_data[cols::MSB8], FE::from(1u64)); // MSB of 0xAA = 1

    // Test with x=0x55, y=0xAA
    let row = row_index(0x55, 0xAA, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::AND], FE::from(0u64)); // 0x55 & 0xAA = 0
    assert_eq!(row_data[cols::OR], FE::from(0xFFu64)); // 0x55 | 0xAA = 0xFF
    assert_eq!(row_data[cols::XOR], FE::from(0xFFu64)); // 0x55 ^ 0xAA = 0xFF
    assert_eq!(row_data[cols::MSB8], FE::from(0u64)); // MSB of 0x55 = 0
}

#[test]
fn test_row_count() {
    let trace = generate_bitwise_trace();
    // 256 * 256 * 16 = 2^20 = 1048576
    assert_eq!(trace.num_rows(), 256 * 256 * 16);
    assert_eq!(trace.num_rows(), 1048576);
}

#[test]
fn test_is_preprocessed() {
    // Bitwise table should be marked as preprocessed
    assert!(is_preprocessed());
}

#[test]
fn test_generate_bitwise_row_matches_trace() {
    // Verify that the const fn generate_bitwise_row produces identical values
    // to the runtime generate_bitwise_trace function.
    let trace = generate_bitwise_trace();

    // Test boundary cases and representative samples
    let test_indices = [
        0,                        // First row: x=0, y=0, z=0
        1,                        // x=1, y=0, z=0
        255,                      // x=255, y=0, z=0
        256,                      // x=0, y=1, z=0
        256 * 256,                // x=0, y=0, z=1
        256 * 256 * 15,           // x=0, y=0, z=15
        NUM_ROWS - 1,             // Last row: x=255, y=255, z=15
        row_index(0xAA, 0x55, 7), // Alternating bits
        row_index(0x55, 0xAA, 8), // Alternating bits reversed
        row_index(128, 128, 4),   // MSB set
        row_index(1, 0, 15),      // Shift boundary
        row_index(0, 128, 1),     // Carry case
    ];

    for &idx in &test_indices {
        let const_row = generate_bitwise_row(idx);
        let trace_row = trace.main_table.get_row(idx);

        // Verify all 11 precomputed columns match
        assert_eq!(const_row.len(), NUM_PRECOMPUTED_COLS);

        assert_eq!(
            const_row[0],
            trace_row[cols::X].canonical_u64(),
            "X mismatch at index {idx}"
        );
        assert_eq!(
            const_row[1],
            trace_row[cols::Y].canonical_u64(),
            "Y mismatch at index {idx}"
        );
        assert_eq!(
            const_row[2],
            trace_row[cols::Z].canonical_u64(),
            "Z mismatch at index {idx}"
        );
        assert_eq!(
            const_row[3],
            trace_row[cols::AND].canonical_u64(),
            "AND mismatch at index {idx}"
        );
        assert_eq!(
            const_row[4],
            trace_row[cols::OR].canonical_u64(),
            "OR mismatch at index {idx}"
        );
        assert_eq!(
            const_row[5],
            trace_row[cols::XOR].canonical_u64(),
            "XOR mismatch at index {idx}"
        );
        assert_eq!(
            const_row[6],
            trace_row[cols::MSB8].canonical_u64(),
            "MSB8 mismatch at index {idx}"
        );
        assert_eq!(
            const_row[7],
            trace_row[cols::MSB16].canonical_u64(),
            "MSB16 mismatch at index {idx}"
        );
        assert_eq!(
            const_row[8],
            trace_row[cols::ZERO].canonical_u64(),
            "ZERO mismatch at index {idx}"
        );
        assert_eq!(
            const_row[9],
            trace_row[cols::SLL].canonical_u64(),
            "SLL mismatch at index {idx}"
        );
        assert_eq!(
            const_row[10],
            trace_row[cols::SLLC].canonical_u64(),
            "SLLC mismatch at index {idx}"
        );
    }
}

#[test]
fn test_generate_bitwise_row_exhaustive_sample() {
    // Test a larger sample across different (x, y, z) regions
    let trace = generate_bitwise_trace();

    // Sample every 1000th row to cover the full range without being too slow
    for idx in (0..NUM_ROWS).step_by(1000) {
        let const_row = generate_bitwise_row(idx);
        let trace_row = trace.main_table.get_row(idx);

        for col in 0..NUM_PRECOMPUTED_COLS {
            let const_val = const_row[col];
            let trace_val = trace_row[col].canonical_u64();
            assert_eq!(
                const_val, trace_val,
                "Mismatch at index {idx}, column {col}: const={const_val}, trace={trace_val}"
            );
        }
    }
}

#[test]
fn test_preprocessed_commitment_is_deterministic() {
    let options = ProofOptions::default_test_options();
    // The commitment should be the same every time we access it
    let c1 = preprocessed_commitment(&options);
    let c2 = preprocessed_commitment(&options);
    assert_eq!(c1, c2, "Preprocessed commitment must be deterministic");
}

#[test]
fn test_preprocessed_commitment_is_nonzero() {
    let options = ProofOptions::default_test_options();
    // The commitment should not be all zeros
    let commitment = preprocessed_commitment(&options);
    assert!(
        commitment.iter().any(|&b| b != 0),
        "Preprocessed commitment should not be all zeros"
    );
}

// =============================================================================
// Soundness Test: Malicious Bitwise Table
// =============================================================================
//
// This test demonstrates the SECURITY VULNERABILITY when is_preprocessed()
// is not properly wired up to the AIR trait.
//
// Attack scenario:
// 1. Malicious prover creates a "backdoored" bitwise table where AND[5,3] = 99
//    (instead of the correct value 1)
// 2. Sender table requests AND[5,3] and claims the result is 99
// 3. LogUp verification passes because sender and receiver values match
// 4. But the computation is WRONG - the prover has cheated!
//
// With proper preprocessed table support:
// - Verifier uses HARDCODED commitment to the correct table
// - Prover's fake table has different commitment
// - Fiat-Shamir challenges differ → proof verification FAILS
//
// WITHOUT proper preprocessed support (current state):
// - Verifier uses commitment FROM PROOF (which prover controls)
// - Prover's fake table commitment is accepted
// - Proof verification PASSES → UNSOUND!

#[cfg(test)]
mod soundness_tests {
    use super::*;
    use stark::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };
    use stark::proof::options::ProofOptions;
    use stark::prover::{IsStarkProver, Prover};
    use stark::trace::TraceTable;
    use stark::traits::AIR;
    use stark::verifier::IsStarkVerifier;

    use crate::tables::types::{GoldilocksExtension, GoldilocksField};

    type F = GoldilocksField;
    type E = GoldilocksExtension;

    // Column indices for sender (CPU-like) table
    mod sender_cols {
        pub const X: usize = 0;
        pub const Y: usize = 1;
        pub const AND_RESULT: usize = 2;
        pub const FLAG: usize = 3; // 1 = perform AND lookup
        pub const NUM_COLUMNS: usize = 4;
    }

    // Column indices for receiver (BITWISE-like) table
    mod receiver_cols {
        pub const X: usize = 0;
        pub const Y: usize = 1;
        pub const AND: usize = 2;
        pub const MU_AND: usize = 3;
        pub const NUM_COLUMNS: usize = 4;
    }

    fn create_sender_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        use crate::tables::types::{BusId, alu_op};

        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(sender_cols::FLAG),
                vec![
                    BusValue::constant(alu_op::AND as u64),
                    BusValue::Packed {
                        start_column: sender_cols::X,
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: sender_cols::Y,
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: sender_cols::AND_RESULT,
                        packing: Packing::Direct,
                    },
                ],
            )],
        };

        AirWithBuses::new(
            sender_cols::NUM_COLUMNS,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    fn create_receiver_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        create_receiver_air_impl(proof_options, None)
    }

    fn create_receiver_air_preprocessed(
        proof_options: &ProofOptions,
        commitment: stark::config::Commitment,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        // 3 precomputed columns: X, Y, AND (column 3 = MU_AND is multiplicity)
        create_receiver_air_impl(proof_options, Some((commitment, 3)))
    }

    fn create_receiver_air_impl(
        proof_options: &ProofOptions,
        preprocessed: Option<(stark::config::Commitment, usize)>,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        use crate::tables::types::{BusId, alu_op};

        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                BusId::ByteAlu,
                Multiplicity::Column(receiver_cols::MU_AND),
                vec![
                    BusValue::constant(alu_op::AND as u64),
                    BusValue::Packed {
                        start_column: receiver_cols::X,
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: receiver_cols::Y,
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: receiver_cols::AND,
                        packing: Packing::Direct,
                    },
                ],
            )],
        };

        let air = AirWithBuses::new(
            receiver_cols::NUM_COLUMNS,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        );

        match preprocessed {
            Some((commitment, num_precomputed)) => {
                air.with_preprocessed(commitment, num_precomputed)
            }
            None => air,
        }
    }

    /// Compute commitment to a receiver trace (for testing preprocessed tables).
    ///
    /// Uses the same LDE commitment computation as the prover:
    /// 1. Interpolate columns to polynomials
    /// 2. Evaluate on LDE domain
    /// 3. Bit-reverse permute
    /// 4. Build Merkle tree from rows
    fn compute_trace_commitment(
        trace: &TraceTable<F, E>,
        proof_options: &ProofOptions,
    ) -> stark::config::Commitment {
        // Create a dummy AIR just to get the domain parameters
        let dummy_air = create_receiver_air(proof_options);

        // Use the prover's commitment computation (3 precomputed cols: X, Y, AND)
        Prover::compute_precomputed_commitment_for_testing(trace, &dummy_air, 3)
            .expect("Failed to compute commitment")
    }

    fn create_sender_trace(x: u8, y: u8, claimed_result: u8) -> TraceTable<F, E> {
        let num_rows = 4; // minimum power of 2
        let mut data = vec![FE::zero(); num_rows * sender_cols::NUM_COLUMNS];

        // Row 0: perform AND lookup
        data[sender_cols::X] = FE::from(x as u64);
        data[sender_cols::Y] = FE::from(y as u64);
        data[sender_cols::AND_RESULT] = FE::from(claimed_result as u64);
        data[sender_cols::FLAG] = FE::one();

        TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
    }

    fn create_malicious_receiver_trace(x: u8, y: u8, fake_and: u8) -> TraceTable<F, E> {
        let num_rows = 4;
        let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

        // Row 0: MALICIOUS - wrong AND value!
        data[receiver_cols::X] = FE::from(x as u64);
        data[receiver_cols::Y] = FE::from(y as u64);
        data[receiver_cols::AND] = FE::from(fake_and as u64); // WRONG!
        data[receiver_cols::MU_AND] = FE::one();

        TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
    }

    fn create_honest_receiver_trace(x: u8, y: u8) -> TraceTable<F, E> {
        let num_rows = 4;
        let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

        // Row 0: CORRECT AND value
        let correct_and = x & y;
        data[receiver_cols::X] = FE::from(x as u64);
        data[receiver_cols::Y] = FE::from(y as u64);
        data[receiver_cols::AND] = FE::from(correct_and as u64);
        data[receiver_cols::MU_AND] = FE::one();

        TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
    }

    /// Test that honest execution (correct AND result) is accepted.
    #[test]
    fn test_honest_bitwise_accepted() {
        let x = 5u8;
        let y = 3u8;
        let correct_result = x & y; // = 1

        let mut sender_trace = create_sender_trace(x, y, correct_result);
        let mut receiver_trace = create_honest_receiver_trace(x, y);

        let proof_options = ProofOptions::default_test_options();
        let sender_air = create_sender_air(&proof_options);
        let receiver_air = create_receiver_air(&proof_options);

        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![
            (&sender_air, &mut sender_trace, &()),
            (&receiver_air, &mut receiver_trace, &()),
        ];

        let multi_proof =
            multi_prove_ram(air_trace_pairs, &mut crate::hash_pin::block_transcript(&[])).unwrap();

        let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
            vec![&sender_air, &receiver_air];

        let result = crate::hash_pin::BlockVerifier::multi_verify(
            &airs,
            &multi_proof,
            &mut crate::hash_pin::block_transcript(&[]),
            &FieldElement::zero(),
        );

        assert!(result, "Honest execution should be accepted");
    }

    /// Documents that WITHOUT preprocessed marking, malicious tables are accepted.
    ///
    /// This test shows the vulnerability when is_preprocessed() returns false:
    /// - Malicious prover uses fake bitwise table with AND[5,3] = 99
    /// - Verifier trusts the commitment from the proof
    /// - Verification PASSES (unsound!)
    ///
    /// This test exists to document the attack vector and ensure we understand
    /// why preprocessed support is necessary.
    #[test]
    fn test_malicious_bitwise_without_preprocessed_is_accepted() {
        let x = 5u8;
        let y = 3u8;
        let fake_result = 99u8; // WRONG! Real answer is 5 & 3 = 1

        let mut sender_trace = create_sender_trace(x, y, fake_result);
        let mut receiver_trace = create_malicious_receiver_trace(x, y, fake_result);

        let proof_options = ProofOptions::default_test_options();
        let sender_air = create_sender_air(&proof_options);
        // NOT using preprocessed - verifier trusts proof's commitment
        let receiver_air = create_receiver_air(&proof_options);

        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![
            (&sender_air, &mut sender_trace, &()),
            (&receiver_air, &mut receiver_trace, &()),
        ];

        let multi_proof =
            multi_prove_ram(air_trace_pairs, &mut crate::hash_pin::block_transcript(&[])).unwrap();

        let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
            vec![&sender_air, &receiver_air];

        let result = crate::hash_pin::BlockVerifier::multi_verify(
            &airs,
            &multi_proof,
            &mut crate::hash_pin::block_transcript(&[]),
            &FieldElement::zero(),
        );

        // Without preprocessed support, malicious proof is ACCEPTED (vulnerability!)
        assert!(
            result,
            "Without preprocessed support, malicious proof should be accepted (documenting vulnerability)"
        );
    }

    /// SECURITY TEST: With preprocessed support, malicious tables are REJECTED.
    ///
    /// This test proves the security fix works:
    /// - Prover uses preprocessed AIR with commitment from MALICIOUS trace (wrong)
    /// - Verifier uses preprocessed AIR with commitment from HONEST trace (correct)
    /// - Prover's transcript has malicious_commitment, verifier's has honest_commitment
    /// - Fiat-Shamir challenges differ → verification FAILS
    ///
    /// This is the desired secure behavior.
    #[test]
    fn test_malicious_bitwise_with_preprocessed_is_rejected() {
        let x = 5u8;
        let y = 3u8;
        let fake_result = 99u8; // WRONG! Real answer is 5 & 3 = 1

        let proof_options = ProofOptions::default_test_options();

        // Compute commitment to the HONEST table (what verifier expects)
        let honest_trace = create_honest_receiver_trace(x, y);
        let honest_commitment = compute_trace_commitment(&honest_trace, &proof_options);

        // Compute commitment to the MALICIOUS table (what prover will use)
        let malicious_trace_for_commitment = create_malicious_receiver_trace(x, y, fake_result);
        let malicious_commitment =
            compute_trace_commitment(&malicious_trace_for_commitment, &proof_options);

        // Sender claims the fake result
        let mut sender_trace = create_sender_trace(x, y, fake_result);

        // Malicious receiver trace
        let mut malicious_trace = create_malicious_receiver_trace(x, y, fake_result);
        let sender_air = create_sender_air(&proof_options);

        // PROVER uses preprocessed AIR with MALICIOUS commitment (wrong values)
        let prover_receiver_air =
            create_receiver_air_preprocessed(&proof_options, malicious_commitment);

        // VERIFIER uses preprocessed AIR with HONEST commitment (correct values)
        let verifier_receiver_air =
            create_receiver_air_preprocessed(&proof_options, honest_commitment);

        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![
            (&sender_air, &mut sender_trace, &()),
            (&prover_receiver_air, &mut malicious_trace, &()),
        ];

        let multi_proof =
            multi_prove_ram(air_trace_pairs, &mut crate::hash_pin::block_transcript(&[])).unwrap();

        // Verifier uses DIFFERENT AIR with honest commitment
        let verifier_airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
            vec![&sender_air, &verifier_receiver_air];

        let result = crate::hash_pin::BlockVerifier::multi_verify(
            &verifier_airs,
            &multi_proof,
            &mut crate::hash_pin::block_transcript(&[]),
            &FieldElement::zero(),
        );

        // With preprocessed support, malicious proof is REJECTED (secure!)
        assert!(
            !result,
            "With preprocessed support, malicious bitwise table should be REJECTED"
        );
    }
}
