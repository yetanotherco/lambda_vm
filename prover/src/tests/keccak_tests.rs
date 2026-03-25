use crate::tables::keccak::{self, KeccakOperation, cols};
use crate::tables::trace_builder::Traces;
use crate::tables::types::FE;
use crate::test_utils::{E, F, asm_elf_bytes};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::execution::keccak_f1600;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::prover::IsStarkProver;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::IsStarkVerifier;

fn create_busless_keccak_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let (transition_constraints, _) = keccak::create_constraints(0);

    AirWithBuses::new(
        cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("KECCAK_TEST")
}

fn prove_and_verify_keccak_trace(trace: &mut TraceTable<F, E>) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let air = create_busless_keccak_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, trace, &())];

    let proof =
        stark::prover::Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
            .expect("Prover failed to generate keccak-only proof");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&air];
    stark::verifier::Verifier::multi_verify(&airs, &proof, &mut DefaultTranscript::<E>::new(&[]))
}

/// Test that the trace has the correct number of rows per permutation.
#[test]
fn test_keccak_trace_rows_per_permutation() {
    let input = [0u64; 25];
    let mut output = input;
    keccak_f1600(&mut output);

    let op = KeccakOperation {
        timestamp: 4,
        state_addr: 0x1000,
        input,
        output,
    };

    let trace = keccak::generate_keccak_trace(&[op]);
    // 24 rows rounded up to next power of 2 = 32
    assert_eq!(trace.num_rows(), 32);
}

/// Test that column count matches the constant.
#[test]
fn test_keccak_column_count() {
    assert_eq!(cols::NUM_COLUMNS, 3139);
}

/// Test trace generation with zero input: step flags, export, mu.
#[test]
fn test_keccak_trace_zero_input_flags() {
    let input = [0u64; 25];
    let mut output = input;
    keccak_f1600(&mut output);

    let op = KeccakOperation {
        timestamp: 4,
        state_addr: 0x1000,
        input,
        output,
    };

    let trace = keccak::generate_keccak_trace(&[op]);

    // Verify step flags are one-hot across 24 real rows
    for row in 0..24 {
        let r = trace.main_table.get_row(row);
        for i in 0..24 {
            if i == row {
                assert_eq!(
                    r[cols::STEP_FLAGS + i],
                    FE::one(),
                    "step_flags[{i}] should be 1 on row {row}"
                );
            } else {
                assert_eq!(
                    r[cols::STEP_FLAGS + i],
                    FE::zero(),
                    "step_flags[{i}] should be 0 on row {row}"
                );
            }
        }
    }

    // Verify export flag is only set on round 23
    for row in 0..24 {
        let r = trace.main_table.get_row(row);
        if row == 23 {
            assert_eq!(r[cols::EXPORT], FE::one());
        } else {
            assert_eq!(r[cols::EXPORT], FE::zero());
        }
    }

    // Verify first flag is only set on round 0
    for row in 0..24 {
        let r = trace.main_table.get_row(row);
        if row == 0 {
            assert_eq!(r[cols::FIRST], FE::one());
        } else {
            assert_eq!(r[cols::FIRST], FE::zero());
        }
    }

    // Verify mu=1 for real rows, mu=0 for padding
    for row in 0..24 {
        let r = trace.main_table.get_row(row);
        assert_eq!(r[cols::MU], FE::one(), "mu should be 1 for real row {row}");
    }
    for row in 24..32 {
        let r = trace.main_table.get_row(row);
        assert_eq!(
            r[cols::MU],
            FE::zero(),
            "mu should be 0 for padding row {row}"
        );
    }
}

/// Test trace generation with empty ops produces minimal padding table.
#[test]
fn test_keccak_trace_empty() {
    let trace = keccak::generate_keccak_trace(&[]);
    // min 4 rows
    assert_eq!(trace.num_rows(), 4);
}

/// Test multiple permutations produce correct row count.
#[test]
fn test_keccak_trace_multiple_permutations() {
    let input1 = [0u64; 25];
    let mut output1 = input1;
    keccak_f1600(&mut output1);

    let mut input2 = [0u64; 25];
    input2[0] = 1;
    let mut output2 = input2;
    keccak_f1600(&mut output2);

    let ops = vec![
        KeccakOperation {
            timestamp: 4,
            state_addr: 0x1000,
            input: input1,
            output: output1,
        },
        KeccakOperation {
            timestamp: 8,
            state_addr: 0x2000,
            input: input2,
            output: output2,
        },
    ];

    let trace = keccak::generate_keccak_trace(&ops);
    // 2 * 24 = 48 rows, next power of 2 = 64
    assert_eq!(trace.num_rows(), 64);

    // Verify both permutations have mu=1
    for row in 0..48 {
        let r = trace.main_table.get_row(row);
        assert_eq!(r[cols::MU], FE::one(), "mu should be 1 for real row {row}");
    }
}

/// Test that preimage on round 0 matches the input state.
#[test]
fn test_keccak_trace_preimage_round_0() {
    let mut input = [0u64; 25];
    input[0] = 0x0123456789ABCDEF;
    input[1] = 0xFEDCBA9876543210;
    let mut output = input;
    keccak_f1600(&mut output);

    let op = KeccakOperation {
        timestamp: 4,
        state_addr: 0x1000,
        input,
        output,
    };

    let trace = keccak::generate_keccak_trace(&[op]);
    let r0 = trace.main_table.get_row(0);

    // Check preimage[0][0] (lane at x=0, y=0) = input[0]
    // limb0 = 0xCDEF, limb1 = 0x89AB, limb2 = 0x4567, limb3 = 0x0123
    assert_eq!(r0[cols::PREIMAGE + 0], FE::from(0xCDEFu64));
    assert_eq!(r0[cols::PREIMAGE + 1], FE::from(0x89ABu64));
    assert_eq!(r0[cols::PREIMAGE + 2], FE::from(0x4567u64));
    assert_eq!(r0[cols::PREIMAGE + 3], FE::from(0x0123u64));

    // Check preimage[1][0] (lane at x=1, y=0) = input[1]
    assert_eq!(r0[cols::PREIMAGE + 4], FE::from(0x3210u64));
    assert_eq!(r0[cols::PREIMAGE + 5], FE::from(0x7654u64));
    assert_eq!(r0[cols::PREIMAGE + 6], FE::from(0xBA98u64));
    assert_eq!(r0[cols::PREIMAGE + 7], FE::from(0xFEDCu64));
}

/// Preimage stays constant across the 24 rows of a permutation, while A advances round-to-round.
#[test]
fn test_keccak_trace_preimage_constant_across_rounds() {
    let mut input = [0u64; 25];
    input[0] = 0x0123456789ABCDEF;
    input[1] = 0xFEDCBA9876543210;
    let mut output = input;
    keccak_f1600(&mut output);

    let op = KeccakOperation {
        timestamp: 4,
        state_addr: 0x1000,
        input,
        output,
    };

    let trace = keccak::generate_keccak_trace(&[op]);
    let r0 = trace.main_table.get_row(0);
    let r7 = trace.main_table.get_row(7);

    for col in cols::PREIMAGE..cols::PREIMAGE_END {
        assert_eq!(
            r0[col], r7[col],
            "preimage should stay constant across rounds"
        );
    }
}

#[test]
fn test_keccak_empty_trace_prove_and_verify() {
    let mut trace = keccak::generate_keccak_trace(&[]);
    assert!(prove_and_verify_keccak_trace(&mut trace));
}

#[test]
fn test_keccak_tampered_preimage_consistency_rejected() {
    let mut input = [0u64; 25];
    input[0] = 1;
    let mut output = input;
    keccak_f1600(&mut output);

    let op = KeccakOperation {
        timestamp: 4,
        state_addr: 0x1000,
        input,
        output,
    };

    let mut trace = keccak::generate_keccak_trace(&[op]);
    let original = trace.get_main(1, cols::PREIMAGE).clone();
    trace.set_main(1, cols::PREIMAGE, original + FE::one());

    assert!(!prove_and_verify_keccak_trace(&mut trace));
}

/// Test that trace generation succeeds for a program with keccak ECALL.
#[test]
fn test_keccak_trace_from_elf() {
    let elf_bytes = asm_elf_bytes("test_keccak");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor = Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    let traces = Traces::from_elf_and_logs(&elf, &result.logs, &Default::default()).unwrap();

    // Should have exactly 1 keccak permutation → 24 real rows, padded to 32
    assert_eq!(traces.keccak.num_rows(), 32);

    // Verify mu=1 for 24 real rows
    for row in 0..24 {
        let r = traces.keccak.main_table.get_row(row);
        assert_eq!(r[cols::MU], FE::one(), "mu should be 1 on real row {row}");
    }

    // Verify timestamp is set on real rows (should be the ECALL timestamp)
    let r0 = traces.keccak.main_table.get_row(0);
    assert_ne!(
        r0[cols::TIMESTAMP_0],
        FE::zero(),
        "timestamp should be nonzero on real rows"
    );
}

/// E2E prove and verify test for a program with keccak ECALL.
#[test]
fn test_keccak_prove_and_verify() {
    let elf_bytes = asm_elf_bytes("test_keccak");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor = Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    let mut traces = Traces::from_elf_and_logs(&elf, &result.logs, &Default::default()).unwrap();

    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).unwrap();

    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        false,
        &traces.page_configs,
        &traces.table_counts(),
    );

    let air_trace_pairs = airs.air_trace_pairs(&mut traces);

    let multi_proof = stark::prover::Prover::multi_prove(
        air_trace_pairs,
        &mut crypto::fiat_shamir::default_transcript::DefaultTranscript::<
            crate::test_utils::E,
        >::new(&[]),
    )
    .expect("Proving failed");

    let verified = stark::verifier::Verifier::multi_verify(
        &airs.air_refs(),
        &multi_proof,
        &mut crypto::fiat_shamir::default_transcript::DefaultTranscript::<
            crate::test_utils::E,
        >::new(&[]),
    );

    assert!(verified, "Proof verification failed for keccak program");
}
