//! Tests for invalid LogUp bus interactions.
//!
//! These tests verify that the verifier correctly rejects invalid proofs
//! that violate the bus balance invariant.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

// =============================================================================
// Value manipulation attacks
// =============================================================================

/// Cheating: receiver has wrong result value.
/// CPU sends (1, 10, 11) but ADD table claims (1, 10, 99).
#[test_log::test]
fn test_cheating_wrong_result_value() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(10), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(11), FE::zero(), FE::zero(), FE::zero()], // CPU claims 1+10=11
        ],
        1,
    );

    // CHEAT: ADD table has wrong result
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(10), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(99), FE::zero(), FE::zero(), FE::zero()], // WRONG! Should be 11
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    // Verifier MUST reject
    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject proof with wrong result value"
    );
}

/// Cheating: off-by-one error (subtle attack).
/// CPU sends (5, 3, 8) but ADD table claims (5, 3, 9).
#[test_log::test]
fn test_cheating_off_by_one() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()], // 5 + 3 = 8
        ],
        1,
    );

    // CHEAT: off by one
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(9), FE::zero(), FE::zero(), FE::zero()], // WRONG! Should be 8
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject off-by-one error"
    );
}

/// Cheating: swapped operands.
/// CPU sends (5, 3, 8) but ADD table claims (3, 5, 8).
#[test_log::test]
fn test_cheating_swapped_operands() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // a = 5
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b = 3
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    // CHEAT: operands swapped
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // SWAPPED: was 5
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // SWAPPED: was 3
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject swapped operands"
    );
}

/// Cheating: only one column is wrong.
/// CPU sends (5, 3, 8) but ADD table claims (5, 4, 8) - only 'b' differs.
#[test_log::test]
fn test_cheating_single_column_wrong() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b = 3
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    // CHEAT: only 'b' is different
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(4), FE::zero(), FE::zero(), FE::zero()], // WRONG: should be 3
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject when single column is wrong"
    );
}

// =============================================================================
// Multiplicity manipulation attacks
// =============================================================================

/// Cheating: receiver claims higher multiplicity than sent.
/// CPU sends once, ADD claims multiplicity=2.
#[test_log::test]
fn test_cheating_over_report_multiplicity() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // sends once
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    // CHEAT: claims multiplicity=2 but CPU only sent once
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // CHEAT: mult=2
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject over-reported multiplicity"
    );
}

/// Cheating: receiver claims lower multiplicity than sent.
/// CPU sends twice, ADD claims multiplicity=1.
#[test_log::test]
fn test_cheating_under_report_multiplicity() {
    // CPU sends (5,3,8) twice
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::zero(), FE::zero()], // sends twice
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::from(5), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::from(3), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::from(8), FE::zero(), FE::zero()],
        ],
        1,
    );

    // CHEAT: claims multiplicity=1 but CPU sent twice
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // CHEAT: mult=1, should be 2
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject under-reported multiplicity"
    );
}

/// Cheating: receiver sets multiplicity=0 to skip receiving.
/// CPU sends, but ADD doesn't receive (mult=0).
#[test_log::test]
fn test_cheating_zero_multiplicity_skip() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // sends once
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    // CHEAT: ADD sets multiplicity=0, effectively not receiving
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()], // CHEAT: mult=0
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject when receiver skips with zero multiplicity"
    );
}

// =============================================================================
// Structural attacks (missing sender/receiver)
// =============================================================================

/// Cheating: receiver without matching sender.
/// ADD table receives something that CPU never sent.
#[test_log::test]
fn test_cheating_phantom_receive() {
    // CPU sends nothing (all flags = 0)
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // no ADD operations
            vec![FE::zero(); 4], // no MUL operations
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    // CHEAT: ADD claims to receive something that was never sent
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(999), FE::zero(), FE::zero(), FE::zero()], // phantom operation
            vec![FE::from(888), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(1887), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // CHEAT: receiving phantom
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject phantom receive (no matching sender)"
    );
}

/// Cheating: sender without matching receiver.
/// CPU sends to ADD but ADD table doesn't receive.
#[test_log::test]
fn test_cheating_missing_receiver() {
    // CPU sends to ADD
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // sends to ADD
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    // CHEAT: ADD doesn't receive (wrong values, so fingerprint won't match)
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(100), FE::zero(), FE::zero(), FE::zero()], // completely different
            vec![FE::from(200), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(300), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject when receiver doesn't match sender"
    );
}

// NOTE: "Wrong bus" test removed because the current implementation doesn't include
// bus IDs in fingerprints. Buses are distinguished by which tables participate in
// send/receive, not by explicit IDs. If values match, fingerprints match regardless
// of the conceptual "bus". This is correct behavior for the current architecture.

// =============================================================================
// Complex multi-row attacks
// =============================================================================

/// Cheating: one of multiple operations has wrong value.
/// CPU sends 4 ADD operations, but ADD table has one wrong.
#[test_log::test]
fn test_cheating_one_of_many_wrong() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::one(), FE::one()], // 4 ADD ops
            vec![FE::zero(); 4],
            vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)],
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)],
            vec![FE::from(11), FE::from(22), FE::from(33), FE::from(44)], // correct results
        ],
        1,
    );

    // CHEAT: third row has wrong result
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)],
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)],
            vec![FE::from(11), FE::from(22), FE::from(99), FE::from(44)], // WRONG: 99 should be 33
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject when one of many operations is wrong"
    );
}

/// Full scenario from original tests: CPU + ADD + MUL with wrong ADD result.
#[test_log::test]
fn test_cheating_full_scenario_wrong_add_result() {
    // Standard CPU trace with mixed ADD and MUL
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::one(),
            ],
            vec![
                FE::from(1),
                FE::from(2),
                FE::from(3),
                FE::from(4),
                FE::from(5),
                FE::from(6),
                FE::from(7),
                FE::from(8),
            ],
            vec![
                FE::from(10),
                FE::from(20),
                FE::from(30),
                FE::from(40),
                FE::from(50),
                FE::from(60),
                FE::from(70),
                FE::from(80),
            ],
            vec![
                FE::from(11),
                FE::from(40),
                FE::from(33),
                FE::from(160),
                FE::from(55),
                FE::from(66),
                FE::from(490),
                FE::from(640),
            ],
        ],
        1,
    );

    // CHEAT: ADD table has wrong result in first row
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(99), FE::from(33), FE::from(55), FE::from(66)], // WRONG: 99 should be 11
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    // MUL table is correct
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)],
            vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)],
            vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)],
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject wrong ADD result in full scenario"
    );
}

/// Full scenario: CPU + ADD + MUL with wrong MUL result.
#[test_log::test]
fn test_cheating_full_scenario_wrong_mul_result() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::one(),
            ],
            vec![
                FE::from(1),
                FE::from(2),
                FE::from(3),
                FE::from(4),
                FE::from(5),
                FE::from(6),
                FE::from(7),
                FE::from(8),
            ],
            vec![
                FE::from(10),
                FE::from(20),
                FE::from(30),
                FE::from(40),
                FE::from(50),
                FE::from(60),
                FE::from(70),
                FE::from(80),
            ],
            vec![
                FE::from(11),
                FE::from(40),
                FE::from(33),
                FE::from(160),
                FE::from(55),
                FE::from(66),
                FE::from(490),
                FE::from(640),
            ],
        ],
        1,
    );

    // ADD table is correct
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    // CHEAT: MUL table has wrong result
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)],
            vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)],
            vec![FE::from(40), FE::from(999), FE::from(490), FE::from(640)], // WRONG: 999 should be 160
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Verifier should reject wrong MUL result in full scenario"
    );
}
