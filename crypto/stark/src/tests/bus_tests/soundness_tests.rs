//! Soundness tests: invalid proofs are rejected.
//!
//! These tests verify that the verifier correctly rejects proofs that violate
//! the bus balance invariant.

use crate::constraints::builder::EmptyConstraints;
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::table::Table;
use crate::test_utils::multi_prove_ram;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Bus ID for packing mismatch tests (single bus)
const TEST_BUS: u64 = 0;

// =============================================================================
// Value manipulation
// =============================================================================

/// Wrong result value: CPU sends (1, 10, 11) but ADD claims (1, 10, 99).
#[test_log::test]
fn test_wrong_result_value() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(10), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(11), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(10), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(99), FE::zero(), FE::zero(), FE::zero()], // WRONG
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// The composition-poly part count is fixed by the AIR's max constraint degree,
/// not chosen by the prover. A proof advertising a different number of parts must
/// be rejected — otherwise a malicious prover could inflate the parts to widen the
/// composition polynomial's degree space and weaken the low-degree test.
#[test_log::test]
fn test_rejects_inflated_composition_part_count() {
    // All-padding traces: a valid, bus-balanced (Σ = 0) proof — the simplest valid case.
    let mut cpu_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 5], 1);
    let mut add_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 4], 1);
    let mut mul_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 4], 1);

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
    let mut multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    // The untampered proof verifies.
    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));

    // Tamper: inflate the first table's composition-poly part count.
    multi_proof.proofs[0]
        .composition_poly_parts_ood_evaluation
        .push(FieldElement::<E>::zero());

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "verifier must reject a composition part count that disagrees with the AIR degree bound"
    );
}

/// Off-by-one error: CPU sends (5, 3, 8) but ADD claims (5, 3, 9).
#[test_log::test]
fn test_off_by_one() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(9), FE::zero(), FE::zero(), FE::zero()], // WRONG: should be 8
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Swapped operands: CPU sends (5, 3, 8) but ADD claims (3, 5, 8).
#[test_log::test]
fn test_swapped_operands() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // SWAPPED
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // SWAPPED
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Single column wrong: CPU sends (5, 3, 8) but ADD claims (5, 4, 8).
#[test_log::test]
fn test_single_column_wrong() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

// =============================================================================
// Multiplicity manipulation
// =============================================================================

/// Over-reported multiplicity: CPU sends once, ADD claims multiplicity=2.
#[test_log::test]
fn test_over_report_multiplicity() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // WRONG: mult=2
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Under-reported multiplicity: CPU sends twice, ADD claims multiplicity=1.
#[test_log::test]
fn test_under_report_multiplicity() {
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

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // WRONG: mult=1, should be 2
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Zero multiplicity skip: CPU sends, ADD sets multiplicity=0 to skip.
#[test_log::test]
fn test_zero_multiplicity_skip() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()], // WRONG: mult=0
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

// =============================================================================
// Structural (missing sender/receiver)
// =============================================================================

/// Phantom receive: ADD receives something CPU never sent.
#[test_log::test]
fn test_phantom_receive() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // no ADD operations
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(999), FE::zero(), FE::zero(), FE::zero()], // phantom
            vec![FE::from(888), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(1887), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // WRONG: receiving phantom
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Missing receiver: CPU sends but ADD has completely different values.
#[test_log::test]
fn test_missing_receiver() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

// =============================================================================
// First-row boundary constraints
// =============================================================================

/// A proof where table_contribution has been tampered with is rejected.
///
/// The table_contribution (L) is used both for the bus balance check
/// (Σ L = 0 across all tables) and for the per-row circular constraint
/// offset (L/N). Corrupting it causes the circular transition constraint
/// to fail, since the committed trace was built with the honest L.
#[test_log::test]
fn test_tampered_table_contribution() {
    // Simple valid trace: CPU sends (5, 3, 8) to the ADD table.
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // add_flag
            vec![FE::zero(); 4],                                 // mul_flag
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // multiplicity = 1
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

    let mut multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    // Corrupt table_contribution in the ADD table's bus public inputs.
    // This changes the per-row offset L/N used in the circular constraint,
    // so the verifier's transition constraint evaluation will disagree with the
    // committed trace (which was built with the honest L).
    let add_proof = &mut multi_proof.proofs[1]; // proofs: [cpu=0, add=1, mul=2]
    let bus_inputs = add_proof
        .bus_public_inputs
        .as_mut()
        .expect("ADD table must have bus public inputs");
    bus_inputs.table_contribution += FieldElement::<E>::one();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Proof with corrupted table_contribution must be rejected"
    );
}

/// A proof where the acc column OOD evaluation is tampered with is rejected.
///
/// The circular transition constraint enforces the relationship between
/// consecutive acc column values at every step. Corrupting the acc column's
/// OOD evaluation breaks this constraint at the OOD point.
#[test_log::test]
fn test_tampered_acc_ood_evaluation() {
    // Simple valid trace: CPU sends (5, 3, 8) to the ADD table.
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // add_flag
            vec![FE::zero(); 4],                                 // mul_flag
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // multiplicity = 1
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

    let mut multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    // Corrupt the acc column OOD evaluation in the ADD table proof.
    // With batching + absorption, ADD has 4 main columns and 1 aux column
    // (the accumulated column — all interactions are absorbed).
    // The acc column is the last aux column.
    let num_main = 4usize;
    let acc_col_ood_idx = num_main; // first (and only) aux column
    let add_proof = &mut multi_proof.proofs[1]; // proofs: [cpu=0, add=1, mul=2]
    let corrupted = *add_proof.trace_ood_evaluations.get(0, acc_col_ood_idx) + FieldElement::one();
    add_proof
        .trace_ood_evaluations
        .set(0, acc_col_ood_idx, corrupted);

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Proof with corrupted acc OOD evaluation must be rejected"
    );
}

/// A proof whose OOD trace-evaluation table has the wrong shape is rejected.
///
/// The table's dimensions are a public function of the AIR (transition offsets
/// x step_size rows, main+aux columns), so the verifier derives the expected
/// shape from AIR metadata and refuses any proof whose table does not match --
/// a malicious prover cannot reshape it (e.g. drop a column) to dodge a check.
#[test_log::test]
fn test_malformed_ood_table_shape_rejected() {
    // Same valid trace as `test_tampered_acc_ood_evaluation`: CPU sends (5,3,8).
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // add_flag
            vec![FE::zero(); 4],                                 // mul_flag
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // multiplicity = 1
        ],
        1,
    );
    let mut mul_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 4], 1);

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

    let mut multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    // Drop one column from the ADD table's OOD evaluations while keeping the
    // table internally consistent (data length matches the new width), so the
    // rejection is the AIR-shape guard, not an out-of-bounds panic.
    let add_proof = &mut multi_proof.proofs[1];
    let old = &add_proof.trace_ood_evaluations;
    assert!(old.width >= 1, "OOD table must have at least one column");
    let new_width = old.width - 1;
    let mut new_data = Vec::with_capacity(new_width * old.height);
    for row in 0..old.height {
        let full = old.get_row(row);
        new_data.extend_from_slice(&full[..new_width]);
    }
    add_proof.trace_ood_evaluations = Table::new(new_data, new_width);

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Proof with a wrong-shaped OOD table must be rejected"
    );
}

// =============================================================================
// Invalid bus public inputs
// =============================================================================

/// A proof where bus_public_inputs is None for a LogUp AIR is rejected.
///
/// The verifier must reject proofs where an AIR declares bus interactions
/// (has_trace_interaction() = true) but the proof omits bus_public_inputs.
/// Without this check, a dishonest prover could bypass both the boundary
/// constraints and the bus balance check.
#[test_log::test]
fn test_missing_bus_public_inputs_rejected() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
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

    let mut multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    // Remove bus_public_inputs from the ADD table proof entirely.
    multi_proof.proofs[1].bus_public_inputs = None;

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Proof with missing bus_public_inputs must be rejected"
    );
}

/// A proof where a non-LogUp AIR has bus_public_inputs injected is rejected.
///
/// A dishonest prover could inject a compensating table_contribution into a
/// table that has no LogUp constraints, making the bus balance ΣL = 0 while
/// the actual lookup interactions don't balance. The verifier must reject
/// any proof that contains bus_public_inputs for an AIR without trace interaction.
#[test_log::test]
fn test_injected_bus_public_inputs_on_non_logup_air_rejected() {
    use crate::examples::dummy_air::{self, DummyAIR};
    use crate::lookup::BusPublicInputs;

    type DummyF = F; // GoldilocksField (same base and extension for DummyAIR)

    let trace_length = 16;
    let mut trace = dummy_air::dummy_trace::<DummyF>(trace_length);
    let proof_options = ProofOptions::default_test_options();
    let air = DummyAIR::new(&proof_options);

    let mut proof = Prover::<DummyF, DummyF, ()>::prove(
        &air,
        &mut trace,
        &(),
        &mut DefaultTranscript::<DummyF>::new(&[]),
    )
    .unwrap();

    // Inject fake bus_public_inputs into a non-LogUp proof.
    // DummyAIR has has_trace_interaction() = false, so this must be rejected.
    proof.bus_public_inputs = Some(BusPublicInputs {
        table_contribution: FieldElement::<DummyF>::from(42u64),
        #[cfg(feature = "debug-checks")]
        per_bus_sums: Default::default(),
        #[cfg(feature = "debug-checks")]
        per_bus_sender_sums: Default::default(),
        #[cfg(feature = "debug-checks")]
        per_bus_receiver_sums: Default::default(),
        #[cfg(feature = "debug-checks")]
        table_name: "FAKE".to_string(),
    });

    assert!(
        !Verifier::<DummyF, DummyF, ()>::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<DummyF>::new(&[])
        ),
        "Proof with injected bus_public_inputs on non-LogUp AIR must be rejected"
    );
}

/// A proof where table_contribution is zeroed out is rejected.
///
/// Setting table_contribution to zero changes the per-row offset L/N to zero,
/// which breaks the circular transition constraint since the committed trace
/// was built with the honest (non-zero) L.
#[test_log::test]
fn test_zeroed_table_contribution_rejected() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
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

    let mut multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    // Zero out table_contribution for the ADD table.
    let add_proof = &mut multi_proof.proofs[1];
    let bus_inputs = add_proof
        .bus_public_inputs
        .as_mut()
        .expect("ADD table must have bus public inputs");
    bus_inputs.table_contribution = FieldElement::<E>::zero();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Proof with zeroed table_contribution must be rejected"
    );
}

// =============================================================================
// Complex scenarios
// =============================================================================

/// One of many wrong: 4 ADD operations, but one has wrong result.
#[test_log::test]
fn test_one_of_many_wrong() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::one(), FE::one()], // 4 ADD ops
            vec![FE::zero(); 4],
            vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)],
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)],
            vec![FE::from(11), FE::from(22), FE::from(33), FE::from(44)],
        ],
        1,
    );

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Full scenario with wrong ADD result.
#[test_log::test]
fn test_full_scenario_wrong_add() {
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

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(99), FE::from(33), FE::from(55), FE::from(66)], // WRONG: 99 should be 11
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Bus label enforcement: Value consumed by wrong table is rejected.
///
/// CPU sends (2, 3, 5) to the ADD bus (add_flag=1), but MUL table tries to receive it.
/// ADD table receives nothing.
///
/// With bus labels in the fingerprint, verification correctly FAILS because:
/// - ADD sender fingerprint includes ADD_BUS (0) as first element
/// - MUL receiver fingerprint includes MUL_BUS (1) as first element
/// - Different fingerprints → bus doesn't balance
#[test_log::test]
fn test_wrong_table_consumes_value_rejected() {
    // CPU sends (2, 3, 5) via ADD interaction (add_flag=1, mul_flag=0)
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // add_flag = 1
            vec![FE::zero(); 4],                                 // mul_flag = 0
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // a
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // c
        ],
        1,
    );

    // ADD table: empty (should receive but doesn't)
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // a (doesn't matter)
            vec![FE::zero(); 4], // b
            vec![FE::zero(); 4], // c
            vec![FE::zero(); 4], // multiplicity = 0 (receives NOTHING)
        ],
        1,
    );

    // MUL table: receives (2, 3, 5) - WRONG! This was sent to ADD bus!
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // a = 2
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b = 3
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // c = 5 (not 6=2*3, but that's not checked here)
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // multiplicity = 1 (CONSUMES IT!)
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    // Verification MUST fail: MUL table cannot consume values sent to ADD bus
    // because bus_id is included in the fingerprint
    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Bus labels should prevent wrong table from consuming values"
    );
}

// =============================================================================
// Packing mismatch (wrong formula)
// =============================================================================
// These tests verify that using different Packing interpretations on sender vs
// receiver causes verification to fail, even when raw column values match.
//
// The fingerprint formula depends on Packing:
// - Direct: z - (v0 + v1*α + v2*α² + ...)
// - Word2L: z - (v0 + 2^16*v1)     (combines columns, then uses α)
//
// If sender uses Direct and receiver uses Word2L, fingerprints differ!

/// Packing mismatch: sender uses Direct (3 separate elements), receiver uses
/// Word2L which combines first two columns differently.
///
/// Sender interprets columns [0,1,2] as: v0 + v1*α + v2*α²
/// Receiver with Word2L interprets [0,1] as: (v0 + 2^16*v1) then + v2*α
///
/// Even with matching values, the fingerprint formulas differ!
#[test_log::test]
fn test_packing_mismatch_direct_vs_word2l() {
    use crate::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };

    fn sender_air_direct(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Sender uses Direct: 2 separate elements
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::Direct.columns(&[1, 2]),
                ),
            ],
        };
        AirWithBuses::new(
            3,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    fn receiver_air_word2l(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Receiver uses Word2L: combines 2 columns into 1 element
                // Formula: (v0 + 2^16 * v1) - different from v0 + α*v1
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::Word2L.columns(&[1]),
                ),
            ],
        };
        AirWithBuses::new(
            3, // multiplicity, col1, col2
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    // Trace with identical raw values
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // multiplicity
            vec![FE::from(100), FE::zero(), FE::zero(), FE::zero()], // value 1
            vec![FE::from(200), FE::zero(), FE::zero(), FE::zero()], // value 2
        ],
        1,
    );

    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // multiplicity
            vec![FE::from(100), FE::zero(), FE::zero(), FE::zero()], // SAME value 1
            vec![FE::from(200), FE::zero(), FE::zero(), FE::zero()], // SAME value 2
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air_direct(&proof_options);
    let receiver = receiver_air_word2l(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    // Verification MUST fail because fingerprint formulas differ!
    // Sender: z - (100 + 200*α)
    // Receiver: z - (100 + 200*2^16) = z - (100 + 13107200)
    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Packing mismatch should cause verification failure"
    );
}

/// Different element count due to combining: sender uses 3 Direct elements,
/// receiver uses Word2L (combines 2→1) + Direct, producing only 2 bus elements.
///
/// Sender: [v0, v1, v2] → z - (v0 + α*v1 + α²*v2)
/// Receiver: [v0 + 2^16*v1, v2] → z - ((v0 + 2^16*v1) + α*v2)
///
/// Different number of α powers = different fingerprint!
#[test_log::test]
fn test_packing_mismatch_element_count() {
    use crate::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };

    fn sender_air_3_direct(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Sender uses 3 Direct elements: produces [col1, col2, col3]
                // Fingerprint: z - (col1 + α*col2 + α²*col3)
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::Direct.columns(&[1, 2, 3]),
                ),
            ],
        };
        AirWithBuses::new(
            4,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    fn receiver_air_word2l_direct(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Receiver uses Word2L (combines cols 1,2 into 1 element) + Direct (col 3)
                // Produces 2 bus elements: [(col1 + 2^16*col2), col3]
                // Fingerprint: z - ((col1 + 2^16*col2) + α*col3)
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    vec![
                        Packing::Word2L.columns(&[1])[0].clone(),
                        Packing::Direct.columns(&[3])[0].clone(),
                    ],
                ),
            ],
        };
        AirWithBuses::new(
            4,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(10), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(20), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(30), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(10), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(20), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(30), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air_3_direct(&proof_options);
    let receiver = receiver_air_word2l_direct(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    // Sender: z - (10 + 20*α + 30*α²)  [3 bus elements]
    // Receiver: z - ((10 + 20*65536) + 30*α) = z - (1310730 + 30*α)  [2 bus elements]
    // Different fingerprints!
    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Element count mismatch (3 vs 2 bus elements) should cause verification failure"
    );
}

/// Wrong shift constant: Word2L uses 2^16, if receiver used a different shift
/// the fingerprints would differ. We simulate this by having sender use
/// Word4L (which uses 2^8 shifts) while receiver expects values that were
/// combined with 2^16 shifts.
#[test_log::test]
fn test_packing_mismatch_shift_constant() {
    use crate::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };

    fn sender_air_word4l(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Word4L: b0 + 2^8*b1 + 2^16*b2 + 2^24*b3
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::Word4L.columns(&[1]),
                ),
            ],
        };
        AirWithBuses::new(
            5,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    fn receiver_air_dwordhl(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // DWordHL: [h0 + 2^16*h1, h2 + 2^16*h3] - different shift pattern!
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::DWordHL.columns(&[1]),
                ),
            ],
        };
        AirWithBuses::new(
            5,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    // Use small values so the different shift formulas give clearly different results
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()], // b0
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // b1
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b2
            vec![FE::from(4), FE::zero(), FE::zero(), FE::zero()], // b3
        ],
        1,
    );

    // Word4L: 1 + 2*256 + 3*65536 + 4*16777216 = 1 + 512 + 196608 + 67108864 = 67305985
    // DWordHL: [1 + 2*65536, 3 + 4*65536] = [131073, 262147] → different!

    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(4), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air_word4l(&proof_options);
    let receiver = receiver_air_dwordhl(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Different shift constants (Word4L vs DWordHL) should cause verification failure"
    );
}

/// Compound packing mismatch: sender uses DWordHHW, receiver uses DWordWHH.
/// Both consume 3 columns but interpret them differently:
/// - DWordHHW: [Word, Half, Half] → [w, h0 + 2^16*h1]
/// - DWordWHH: [Half, Half, Word] → [h0 + 2^16*h1, w]
///
/// Same raw values, different layout interpretation = different fingerprints.
#[test_log::test]
fn test_compound_mismatch_dwordhhw_vs_dwordwhh() {
    use crate::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };

    fn sender_air_dwordhhw(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // DWordHHW: [Word, Half, Half] at columns 1, 2, 3
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::DWordHHW.columns(&[1]),
                ),
            ],
        };
        AirWithBuses::new(
            4,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    fn receiver_air_dwordwhh(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // DWordWHH: [Half, Half, Word] at columns 1, 2, 3
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::DWordWHH.columns(&[1]),
                ),
            ],
        };
        AirWithBuses::new(
            4,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    // Trace with values that expose the layout difference
    // Column 1: 0x1000 (could be Word or Half)
    // Column 2: 0x2000 (could be Half)
    // Column 3: 0x3000 (could be Half or Word)
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x1000u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x2000u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x3000u64), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x1000u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x2000u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x3000u64), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    // Sender DWordHHW: [0x1000, 0x2000 + 0x3000*2^16] = [0x1000, 0x30002000]
    // Receiver DWordWHH: [0x1000 + 0x2000*2^16, 0x3000] = [0x20001000, 0x3000]
    // Completely different bus elements!

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air_dwordhhw(&proof_options);
    let receiver = receiver_air_dwordwhh(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Compound layout mismatch (DWordHHW vs DWordWHH) should cause verification failure"
    );
}

/// Compound vs primitive mismatch: sender uses DWordHL (compound), receiver
/// uses the equivalent 2× Word2L manually. Should PASS because they're equivalent!
#[test_log::test]
fn test_compound_equals_primitive_expansion() {
    use crate::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };

    fn sender_air_compound(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // DWordHL (compound): 4 halves at columns 1-4
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::DWordHL.columns(&[1]),
                ),
            ],
        };
        AirWithBuses::new(
            5,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    fn receiver_air_primitives(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Equivalent: 2× Word2L at columns 1-2 and 3-4
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::Column(0),
                    Packing::Word2L.columns(&[1, 3]),
                ),
            ],
        };
        AirWithBuses::new(
            5,
            auxiliary_trace_build_data,
            proof_options,
            1,
            EmptyConstraints,
        )
    }

    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x1111u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x2222u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x3333u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x4444u64), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x1111u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x2222u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x3333u64), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(0x4444u64), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air_compound(&proof_options);
    let receiver = receiver_air_primitives(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    // This should PASS - compound and primitive expansion are equivalent
    assert!(
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Compound should be equivalent to its primitive expansion"
    );
}

// =============================================================================
// Complex scenarios
// =============================================================================

/// Full scenario with wrong MUL result.
#[test_log::test]
fn test_full_scenario_wrong_mul() {
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

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}
