//! Tests for `multi_prove_unified` — batched commitment + single FRI.
//!
//! Uses the same synthetic multi-table AIRs as bus_tests/completeness_tests
//! (CPU dispatching to ADD and MUL tables via LogUp bus).

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
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Test unified prove + verify with ADD and MUL tables (same height = 4).
///
/// Uses the exact same trace data as completeness_tests but only batches
/// the same-height tables (ADD + MUL, both 4 rows).
#[test_log::test]
fn test_unified_prove_verify_same_height() {
    // ADD Trace (4 rows)
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    // MUL Trace (4 rows)
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
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    assert_eq!(add_trace.num_rows(), mul_trace.num_rows());

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let unified_proof = Prover::multi_prove_unified(
        air_trace_pairs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("multi_prove_unified failed");

    assert_eq!(unified_proof.table_data.len(), 2);
    assert_eq!(unified_proof.column_layout.len(), 2);

    // Verify the unified proof
    let airs_refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&add_air, &mul_air];
    let pub_inputs: Vec<&()> = vec![&(), &()];

    // Compute the actual bus balance from the proof (no CPU sender, so non-zero)
    let actual_bus_balance: FieldElement<E> = unified_proof
        .table_data
        .iter()
        .filter_map(|td| td.bus_public_inputs.as_ref())
        .map(|bpi| &bpi.table_contribution)
        .fold(FieldElement::zero(), |acc, c| acc + c);

    assert!(
        Verifier::multi_verify_unified(
            &airs_refs,
            &pub_inputs,
            &unified_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &actual_bus_balance,
        ),
        "Unified proof verification failed"
    );

    println!(
        "Unified prove+verify OK: {} tables, {} FRI layers, {} queries",
        unified_proof.table_data.len(),
        unified_proof.fri_layers_merkle_roots.len(),
        unified_proof.fri_query_list.len(),
    );
}

/// All-padding variant: every multiplicity is 0. Bus balances at zero.
/// This is the complete prove+verify end-to-end test for the unified path.
#[test_log::test]
fn test_unified_prove_verify_all_padding() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![vec![FE::zero(); 4]; 5],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![vec![FE::zero(); 4]; 4],
        1,
    );
    let mut mul_trace = TraceTable::from_columns_main(
        vec![vec![FE::zero(); 4]; 4],
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

    let unified_proof = Prover::multi_prove_unified(
        air_trace_pairs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("multi_prove_unified with all-padding should succeed");

    assert_eq!(unified_proof.table_data.len(), 3);

    // Verify: bus should balance at zero (all multiplicities are 0)
    let airs_refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    let pub_inputs: Vec<&()> = vec![&(), &(), &()];

    assert!(
        Verifier::multi_verify_unified(
            &airs_refs,
            &pub_inputs,
            &unified_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Unified all-padding proof verification failed"
    );

    println!("Unified prove+verify (all padding) OK: {} tables", unified_proof.table_data.len());
}
