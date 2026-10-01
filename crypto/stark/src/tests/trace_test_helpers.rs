use crate::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::table::Table;
use crate::trace::{TraceTable, compute_frame_evaluation_points};
use crate::traits::AIR;
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use itertools::Itertools;
use math::field::{
    element::FieldElement,
    goldilocks::GoldilocksField,
    traits::{IsField, IsSubFieldOf},
};
use math::polynomial::Polynomial;

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Builds a valid 2-row `SimpleAddition` proof. Shared base for the
/// proof-tamper / rejection tests in `small_trace_tests` and
/// `row_pair_opening_tests`.
pub fn make_valid_simple_proof() -> (
    SimpleAdditionAIR<GoldilocksField>,
    crate::proof::stark::StarkProof<
        GoldilocksField,
        GoldilocksField,
        SimpleAdditionPublicInputs<GoldilocksField>,
    >,
) {
    let mut trace = simple_addition_trace::<GoldilocksField>(2);
    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = SimpleAdditionPublicInputs {
        a: FieldElement::from(1u64),
        b: FieldElement::from(2u64),
    };
    let air = SimpleAdditionAIR::<GoldilocksField>::new(&proof_options);
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .unwrap();
    (air, proof)
}

/// Reference Horner-based trace-evaluation used as an oracle by the prover
/// tests (`tests::prover_tests`). The production prover uses the LDE-based
/// barycentric `get_trace_evaluations_from_lde`; the two are
/// cross-checked in tests.
pub fn get_trace_evaluations<F, E>(
    main_trace_polys: &[Polynomial<FieldElement<F>>],
    aux_trace_polys: &[Polynomial<FieldElement<E>>],
    x: &FieldElement<E>,
    frame_offsets: &[usize],
    primitive_root: &FieldElement<F>,
    step_size: usize,
) -> Table<E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let evaluation_points =
        compute_frame_evaluation_points(x, frame_offsets, primitive_root, step_size);

    let main_evaluations = evaluation_points
        .iter()
        .map(|eval_point| {
            main_trace_polys
                .iter()
                .map(|main_poly| main_poly.evaluate(eval_point))
                .collect_vec()
        })
        .collect_vec();

    let aux_evaluations = evaluation_points
        .iter()
        .map(|eval_point| {
            aux_trace_polys
                .iter()
                .map(|aux_poly| aux_poly.evaluate(eval_point))
                .collect_vec()
        })
        .collect_vec();

    debug_assert_eq!(main_evaluations.len(), aux_evaluations.len());
    let mut main_evaluations = main_evaluations;
    let mut table_data = Vec::new();
    for (main_row, aux_row) in main_evaluations.iter_mut().zip(aux_evaluations) {
        main_row.extend_from_slice(&aux_row);
        table_data.extend_from_slice(main_row);
    }

    let main_trace_width = main_trace_polys.len();
    let aux_trace_width = aux_trace_polys.len();
    let table_width = main_trace_width + aux_trace_width;

    Table::new(table_data, table_width)
}

/// Test-only inherent impl: interpolate main trace columns into coefficient-form
/// polynomials. Used by prover_tests to build the Horner oracle.
impl<F, E> TraceTable<F, E>
where
    E: math::field::traits::IsField,
    F: IsSubFieldOf<E> + math::field::traits::IsFFTField,
{
    pub fn compute_trace_polys_main<S>(&self) -> Vec<Polynomial<FieldElement<F>>>
    where
        S: math::field::traits::IsFFTField + IsSubFieldOf<F>,
        F: Send + Sync,
        FieldElement<F>: Send + Sync,
    {
        let columns = self.columns_main();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft::<S>(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, math::fft::errors::FFTError>>()
            .expect("interpolate_fft failed in compute_trace_polys_main")
    }
}
