//! Negative tests for the trace-opening column split
//! (`verifier::trace_opening_widths_well_formed`).
//!
//! A query opening carries the trace row as three prover-supplied vectors —
//! `precomputed ‖ main` (base field) and `aux` (extension field) — which the
//! DEEP reconstruction consumes as one concatenated row. Only their *sum* used
//! to be pinned (against the AIR-pinned OOD width), and the Merkle leaf hash
//! pins neither split: `hash_data_from_slices` streams `evaluations ‖
//! evaluations_sym` with no length prefix and no separator.
//!
//! That mattered because the three trees are transcript-bound at different
//! times. This file covers the **precomputed↔main** term; the main↔aux term —
//! the LogUp break, and the instance with an executed false statement — lives in
//! `tests::aux_opening_width_tests`.
//!
//! Two layers, both free of any prover modification:
//!
//! * `precomputed_opening_narrower_than_the_air_declares_is_rejected` — end to
//!   end through `Verifier::verify`, accepted on stock `main`. The prover and
//!   the verifier's AIR disagree about how many columns the precomputed
//!   commitment pins, while both absorb the same constant, so the transcripts
//!   agree and the honest in-repo prover builds the proof.
//! * `opening_widths_*` — the guard called directly on surgically re-split
//!   openings. These reach what no end-to-end test can: the `evaluations_sym`
//!   slot (a separate prover-supplied vector the leaf hash does not pin apart
//!   from `evaluations`) and the "a non-preprocessed AIR must declare zero
//!   precomputed columns" direction, whose end-to-end form is masked by
//!   transcript divergence and so proves nothing on its own.

use std::marker::PhantomData;

use crate::config::Commitment;
use crate::constraints::{
    boundary::{BoundaryConstraint, BoundaryConstraints},
    builder::{
        ConstraintMeta, ConstraintSet, num_base_from_meta, run_transition_prover,
        run_transition_verifier,
    },
};
use crate::context::AirContext;
use crate::examples::fibonacci_2_columns::{Fibonacci2ColsConstraints, compute_trace};
use crate::examples::fibonacci_rap::{FibonacciRAP, FibonacciRAPPublicInputs, fibonacci_rap_trace};
use crate::examples::simple_fibonacci::FibonacciPublicInputs;
use crate::proof::options::ProofOptions;
use crate::proof::stark::StarkProof;
use crate::proof::view::StarkProofView;
use crate::prover::{IsStarkProver, Prover};
use crate::traits::{AIR, TransitionEvaluationContext};
use crate::verifier::{IsStarkVerifier, Verifier};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;

type F = GoldilocksField;
type Felt = FieldElement<GoldilocksField>;

const TRACE_LEN: usize = 16;

/// `Fibonacci2ColsAIR` with two declaration knobs:
///
/// * `precomputed_columns` — how many leading columns the AIR claims live in the
///   precomputed tree (0 = not preprocessed). Prover and verifier are handed
///   instances that disagree about this, which is the whole point.
/// * `out`, when set, adds a public-output boundary on the last row of column 1.
///   Since `(a0, a1)` determine the whole trace, a wrong `out` would make the
///   claimed statement FALSE.
pub struct FibonacciSplitAIR<F: IsFFTField> {
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    out: Option<FieldElement<F>>,
    precomputed_columns: usize,
    precomputed_commitment: Commitment,
    phantom: PhantomData<F>,
}

impl<F: IsFFTField + Send + Sync + 'static> FibonacciSplitAIR<F> {
    /// The AIR as the verifier sees it: plain, non-preprocessed.
    fn honest(proof_options: &ProofOptions, out: Option<FieldElement<F>>) -> Self {
        let mut air = <Self as AIR>::new(proof_options);
        air.out = out;
        air
    }

    /// The AIR the hostile prover proves against: same width, same constraints,
    /// same boundary constraints — only the precomputed declaration differs.
    fn split(
        proof_options: &ProofOptions,
        out: Option<FieldElement<F>>,
        commitment: Commitment,
    ) -> Self {
        Self::preprocessed_declaring(proof_options, out, 1, commitment)
    }

    /// A preprocessed declaration with an explicit precomputed-column count.
    /// Handing the verifier a different count than the prover used is how the
    /// hook-free test below reaches the precomputed term of the guard: both
    /// sides still absorb the same commitment, so the transcripts agree.
    fn preprocessed_declaring(
        proof_options: &ProofOptions,
        out: Option<FieldElement<F>>,
        precomputed_columns: usize,
        commitment: Commitment,
    ) -> Self {
        let mut air = Self::honest(proof_options, out);
        air.precomputed_columns = precomputed_columns;
        air.precomputed_commitment = commitment;
        air
    }
}

impl<F> AIR for FibonacciSplitAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = FibonacciPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = Fibonacci2ColsConstraints::<F>::default().meta();
        let context = AirContext {
            proof_options: proof_options.clone(),
            transition_offsets: vec![0, 1],
            num_transition_constraints: meta.len(),
            trace_columns: 2,
        };
        Self {
            context,
            meta,
            out: None,
            precomputed_columns: 0,
            precomputed_commitment: [0u8; 32],
            phantom: PhantomData,
        }
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::Field>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        let mut constraints = vec![
            BoundaryConstraint::new_main(0, 0, pub_inputs.a0.clone()),
            BoundaryConstraint::new_main(1, 0, pub_inputs.a1.clone()),
        ];
        if let Some(out) = &self.out {
            constraints.push(BoundaryConstraint::new_main(1, TRACE_LEN - 1, out.clone()));
        }
        BoundaryConstraints::from_constraints(constraints)
    }

    fn constraints_meta(&self) -> &[ConstraintMeta] {
        &self.meta
    }

    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        run_transition_prover(
            &Fibonacci2ColsConstraints::default(),
            evaluation_context,
            base_evals,
            ext_evals,
        );
    }

    fn compute_transition(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        run_transition_verifier(
            &Fibonacci2ColsConstraints::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&Fibonacci2ColsConstraints::<F>::default().meta())
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2, 0)
    }

    fn is_preprocessed(&self) -> bool {
        self.precomputed_columns > 0
    }

    fn num_precomputed_columns(&self) -> usize {
        self.precomputed_columns
    }

    fn precomputed_commitment(&self) -> Commitment {
        self.precomputed_commitment
    }
}

fn pub_inputs() -> FibonacciPublicInputs<F> {
    FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    }
}

/// Tripwire. Every break test in this file and in
/// `tests::aux_opening_width_tests` asserts a *rejection*, and a rejection is
/// only evidence if it comes from the width pin — a verifier that rejected
/// everything, or that rejected these proofs for some incidental reason, would
/// satisfy them just as well. A sibling PoC was once misread exactly that way,
/// off a worktree whose verifier was not the one being claimed about.
///
/// So: the guard must be *defined and called*, not merely present. Deleting the
/// call site while keeping the function — the plausible bad refactor — fails
/// here rather than silently turning the whole file green for the wrong reason.
/// The break tests additionally assert attribution behaviourally, by calling the
/// guard on the very proof they reject.
///
/// (The prosecution PoC pinned a hash of the whole verifier source. That is
/// right for a throwaway branch and wrong in-repo, where it would break on every
/// unrelated verifier edit.)
#[test_log::test]
fn precheck_the_width_pin_is_compiled_in() {
    let src = include_str!("../verifier.rs");
    assert!(
        src.contains("fn trace_opening_widths_well_formed("),
        "the opening-width guard is gone from the verifier compiled into this binary",
    );
    assert!(
        src.contains("Self::trace_opening_widths_well_formed("),
        "the opening-width guard is defined but never called: every rejection \
         asserted in this file would then be proving something else",
    );
}

/// The precomputed term, end to end and **hook-free**: the prover commits ONE
/// column in the precomputed tree; the verifier's AIR declares TWO. Both sides
/// absorb the same commitment (the AIR's constant is the tree the prover built),
/// so the transcripts agree and the honest in-repo prover produces the proof —
/// no attacker-side prover switch involved.
///
/// Stock `main` accepts it: the widths sum to the OOD width and the DEEP
/// reconstruction reads the same concatenated row either way. What the verifier
/// is wrong about is *which* columns the hardcoded commitment pins — it believes
/// two, and only one is in that tree, so the other is prover-supplied while the
/// verifier treats it as fixed.
///
/// For a *real* preprocessed table (bitwise, decode, keccak_rc) the round-1 root
/// equality would also catch this, since an honest constant is a root over
/// exactly `num_precomputed_columns()` columns and a narrower tree hashes
/// differently. That defence is incidental: nothing states the invariant and
/// nothing checks it, and it does not exist at all for a non-preprocessed AIR,
/// where the root is never absorbed and the same re-split lets a prover choose
/// trace columns after the round-2 challenge. This test pins the width itself,
/// which is the property the reconstruction actually depends on.
#[test_log::test]
fn precomputed_opening_narrower_than_the_air_declares_is_rejected() {
    let proof_options = ProofOptions::default_test_options();
    let mut trace = compute_trace([Felt::one(), Felt::one()], TRACE_LEN);
    let reference = FibonacciSplitAIR::<F>::honest(&proof_options, None);
    let commitment = Prover::compute_precomputed_commitment_for_testing(&trace, &reference, 1)
        .expect("precomputed commitment");

    // Prover: one precomputed column, one main column.
    let prover_air =
        FibonacciSplitAIR::<F>::preprocessed_declaring(&proof_options, None, 1, commitment);
    let proof = Prover::prove(
        &prover_air,
        &mut trace,
        &pub_inputs(),
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("prove");
    assert_eq!(
        proof.deep_poly_openings[0]
            .precomputed_trace_polys
            .as_ref()
            .expect("preprocessed proof opens a precomputed tree")
            .evaluations
            .len(),
        1,
        "test precondition: the proof serves one precomputed column",
    );

    // Verifier: same commitment constant, but the AIR declares two precomputed
    // columns — so the second is served from the main tree, not the pinned one.
    let verifier_air =
        FibonacciSplitAIR::<F>::preprocessed_declaring(&proof_options, None, 2, commitment);
    assert!(
        !Verifier::verify(&proof, &verifier_air, &mut DefaultTranscript::<F>::new(&[])),
        "Verifier must reject a precomputed opening narrower than the AIR declares",
    );
    // Attribution: the rejection is the width pin's, not an incidental failure
    // elsewhere in verification.
    assert!(
        !Verifier::trace_opening_widths_well_formed(
            &verifier_air,
            StarkProofView::Owned(&proof),
            verifier_air.options().fri_number_of_queries,
        ),
        "the rejection above must come from the opening-width guard",
    );
}

/// Non-vacuity, and the completeness case that matters: a table that genuinely
/// IS preprocessed has `num_precomputed_columns() > 0`, and its proof — with the
/// honest prover, verified against the same preprocessed AIR — must still be
/// accepted. A guard that rejected every split would pass every test above.
#[test_log::test]
fn honest_preprocessed_proof_still_verifies() {
    let proof_options = ProofOptions::default_test_options();
    let mut trace = compute_trace([Felt::one(), Felt::one()], TRACE_LEN);
    let reference = FibonacciSplitAIR::<F>::honest(&proof_options, None);
    let commitment = Prover::compute_precomputed_commitment_for_testing(&trace, &reference, 1)
        .expect("precomputed commitment");
    let split_air = FibonacciSplitAIR::<F>::split(&proof_options, None, commitment);

    let proof = Prover::prove(
        &split_air,
        &mut trace,
        &pub_inputs(),
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("prove");

    assert!(
        Verifier::verify(&proof, &split_air, &mut DefaultTranscript::<F>::new(&[])),
        "a genuinely preprocessed table must still verify",
    );
}

/// Non-vacuity for the plain path: the same AIR without any split declaration.
#[test_log::test]
fn honest_non_preprocessed_proof_still_verifies() {
    let proof_options = ProofOptions::default_test_options();
    let mut trace = compute_trace([Felt::one(), Felt::one()], TRACE_LEN);
    let out = trace.columns_main()[1][TRACE_LEN - 1];
    let air = FibonacciSplitAIR::<F>::honest(&proof_options, Some(out));

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs(),
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("prove");

    assert!(
        Verifier::verify(&proof, &air, &mut DefaultTranscript::<F>::new(&[])),
        "an honest proof of a true statement must verify",
    );
}

// ---------------------------------------------------------------------------
// Direct tests of the guard, on a RAP proof (2 main + 1 aux columns).
//
// These reach the cases no end-to-end test can: the `evaluations_sym` slot is a
// separate prover-supplied vector that the leaf hash does not pin apart from
// `evaluations` (`hash_data_from_slices` concatenates them), and the aux width
// has its own transcript-timing problem (the aux root is absorbed only after
// the shared LogUp challenges).
// ---------------------------------------------------------------------------

type RapProof = StarkProof<F, F, FibonacciRAPPublicInputs<F>>;

fn make_valid_rap_proof() -> (FibonacciRAP<F>, RapProof) {
    let mut trace = fibonacci_rap_trace([Felt::one(), Felt::one()], TRACE_LEN);
    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = FibonacciRAPPublicInputs {
        steps: TRACE_LEN,
        a0: Felt::one(),
        a1: Felt::one(),
    };
    let air = FibonacciRAP::<F>::new(&proof_options);
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("prove");
    (air, proof)
}

fn widths_well_formed(air: &FibonacciRAP<F>, proof: &RapProof) -> bool {
    Verifier::trace_opening_widths_well_formed(
        air,
        StarkProofView::Owned(proof),
        air.options().fri_number_of_queries,
    )
}

/// Baseline: the honest proof's split is the AIR's split.
#[test_log::test]
fn opening_widths_accept_an_honest_rap_proof() {
    let (air, proof) = make_valid_rap_proof();
    assert_eq!(air.trace_layout(), (2, 1));
    assert!(!air.is_preprocessed());
    assert!(
        widths_well_formed(&air, &proof),
        "the guard must accept an honest proof",
    );
}

/// Each of the three widths, in each of the two slots, must be pinned. Every
/// mutation below keeps the *total* column count reachable by the old sum check
/// out of scope — the point is that the individual terms are now checked.
#[test_log::test]
fn opening_widths_reject_every_mismatched_term() {
    let (air, proof) = make_valid_rap_proof();
    let extra = Felt::one();

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[0]
        .main_trace_polys
        .evaluations
        .push(extra);
    assert!(
        !widths_well_formed(&air, &tampered),
        "an over-wide main opening must be rejected",
    );

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[0]
        .main_trace_polys
        .evaluations
        .pop();
    assert!(
        !widths_well_formed(&air, &tampered),
        "an under-wide main opening must be rejected",
    );

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[0]
        .main_trace_polys
        .evaluations_sym
        .push(extra);
    assert!(
        !widths_well_formed(&air, &tampered),
        "an over-wide symmetric main opening must be rejected",
    );

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[0]
        .aux_trace_polys
        .as_mut()
        .expect("the RAP AIR has an aux trace")
        .evaluations
        .push(extra);
    assert!(
        !widths_well_formed(&air, &tampered),
        "an over-wide aux opening must be rejected",
    );

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[0]
        .aux_trace_polys
        .as_mut()
        .expect("the RAP AIR has an aux trace")
        .evaluations_sym
        .push(extra);
    assert!(
        !widths_well_formed(&air, &tampered),
        "an over-wide symmetric aux opening must be rejected",
    );

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[0].aux_trace_polys = None;
    assert!(
        !widths_well_formed(&air, &tampered),
        "a missing aux opening must be rejected when the AIR declares aux columns",
    );

    let mut tampered = proof.clone();
    let mut precomputed = tampered.deep_poly_openings[0].main_trace_polys.clone();
    precomputed.evaluations.truncate(1);
    precomputed.evaluations_sym.truncate(1);
    tampered.deep_poly_openings[0].precomputed_trace_polys = Some(precomputed);
    assert!(
        !widths_well_formed(&air, &tampered),
        "precomputed openings must be rejected for a non-preprocessed AIR",
    );
}

/// The guard covers every query the FRI phase will read, not just the first.
#[test_log::test]
fn opening_widths_are_checked_for_every_query() {
    let (air, proof) = make_valid_rap_proof();
    let last = air.options().fri_number_of_queries - 1;
    assert!(last > 0, "test precondition: more than one query");

    let mut tampered = proof.clone();
    tampered.deep_poly_openings[last]
        .main_trace_polys
        .evaluations
        .push(Felt::one());
    assert!(
        !widths_well_formed(&air, &tampered),
        "a mismatched split in the last query's opening must be rejected",
    );
}

/// Fewer openings than queries is rejected rather than indexed past the end.
#[test_log::test]
fn opening_widths_reject_a_truncated_opening_list() {
    let (air, proof) = make_valid_rap_proof();
    let mut tampered = proof.clone();
    tampered.deep_poly_openings.pop();
    assert!(
        !widths_well_formed(&air, &tampered),
        "an opening list shorter than the query count must be rejected",
    );
}
