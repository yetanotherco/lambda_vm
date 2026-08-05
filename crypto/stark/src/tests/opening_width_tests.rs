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
//! times. In particular, for a **non-preprocessed** AIR the verifier never
//! absorbs the precomputed root at all, so any column a prover declares
//! "precomputed" is bound by nothing and can be chosen *after* the round-2
//! challenges — enough to accept a demonstrably false statement, which
//! `false_statement_under_split_declaration_is_rejected` exercises end to end.
//!
//! The end-to-end tests here need a hostile prover, simulated by
//! [`TEST_ONLY_SKIP_PRECOMPUTED_ROOT_ABSORB`]: without it the same proof is
//! rejected for transcript divergence instead of for its split, which would
//! prove nothing. The `opening_widths_*` unit tests at the bottom need no such
//! switch — they call the guard directly on surgically re-split openings, and
//! cover the `evaluations_sym` slot, which is an independent attack surface.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

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
use crate::trace::TraceTable;
use crate::traits::{AIR, TransitionEvaluationContext};
use crate::verifier::{IsStarkVerifier, Verifier};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;

type F = GoldilocksField;
type Felt = FieldElement<GoldilocksField>;

const TRACE_LEN: usize = 16;

/// Attacker simulation, read only by the prover's round-1 commit loop under
/// `#[cfg(test)]`: skip absorbing the precomputed Merkle root. Nothing in the
/// protocol forces a hostile prover to absorb a root the verifier never reads —
/// a real attacker simply runs their own prover — but the in-repo prover is
/// honest, so the tests need this switch to build the proof an attacker would.
pub static TEST_ONLY_SKIP_PRECOMPUTED_ROOT_ABSORB: AtomicBool = AtomicBool::new(false);

/// The switch is process-global while the test harness runs tests in parallel,
/// so every test that proves anything takes this lock for its whole body.
static PROVER_MODE: Mutex<()> = Mutex::new(());

/// A failing test would otherwise poison the mutex and cascade into the rest.
fn lock_prover_mode() -> MutexGuard<'static, ()> {
    PROVER_MODE.lock().unwrap_or_else(|e| e.into_inner())
}

fn set_hostile_prover(hostile: bool) {
    TEST_ONLY_SKIP_PRECOMPUTED_ROOT_ABSORB.store(hostile, Ordering::SeqCst);
}

/// `Fibonacci2ColsAIR` with two knobs, both attacker-side:
///
/// * `split` declares trace column 0 as *precomputed*, so the prover commits it
///   in a separate Merkle tree. The verifier is handed the `split == false`
///   instance, whose `is_preprocessed()` branch never absorbs that root.
/// * `out`, when set, adds a public-output boundary on the last row of column 1.
///   Since `(a0, a1)` determine the whole trace, a wrong `out` makes the claimed
///   statement FALSE — that is what turns "an invalid witness is accepted" into
///   "a false statement is accepted".
pub struct FibonacciSplitAIR<F: IsFFTField> {
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    out: Option<FieldElement<F>>,
    split: bool,
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
        let mut air = Self::honest(proof_options, out);
        air.split = true;
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
            split: false,
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
        self.split
    }

    fn num_precomputed_columns(&self) -> usize {
        usize::from(self.split)
    }

    fn precomputed_commitment(&self) -> Commitment {
        self.precomputed_commitment
    }
}

type FibProof = StarkProof<F, F, FibonacciPublicInputs<F>>;

fn pub_inputs() -> FibonacciPublicInputs<F> {
    FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    }
}

/// The transcript state the verifier reaches right before sampling the round-2
/// challenge, for a single non-preprocessed table with no aux trace: absorb the
/// main root, then sample. Column 0 lives in the precomputed tree, so it does
/// not enter here — which is exactly the hole.
fn challenge_from_main_root(main_root: &Commitment) -> Felt {
    let mut transcript = DefaultTranscript::<F>::new(&[]);
    transcript.append_bytes(main_root);
    transcript.sample_field_element()
}

/// Build a trace that is NOT a valid 2-column Fibonacci trace but whose
/// beta-combined transition constraint `c0 + beta*c1` vanishes on every row.
///
/// `c0_i = A_{i+1} - A_i - B_i`, `c1_i = B_{i+1} - B_i - A_{i+1}`, so demanding
/// `c0_i + beta*c1_i = 0` gives
/// `A_{i+1} = [A_i + B_i(1 + beta) - beta*B_{i+1}] / (1 - beta)`.
fn forge_trace(b_col: &[Felt], a0: Felt, beta: &Felt) -> TraceTable<F, F> {
    let denom = (Felt::one() - beta).inv().expect("beta != 1");
    let mut a_col = vec![Felt::zero(); b_col.len()];
    a_col[0] = a0;
    for i in 0..b_col.len() - 1 {
        a_col[i + 1] =
            (&a_col[i] + &b_col[i] * (Felt::one() + beta) - beta * &b_col[i + 1]) * &denom;
    }
    TraceTable::from_columns_main(vec![a_col, b_col.to_vec()], 1)
}

/// How many rows of the trace violate the real Fibonacci constraints.
fn violations(trace: &TraceTable<F, F>) -> usize {
    let cols = trace.columns_main();
    let (a, b) = (&cols[0], &cols[1]);
    (0..a.len() - 1)
        .map(|i| {
            usize::from(&a[i + 1] - &a[i] - &b[i] != Felt::zero())
                + usize::from(&b[i + 1] - &b[i] - &a[i + 1] != Felt::zero())
        })
        .sum()
}

/// Attacker-chosen column 1. `b[0]` is pinned by the `a1` boundary; `b[last]` by
/// the public-output boundary when there is one; everything between is free.
fn attacker_b_column(a1: Felt, out: Option<Felt>) -> Vec<Felt> {
    let free = if out.is_some() {
        TRACE_LEN - 2
    } else {
        TRACE_LEN - 1
    };
    let mut b = vec![a1];
    let mut x = Felt::from(987654321u64);
    for _ in 0..free {
        x = &x * Felt::from(1442695040u64) + Felt::from(1013904223u64);
        b.push(x);
    }
    b.extend(out);
    b
}

/// Prove `trace` against the split declaration, with the hostile prover
/// (precomputed root never absorbed). Returns the proof a real attacker could
/// produce for the non-preprocessed AIR.
fn prove_with_split_declaration(
    proof_options: &ProofOptions,
    trace: &TraceTable<F, F>,
    out: Option<Felt>,
) -> FibProof {
    set_hostile_prover(true);
    let reference = FibonacciSplitAIR::<F>::honest(proof_options, out);
    let commitment = Prover::compute_precomputed_commitment_for_testing(trace, &reference, 1)
        .expect("precomputed commitment");
    let split_air = FibonacciSplitAIR::<F>::split(proof_options, out, commitment);
    let mut trace = trace.clone();
    Prover::prove(
        &split_air,
        &mut trace,
        &pub_inputs(),
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("prove under split declaration")
}

/// The adaptive forgery: learn the round-2 challenge from a probe run (round 1
/// binds only column 1), then solve for column 0.
fn forge_under_split_declaration(proof_options: &ProofOptions, out: Option<Felt>) -> FibProof {
    let b_col = attacker_b_column(pub_inputs().a1, out);
    let probe_trace =
        TraceTable::from_columns_main(vec![vec![Felt::zero(); TRACE_LEN], b_col.clone()], 1);
    let probe = prove_with_split_declaration(proof_options, &probe_trace, out);
    let beta = challenge_from_main_root(&probe.lde_trace_main_merkle_root);

    let forged = forge_trace(&b_col, pub_inputs().a0, &beta);
    assert!(
        violations(&forged) > 0,
        "test precondition: the forged trace must violate the real constraints",
    );
    let proof = prove_with_split_declaration(proof_options, &forged, out);
    assert_eq!(
        proof.lde_trace_main_merkle_root, probe.lde_trace_main_merkle_root,
        "test precondition: round 1 must not bind column 0, else the attack is not adaptive",
    );
    proof
}

/// The cheapest discriminating case, and not a forgery: an *honest* trace proven
/// under the split declaration. `origin/main` accepts it against the plain
/// non-preprocessed AIR, because nothing pins the precomputed/main split of the
/// openings. Verification must not depend on a width the prover chose.
#[test_log::test]
fn honest_trace_under_split_declaration_is_rejected() {
    let _prover_mode = lock_prover_mode();
    let proof_options = ProofOptions::default_test_options();
    let trace = compute_trace([Felt::one(), Felt::one()], TRACE_LEN);
    let proof = prove_with_split_declaration(&proof_options, &trace, None);

    let honest_air = FibonacciSplitAIR::<F>::honest(&proof_options, None);
    assert!(!honest_air.is_preprocessed());
    assert_eq!(
        proof.deep_poly_openings[0]
            .precomputed_trace_polys
            .as_ref()
            .expect("split proof opens a precomputed tree")
            .evaluations
            .len(),
        1,
        "test precondition: the proof declares one precomputed column",
    );

    assert!(
        !Verifier::verify(&proof, &honest_air, &mut DefaultTranscript::<F>::new(&[])),
        "Verifier must reject an opening split the AIR does not declare",
    );
}

/// The same split, now carrying an invalid witness chosen *after* the round-2
/// challenge — the reason the split matters.
#[test_log::test]
fn forged_trace_under_split_declaration_is_rejected() {
    let _prover_mode = lock_prover_mode();
    let proof_options = ProofOptions::default_test_options();
    let proof = forge_under_split_declaration(&proof_options, None);
    let honest_air = FibonacciSplitAIR::<F>::honest(&proof_options, None);

    assert!(
        !Verifier::verify(&proof, &honest_air, &mut DefaultTranscript::<F>::new(&[])),
        "Verifier must reject a trace forged against the sampled challenge",
    );
}

/// The strongest form: the claimed public output is unreachable from `(a0, a1)`,
/// so the statement has NO witness at all, and `origin/main` accepts it.
#[test_log::test]
fn false_statement_under_split_declaration_is_rejected() {
    let _prover_mode = lock_prover_mode();
    let proof_options = ProofOptions::default_test_options();
    let honest_trace = compute_trace([Felt::one(), Felt::one()], TRACE_LEN);
    let true_out = honest_trace.columns_main()[1][TRACE_LEN - 1];
    let claimed_out = Felt::from(999u64);
    assert_ne!(
        true_out, claimed_out,
        "test precondition: the claimed output must be unreachable, else the statement is true",
    );

    let honest_air = FibonacciSplitAIR::<F>::honest(&proof_options, Some(claimed_out));
    let proof = forge_under_split_declaration(&proof_options, Some(claimed_out));

    assert!(
        !Verifier::verify(&proof, &honest_air, &mut DefaultTranscript::<F>::new(&[])),
        "Verifier must reject a proof of a false statement",
    );
}

/// The mirror shape: keep the re-split openings but drop the precomputed root,
/// so the presence guards see `None`. The reindexed columns still reach the DEEP
/// reconstruction, so the width check — not the root check — has to reject it.
#[test_log::test]
fn precomputed_openings_without_root_are_rejected() {
    let _prover_mode = lock_prover_mode();
    let proof_options = ProofOptions::default_test_options();
    let mut proof = forge_under_split_declaration(&proof_options, None);
    proof.lde_trace_precomputed_merkle_root = None;
    assert!(
        proof.deep_poly_openings[0]
            .precomputed_trace_polys
            .is_some()
    );

    let honest_air = FibonacciSplitAIR::<F>::honest(&proof_options, None);
    assert!(
        !Verifier::verify(&proof, &honest_air, &mut DefaultTranscript::<F>::new(&[])),
        "Verifier must reject precomputed openings the AIR does not declare, root or no root",
    );
}

/// Non-vacuity, and the completeness case that matters: a table that genuinely
/// IS preprocessed has `num_precomputed_columns() > 0`, and its proof — with the
/// honest prover, verified against the same preprocessed AIR — must still be
/// accepted. A guard that rejected every split would pass every test above.
#[test_log::test]
fn honest_preprocessed_proof_still_verifies() {
    let _prover_mode = lock_prover_mode();
    set_hostile_prover(false);
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
    let _prover_mode = lock_prover_mode();
    set_hostile_prover(false);
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
    let _prover_mode = lock_prover_mode();
    set_hostile_prover(false);
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
