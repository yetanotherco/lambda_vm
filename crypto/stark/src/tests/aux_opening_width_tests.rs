//! Regression tests for the **main↔aux** term of the opening-width pin
//! (`verifier::trace_opening_widths_well_formed`); the precomputed↔main term and
//! the direct guard tests live in `tests::opening_width_tests`.
//!
//! Everything here is attacker-side — a hostile AIR *declaration* plus the trace
//! it implies. Unlike the precomputed instance, this one needs **no prover
//! change at all**: both sides absorb main-root-then-aux-root either way, so the
//! transcripts agree and an untouched prover produces the forgery.
//!
//! Mechanism
//! ---------
//! `verify_trace_openings` only Merkle-checks each of the three trace openings
//! against its own root; it never compared the aux opening width against
//! `air.num_auxiliary_rap_columns()`. The only width constraint was, in
//! `reconstruct_deep_composition_poly_evaluation_pair`:
//!
//!     num_base + num_aux == ood_width
//!
//! with `num_base` and `num_aux` read off the *prover-supplied openings*. The
//! **total** is pinned (`ood_blocks_well_formed`) but the **split** was not, so a
//! prover could commit the last `k` main columns in the AUXILIARY tree instead.
//!
//! Why that breaks LogUp: the main root is absorbed in round 1 phase A, the
//! shared LogUp challenges `z`/`alpha` are sampled immediately after, and the aux
//! root only in phase C. A column moved into the aux tree is therefore chosen
//! AFTER `z` and `alpha` are known, which collapses the multiset equality into a
//! single scalar equation the prover solves — no fingerprint collision needed.
//!
//! Vehicle: `LogReadOnlyRAP`, the in-repo continuous read-only-memory AIR whose
//! memory consistency rests entirely on LogUp. Honest layout (5, 1):
//! main = [a, v, a', v', m], aux = [s]. The attacker declares (4, 2):
//! main = [a, v, a', v'], aux = [m, s] — same global column order, same
//! constraints, same OOD width, so an unpinned verifier cannot tell. The
//! multiplicity column `m` is then picked after `z`/`alpha`. The moved column is
//! the multiplicity column on purpose: `traits.rs:182-188` documents the trailing
//! main columns of every preprocessed table as exactly the multiplicities.
//!
//! On stock `main` the two break tests below are ACCEPTED, including over the
//! rkyv wire through `multi_verify_archived` (the recursion-guest path). The
//! three controls are rejected on both, and discriminate the harness.

use std::marker::PhantomData;

use crate::constraints::{
    boundary::{BoundaryConstraint, BoundaryConstraints},
    builder::{
        ConstraintBuilder, ConstraintMeta, ConstraintSet, RowDomain, num_base_from_meta,
        run_transition_prover, run_transition_verifier,
    },
};
use crate::context::AirContext;
use crate::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use crate::proof::options::ProofOptions;
use crate::proof::view::StarkProofView;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::{AIR, TransitionEvaluationContext};
use crate::verifier::{IsStarkVerifier, Verifier};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type Ext = FieldElement<E>;

// =============================================================================
// The hostile constraint body: byte-for-byte `LogReadOnlyRAPConstraints` with
// the multiplicity column re-addressed from main[4] to aux[0] and the LogUp
// accumulator from aux[0] to aux[1]. Same values, same degrees, same meta.
// =============================================================================

pub struct SplitLogUpConstraints;

impl ConstraintSet<F, E> for SplitLogUpConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let a_sorted_0 = b.main(0, 2);
        let a_sorted_1 = b.main(1, 2);
        let v_sorted_0 = b.main(0, 3);
        let v_sorted_1 = b.main(1, 3);
        let one = b.one();
        let addr_diff = a_sorted_1 - a_sorted_0;

        b.emit_base_rows(
            0,
            RowDomain::except_last(1),
            addr_diff.clone() * (addr_diff.clone() - one.clone()),
        );
        b.emit_base_rows(
            1,
            RowDomain::except_last(1),
            (v_sorted_1 - v_sorted_0) * (addr_diff - one),
        );

        // ---- the only difference: s is aux[1], m is aux[0] (was main[4]) ----
        let s0 = b.aux(0, 1);
        let s1 = b.aux(1, 1);
        let z = b.challenge(0);
        let alpha = b.challenge(1);
        let a1 = b.main(1, 0);
        let v1 = b.main(1, 1);
        let a_sorted_1 = b.main(1, 2);
        let v_sorted_1 = b.main(1, 3);
        let m = b.aux(1, 0);
        let unsorted_term = -(a1 + v1 * alpha.clone()) + z.clone();
        let sorted_term = -(a_sorted_1 + v_sorted_1 * alpha) + z;
        b.emit_ext_rows(
            2,
            RowDomain::except_last(1),
            s0 * unsorted_term.clone() * sorted_term.clone() + m * unsorted_term.clone()
                - sorted_term.clone()
                - s1 * unsorted_term * sorted_term,
        );
    }
}

/// How the attacker fills the moved multiplicity column.
#[derive(Clone)]
pub enum MPlan {
    /// Honest multiplicities, merely committed in the wrong tree.
    Honest(Vec<Felt>),
    /// Honest multiplicities except index `idx`, which is SOLVED after `z`,
    /// `alpha` are known so the LogUp accumulator still lands on zero.
    Forge { base: Vec<Felt>, idx: usize },
}

pub struct SplitLogUpAIR {
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    plan: MPlan,
    /// Records the challenge-dependent multiplicity the attack solved for.
    pub forged_value: std::sync::Mutex<Option<Ext>>,
    /// Records the committed multiplicity column and the (z, alpha) it was
    /// solved against, so a test can replay the LogUp identity off-protocol.
    pub committed_m: std::sync::Mutex<Option<(Vec<Ext>, Ext, Ext)>>,
    phantom: PhantomData<(F, E)>,
}

impl SplitLogUpAIR {
    pub fn with_plan(proof_options: &ProofOptions, plan: MPlan) -> Self {
        let mut air = <Self as AIR>::new(proof_options);
        air.plan = plan;
        air
    }
}

impl AIR for SplitLogUpAIR {
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = LogReadOnlyPublicInputs<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = ConstraintSet::<F, E>::meta(&SplitLogUpConstraints);
        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 6,
            transition_offsets: vec![0, 1],
            num_transition_constraints: meta.len(),
        };
        Self {
            context,
            meta,
            plan: MPlan::Honest(Vec::new()),
            forged_value: std::sync::Mutex::new(None),
            committed_m: std::sync::Mutex::new(None),
            phantom: PhantomData,
        }
    }

    /// Runs AFTER the main root is absorbed and AFTER `z`, `alpha` are sampled.
    /// Fills aux[0] = m (the moved main column) and aux[1] = s.
    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<F, E>,
        challenges: &[Ext],
    ) -> Option<crate::lookup::BusPublicInputs<E>> {
        let cols = trace.columns_main();
        let (a, v, a_sorted, v_sorted) = (&cols[0], &cols[1], &cols[2], &cols[3]);
        let z = &challenges[0];
        let alpha = &challenges[1];
        let n = trace.num_rows();

        // u_i = 1/(z - (a_i + alpha*v_i)) ; t_i = 1/(z - (a'_i + alpha*v'_i))
        let u: Vec<Ext> = (0..n)
            .map(|i| (-(&a[i] + &v[i] * alpha) + z).inv().unwrap())
            .collect();
        let t: Vec<Ext> = (0..n)
            .map(|i| (-(&a_sorted[i] + &v_sorted[i] * alpha) + z).inv().unwrap())
            .collect();

        let m: Vec<Ext> = match &self.plan {
            MPlan::Honest(base) => base.iter().map(|x| x.to_extension()).collect(),
            MPlan::Forge { base, idx } => {
                let mut m: Vec<Ext> = base.iter().map(|x| x.to_extension()).collect();
                // Solve  sum_i m_i t_i = sum_i u_i  for m_idx.
                let mut rhs = u.iter().fold(Ext::zero(), |acc, x| acc + x);
                for i in 0..n {
                    if i != *idx {
                        rhs = rhs - &m[i] * &t[i];
                    }
                }
                let solved = rhs * t[*idx].inv().unwrap();
                *self.forged_value.lock().unwrap() = Some(solved);
                m[*idx] = solved;
                m
            }
        };

        *self.committed_m.lock().unwrap() = Some((m.clone(), *z, *alpha));

        let mut s = Vec::with_capacity(n);
        s.push(&m[0] * &t[0] - &u[0]);
        for i in 0..n - 1 {
            let next = &s[i] + &m[i + 1] * &t[i + 1] - &u[i + 1];
            s.push(next);
        }

        for i in 0..n {
            trace.set_aux(i, 0, m[i]);
            trace.set_aux(i, 1, s[i]);
        }
        None
    }

    /// The lie: 4 main columns, 2 aux columns (honest AIR says 5 and 1).
    fn trace_layout(&self) -> (usize, usize) {
        (4, 2)
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        rap_challenges: &[Ext],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<E>>,
        trace_length: usize,
    ) -> BoundaryConstraints<E> {
        let a0 = &pub_inputs.a0;
        let v0 = &pub_inputs.v0;
        let a_sorted_0 = &pub_inputs.a_sorted_0;
        let v_sorted_0 = &pub_inputs.v_sorted_0;
        let m0 = &pub_inputs.m0;
        let z = &rap_challenges[0];
        let alpha = &rap_challenges[1];

        let c1 = BoundaryConstraint::new_main(0, 0, a0.to_extension());
        let c2 = BoundaryConstraint::new_main(1, 0, v0.to_extension());
        let c3 = BoundaryConstraint::new_main(2, 0, a_sorted_0.to_extension());
        let c4 = BoundaryConstraint::new_main(3, 0, v_sorted_0.to_extension());
        // main[4] under the honest layout -> aux[0] here. Same GLOBAL index 4,
        // which is all the verifier's `main_trace_width + col` mapping sees.
        let c5 = BoundaryConstraint::new_aux(0, 0, m0.to_extension());

        let unsorted_term = (-(a0 + v0 * alpha) + z).inv().unwrap();
        let sorted_term = (-(a_sorted_0 + v_sorted_0 * alpha) + z).inv().unwrap();
        let p0_value = m0 * sorted_term - unsorted_term;

        let c_aux1 = BoundaryConstraint::new_aux(1, 0, p0_value);
        let c_aux2 = BoundaryConstraint::new_aux(1, trace_length - 1, Ext::zero());

        BoundaryConstraints::from_constraints(vec![c1, c2, c3, c4, c5, c_aux1, c_aux2])
    }

    fn constraints_meta(&self) -> &[ConstraintMeta] {
        &self.meta
    }

    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        base_evals: &mut [Felt],
        ext_evals: &mut [Ext],
    ) {
        run_transition_prover(
            &SplitLogUpConstraints,
            evaluation_context,
            base_evals,
            ext_evals,
        );
    }

    fn compute_transition(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
    ) -> Vec<Ext> {
        run_transition_verifier(
            &SplitLogUpConstraints,
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&ConstraintSet::<F, E>::meta(&SplitLogUpConstraints))
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length * 2
    }
}

// =============================================================================
// Fixtures
// =============================================================================

/// The exact data of the in-repo happy-path test
/// (`air_tests.rs::test_prove_read_only_memory_logup`): a continuous read-only
/// memory over addresses 1..=5.
fn honest_reads() -> (Vec<Felt>, Vec<Felt>) {
    (
        vec![3, 2, 2, 3, 4, 5, 1, 3]
            .into_iter()
            .map(Felt::from)
            .collect(),
        vec![30, 20, 20, 30, 40, 50, 10, 30]
            .into_iter()
            .map(Felt::from)
            .collect(),
    )
}

fn public_inputs() -> LogReadOnlyPublicInputs<F> {
    LogReadOnlyPublicInputs {
        a0: Felt::from(3),
        v0: Felt::from(30),
        a_sorted_0: Felt::from(1),
        v_sorted_0: Felt::from(10),
        m0: Felt::from(1),
    }
}

/// Split an honest 5-main-column LogUp trace into the attacker's shape:
/// 4 main columns + 2 (zeroed) aux columns. Returns the m column separately.
fn split_trace(addresses: Vec<Felt>, values: Vec<Felt>) -> (TraceTable<F, E>, Vec<Felt>) {
    let honest: TraceTable<F, E> = read_only_logup_trace(addresses, values);
    let cols = honest.columns_main();
    let n = cols[0].len();
    let m = cols[4].clone();
    let main = vec![
        cols[0].clone(),
        cols[1].clone(),
        cols[2].clone(),
        cols[3].clone(),
    ];
    let aux = vec![vec![Ext::zero(); n], vec![Ext::zero(); n]];
    (TraceTable::from_columns(main, aux, 1), m)
}

fn opts() -> ProofOptions {
    ProofOptions::default_test_options()
}

fn honest_air() -> LogReadOnlyRAP<F, E> {
    LogReadOnlyRAP::<F, E>::new(&opts())
}

fn tr() -> DefaultTranscript<E> {
    DefaultTranscript::<E>::new(&[])
}

// =============================================================================
// The two AIRs are indistinguishable to the verifier except for the split, so
// nothing but an explicit width pin can tell them apart.
// =============================================================================

#[test_log::test]
fn split_declaration_differs_from_the_honest_air_only_in_the_layout() {
    let h = honest_air();
    let a = SplitLogUpAIR::with_plan(&opts(), MPlan::Honest(Vec::new()));
    assert_eq!(
        format!("{:?}", h.constraints_meta()),
        format!("{:?}", a.constraints_meta()),
        "meta must match"
    );
    assert_eq!(h.context().trace_columns, a.context().trace_columns);
    assert_eq!(
        h.context().transition_offsets,
        a.context().transition_offsets
    );
    assert_eq!(
        h.num_transition_constraints(),
        a.num_transition_constraints()
    );
    assert_eq!(
        h.num_base_transition_constraints(),
        a.num_base_transition_constraints()
    );
    assert_eq!(
        h.trace_ood_next_row_columns(),
        a.trace_ood_next_row_columns()
    );
    assert_eq!(
        h.composition_poly_degree_bound(8),
        a.composition_poly_degree_bound(8)
    );
    assert_eq!(h.has_aux_trace(), a.has_aux_trace());
    assert_eq!(h.has_trace_interaction(), a.has_trace_interaction());
    // The ONLY divergence:
    assert_eq!(h.trace_layout(), (5, 1));
    assert_eq!(a.trace_layout(), (4, 2));
    assert_eq!(h.num_auxiliary_rap_columns(), 1);
    assert_eq!(a.num_auxiliary_rap_columns(), 2);
    println!("AUXSPLIT/0 honest layout (5,1)  attacker layout (4,2)  — everything else identical");
}

// =============================================================================
// The structural case: a proof whose aux opening is 2 columns wide, verified
// against an AIR that declares exactly 1. Accepted on stock `main`, and it needs
// no forgery at all — the trace here is honest.
// =============================================================================

#[test_log::test]
fn mis_split_aux_opening_is_rejected() {
    let (addr, val) = honest_reads();
    let (mut trace, m) = split_trace(addr, val);
    let pi = public_inputs();
    let attack_air = SplitLogUpAIR::with_plan(&opts(), MPlan::Honest(m));

    let proof = Prover::prove(&attack_air, &mut trace, &pi, &mut tr()).expect("prove");

    let aux_w = proof.deep_poly_openings[0]
        .aux_trace_polys
        .as_ref()
        .unwrap()
        .evaluations
        .len();
    let main_w = proof.deep_poly_openings[0]
        .main_trace_polys
        .evaluations
        .len();
    let h = honest_air();
    println!(
        "AUXSPLIT/1 opening widths: main={main_w} aux={aux_w}   AIR declares main={} aux={}",
        h.trace_layout().0,
        h.num_auxiliary_rap_columns()
    );
    assert_eq!(main_w, 4);
    assert_eq!(aux_w, 2);
    assert_ne!(aux_w, h.num_auxiliary_rap_columns());

    let accepted = Verifier::verify(&proof, &h, &mut tr());
    println!("AUXSPLIT/1 STOCK VERIFIER ACCEPTED MIS-SPLIT PROOF = {accepted}");
    assert!(
        !accepted,
        "the verifier must reject an aux opening wider than the AIR declares",
    );

    // Attribution: the rejection is the width pin's, not an incidental failure
    // elsewhere in verification. A "rejected" verdict is only evidence if it
    // comes from the guard under test.
    assert!(
        !Verifier::trace_opening_widths_well_formed(
            &h,
            StarkProofView::Owned(&proof),
            h.options().fri_number_of_queries,
        ),
        "the rejection above must come from the opening-width guard",
    );
}

// =============================================================================
// CONTROL — the harness discriminates: corrupting one value in the (wrongly
// wide) aux opening must be rejected. Passes on stock `main` too.
// =============================================================================

#[test_log::test]
fn corrupted_aux_opening_is_rejected() {
    let (addr, val) = honest_reads();
    let (mut trace, m) = split_trace(addr, val);
    let pi = public_inputs();
    let attack_air = SplitLogUpAIR::with_plan(&opts(), MPlan::Honest(m));
    let proof = Prover::prove(&attack_air, &mut trace, &pi, &mut tr()).expect("prove");

    let mut corrupted = proof.clone();
    corrupted.deep_poly_openings[0]
        .aux_trace_polys
        .as_mut()
        .unwrap()
        .evaluations[0] += Ext::one();
    let accepted = Verifier::verify(&corrupted, &honest_air(), &mut tr());
    println!("AUXSPLIT/CONTROL-A corrupted aux opening accepted = {accepted}");
    assert!(!accepted, "harness must discriminate");
}

// =============================================================================
// The break: a FALSE statement, accepted on stock `main`.
//
// The read column contains address 3 -> 30 (rows 0, 3) AND address 3 -> 999999
// (row 7). No single-valued read-only memory can serve both, so the LogUp
// multiset equality that this AIR exists to enforce is FALSE. With `m` moved
// into the aux tree the prover solves for m[1] AFTER seeing z, alpha, and the
// stock verifier accepts.
// =============================================================================

const BOGUS: u64 = 999999;

#[test_log::test]
fn false_memory_read_under_aux_split_is_rejected() {
    let (addr, mut val) = honest_reads();
    // Honest sorted memory table, built from the HONEST reads.
    let (_, honest_m) = split_trace(addr.clone(), val.clone());
    let honest_trace: TraceTable<F, E> = read_only_logup_trace(addr.clone(), val.clone());
    let sorted_a = honest_trace.columns_main()[2].clone();
    let sorted_v = honest_trace.columns_main()[3].clone();

    // The lie: read #7 (address 3) now claims value 999999.
    val[7] = Felt::from(BOGUS);

    // Sanity: the read multiset is now impossible for a single-valued memory.
    let mut same_addr_values: Vec<Felt> = Vec::new();
    for i in 0..addr.len() {
        if addr[i] == Felt::from(3) && !same_addr_values.contains(&val[i]) {
            same_addr_values.push(val[i]);
        }
    }
    println!(
        "AUXSPLIT/2 reads at address 3 claim {} distinct values: {same_addr_values:?}",
        same_addr_values.len()
    );
    assert!(
        same_addr_values.len() > 1,
        "the statement must be false: address 3 must carry two different values"
    );

    let n = addr.len();
    let main = vec![addr.clone(), val.clone(), sorted_a, sorted_v];
    let aux = vec![vec![Ext::zero(); n], vec![Ext::zero(); n]];
    let mut trace = TraceTable::<F, E>::from_columns(main, aux, 1);

    let pi = public_inputs();
    let attack_air = SplitLogUpAIR::with_plan(
        &opts(),
        MPlan::Forge {
            base: honest_m,
            idx: 1,
        },
    );
    let proof = Prover::prove(&attack_air, &mut trace, &pi, &mut tr()).expect("prove");

    let forged = attack_air.forged_value.lock().unwrap().unwrap();
    println!("AUXSPLIT/2 solved multiplicity m[1] (challenge-dependent) = {forged:?}");

    let accepted = Verifier::verify(&proof, &honest_air(), &mut tr());
    println!("AUXSPLIT/2 FALSE STATEMENT ACCEPTED BY STOCK VERIFIER = {accepted}");
    assert!(
        !accepted,
        "the verifier must reject a false statement carried by an aux mis-split",
    );

    // Attribution: the rejection is the width pin's, not an incidental failure
    // elsewhere in verification. A "rejected" verdict is only evidence if it
    // comes from the guard under test.
    assert!(
        !Verifier::trace_opening_widths_well_formed(
            &honest_air(),
            StarkProofView::Owned(&proof),
            honest_air().options().fri_number_of_queries,
        ),
        "the rejection above must come from the opening-width guard",
    );

    // -------- the same forgery over the WIRE: rkyv-serialize and verify
    // through `multi_verify_archived`, the read-in-place path the recursion
    // guest uses. Proves this is a transmissible proof, not an in-process
    // artefact, and that the archived path shares the hole. -----------------
    let multi = crate::proof::stark::MultiProof {
        proofs: vec![proof.clone()],
    };
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&multi).unwrap();
    println!("AUXSPLIT/2 serialized forged proof: {} bytes", bytes.len());
    let archived = rkyv::access::<
        crate::proof::stark::ArchivedMultiProof<F, E, LogReadOnlyPublicInputs<F>>,
        rkyv::rancor::Error,
    >(&bytes)
    .unwrap();
    let h = honest_air();
    let airs: Vec<
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>,
    > = vec![&h];
    let accepted_archived =
        Verifier::multi_verify_archived(&airs, archived, &mut tr(), &Ext::zero());
    println!("AUXSPLIT/2 ARCHIVED (wire) PATH ACCEPTED = {accepted_archived}");
    assert!(
        !accepted_archived,
        "the archived (recursion-guest) path must reject it too",
    );

    // -------- diagnostic: the accepted LogUp identity is NOT a multiset
    // equality, it holds only at the protocol's own (z, alpha). -------------
    let (m_committed, z, alpha) = attack_air.committed_m.lock().unwrap().clone().unwrap();
    let cols = trace.columns_main();
    let logup_residual = |z: &Ext, alpha: &Ext| -> Ext {
        let mut acc = Ext::zero();
        for i in 0..n {
            let u = (-(&cols[0][i] + &cols[1][i] * alpha) + z).inv().unwrap();
            let t = (-(&cols[2][i] + &cols[3][i] * alpha) + z).inv().unwrap();
            acc = acc + &m_committed[i] * t - u;
        }
        acc
    };
    let at_protocol = logup_residual(&z, &alpha);
    let z2 = z + Ext::from(7u64);
    let a2 = alpha + Ext::from(11u64);
    let at_fresh = logup_residual(&z2, &a2);
    println!("AUXSPLIT/2 LogUp residual at the protocol's (z,alpha) = {at_protocol:?}");
    println!("AUXSPLIT/2 LogUp residual at a FRESH (z',alpha')      = {at_fresh:?}");
    assert_eq!(
        at_protocol,
        Ext::zero(),
        "the attack balances the bus at the sampled challenges"
    );
    assert_ne!(
        at_fresh,
        Ext::zero(),
        "…but not as a rational identity: the two multisets genuinely differ"
    );
}

// =============================================================================
// CONTROL — the SAME false trace, proven WITHOUT the split (honest layout,
// honest multiplicities in the main tree). `m` is then bound before z/alpha and
// the bus cannot be made to balance: the proof must be rejected (or the prover
// must refuse). Shows the acceptance above comes from the split, not from a hole
// in the AIR. Passes on stock `main` too.
// =============================================================================

#[test_log::test]
fn same_false_read_without_the_split_is_rejected() {
    let (addr, mut val) = honest_reads();
    let honest_trace: TraceTable<F, E> = read_only_logup_trace(addr.clone(), val.clone());
    let sorted_a = honest_trace.columns_main()[2].clone();
    let sorted_v = honest_trace.columns_main()[3].clone();
    let m = honest_trace.columns_main()[4].clone();
    val[7] = Felt::from(BOGUS);

    let n = addr.len();
    let main = vec![addr, val, sorted_a, sorted_v, m];
    let aux = vec![vec![Ext::zero(); n]];
    let mut trace = TraceTable::<F, E>::from_columns(main, aux, 1);
    let pi = public_inputs();
    let h = honest_air();

    match Prover::prove(&h, &mut trace, &pi, &mut tr()) {
        Ok(proof) => {
            let accepted = Verifier::verify(&proof, &h, &mut tr());
            println!("AUXSPLIT/CONTROL-B no-split false trace accepted = {accepted}");
            assert!(!accepted, "control must be rejected");
        }
        Err(e) => println!("AUXSPLIT/CONTROL-B no-split prover refused: {e:?}"),
    }
}

// =============================================================================
// CONTROL — the split path is not a free pass: the SAME split declaration with
// HONEST multiplicities over the FALSE read column must be rejected. Only the
// challenge-dependent solve makes the forgery go through. Passes on stock `main`
// too.
// =============================================================================

#[test_log::test]
fn aux_split_without_the_challenge_solve_is_rejected() {
    let (addr, mut val) = honest_reads();
    let honest_trace: TraceTable<F, E> = read_only_logup_trace(addr.clone(), val.clone());
    let sorted_a = honest_trace.columns_main()[2].clone();
    let sorted_v = honest_trace.columns_main()[3].clone();
    let m = honest_trace.columns_main()[4].clone();
    val[7] = Felt::from(BOGUS);

    let n = addr.len();
    let main = vec![addr, val, sorted_a, sorted_v];
    let aux = vec![vec![Ext::zero(); n], vec![Ext::zero(); n]];
    let mut trace = TraceTable::<F, E>::from_columns(main, aux, 1);
    let pi = public_inputs();
    let attack_air = SplitLogUpAIR::with_plan(&opts(), MPlan::Honest(m));

    match Prover::prove(&attack_air, &mut trace, &pi, &mut tr()) {
        Ok(proof) => {
            let accepted = Verifier::verify(&proof, &honest_air(), &mut tr());
            println!("AUXSPLIT/CONTROL-C split + honest m over false reads accepted = {accepted}");
            assert!(!accepted, "control must be rejected");
        }
        Err(e) => println!("AUXSPLIT/CONTROL-C prover refused: {e:?}"),
    }
}

// =============================================================================
// NON-VACUITY — the honest `LogReadOnlyRAP` (layout (5, 1), aux width 1) must
// still verify. A pin that rejected every aux opening would satisfy every
// rejection test above.
// =============================================================================

#[test_log::test]
fn honest_logup_rap_proof_still_verifies() {
    let (addr, val) = honest_reads();
    let mut trace: TraceTable<F, E> = read_only_logup_trace(addr, val);
    let air = honest_air();
    let proof = Prover::prove(&air, &mut trace, &public_inputs(), &mut tr()).expect("prove");

    assert!(
        Verifier::verify(&proof, &air, &mut tr()),
        "an honest LogUp proof must verify",
    );
}
