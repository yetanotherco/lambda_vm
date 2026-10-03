//! LogUp arities wider than pairs (`LogUpPolicy::{K3, K4, Best}`), end to end
//! on the host: the AIR's layout and part count under each policy, the host
//! aux build, small proofs that verify, and the tampered proofs the verifier
//! must refuse.
//!
//! The tables here balance on their own: `p` sender/receiver pairs ordered
//! `[s_0 … s_{p−1}, r_0 … r_{p−1}]`, so every committed group mixes buses and
//! no term column is identically zero, while the whole table sums to zero.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::constraints::builder::{ConstraintBuilder, ConstraintSet, EmptyConstraints};
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder,
};
use crate::proof::options::{LogUpPolicy, ProofFormat, ProofOptions};
use crate::proof::stark::MultiProof;
use crate::test_utils::multi_prove_ram;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;
type FE3 = FieldElement<E>;
type Air<CS> = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), CS>;

const ROWS: usize = 64;

/// `pairs` balanced sender/receiver pairs plus `idle` senders. Pair `b` reads
/// its multiplicity from main column `2b` and its value from `2b + 1`; idle
/// sender `e` reads `2p + 2e` and `2p + 2e + 1` (its multiplicity column is
/// all zero in [`trace`], so the table stays balanced).
fn interactions(pairs: usize, idle: usize) -> Vec<BusInteraction> {
    let value = |c: usize| vec![BusValue::column(c)];
    let mut out: Vec<BusInteraction> = (0..pairs)
        .map(|b| {
            BusInteraction::sender(5 + b as u64, Multiplicity::Column(2 * b), value(2 * b + 1))
        })
        .collect();
    out.extend((0..pairs).map(|b| {
        BusInteraction::receiver(5 + b as u64, Multiplicity::Column(2 * b), value(2 * b + 1))
    }));
    out.extend((0..idle).map(|e| {
        let c = 2 * pairs + 2 * e;
        BusInteraction::sender(100 + e as u64, Multiplicity::Column(c), value(c + 1))
    }));
    out
}

/// Random values, small random multiplicities, all-zero idle multiplicities.
fn trace(pairs: usize, idle: usize, seed: u64) -> TraceTable<F, E> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    let mut columns = Vec::new();
    for _ in 0..pairs {
        columns.push((0..ROWS).map(|_| FE::from(next() % 7)).collect());
        columns.push((0..ROWS).map(|_| FE::from(next())).collect());
    }
    for _ in 0..idle {
        columns.push(vec![FE::zero(); ROWS]);
        columns.push((0..ROWS).map(|_| FE::from(next())).collect());
    }
    TraceTable::from_columns_main(columns, 1)
}

fn options(blowup: u8, logup: LogUpPolicy) -> ProofOptions {
    ProofOptions {
        blowup_factor: blowup,
        grinding_factor: 0,
        fri_final_poly_log_degree: 2,
        format: ProofFormat {
            logup,
            ..ProofFormat::DEFAULT
        },
        ..ProofOptions::default_test_options()
    }
}

fn air(pairs: usize, idle: usize, opts: &ProofOptions) -> Air<EmptyConstraints> {
    Air::new(
        2 * (pairs + idle),
        AuxiliaryTraceBuildData {
            interactions: interactions(pairs, idle),
        },
        opts,
        1,
        EmptyConstraints,
    )
}

fn num_parts(air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>) -> usize {
    air.composition_poly_degree_bound(ROWS) / ROWS
}

fn prove<CS: ConstraintSet<F, E>>(
    air: &Air<CS>,
    mut trace: TraceTable<F, E>,
) -> MultiProof<F, E, ()> {
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(air, &mut trace, &())];
    multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).expect("prove")
}

fn verifies<CS: ConstraintSet<F, E>>(air: &Air<CS>, proof: &MultiProof<F, E, ()>) -> bool {
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![air];
    Verifier::multi_verify(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

fn proof_bytes(proof: &MultiProof<F, E, ()>) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// ★ Under each policy at blowup 4 the AIR commits the layout the rule picks
/// (`⌈N/k⌉` aux columns, `max(base, k + 1) − 1` parts), and its proof
/// verifies — on every absorbed count `r ∈ 1..=4`.
#[test]
fn wide_layouts_prove_and_verify_on_the_host() {
    use LogUpPolicy::{Best, K3, K4, Pair};
    // (pairs, idle, policy, aux columns, parts)
    for (pairs, idle, policy, aux, parts) in [
        (6, 0, K4, 3, 4),    // N = 12: 2 groups of 4, r = 4
        (6, 1, K4, 4, 4),    // N = 13: r = 1
        (7, 0, K4, 4, 4),    // N = 14: r = 2
        (7, 1, K4, 4, 4),    // N = 15: r = 3
        (6, 0, K3, 4, 3),    // N = 12: 3 groups of 3, r = 3
        (6, 0, Best, 4, 3),  // N = 12: k3 and k4 tie at 7 columns, the smaller wins
        (10, 0, Best, 5, 4), // N = 20: k4 (5 + 4) beats k3 (7 + 3)
        (3, 0, K4, 3, 2),    // N = 6: k4 does not pay, pairs stay
        (6, 0, Pair, 6, 2),  // today
    ] {
        let opts = options(4, policy);
        let air = air(pairs, idle, &opts);
        let n = 2 * pairs + idle;
        assert_eq!(air.trace_layout().1, aux, "N={n} {policy}");
        assert_eq!(num_parts(&air), parts, "N={n} {policy}");
        let proof = prove(&air, trace(pairs, idle, n as u64));
        assert_eq!(
            proof.proofs[0].composition_poly_parts_ood_evaluation.len(),
            parts
        );
        assert!(
            verifies(&air, &proof),
            "N={n} {policy}: the proof must verify"
        );
    }
}

/// The production levers around a k = 4 table: the Merkle cap, the DP fold
/// schedule and one-row openings (on, and the per-table rule) all carry four
/// composition parts.
#[test]
fn k4_proves_under_the_production_format_levers() {
    use crate::proof::options::{CapPolicy, FriMode, OneRowMode};
    for one_row in [OneRowMode::Off, OneRowMode::On, OneRowMode::Auto] {
        let mut opts = options(4, LogUpPolicy::K4);
        opts.format = ProofFormat {
            merkle_cap: CapPolicy::Auto,
            fri_mode: FriMode::Dp,
            one_row,
            ..opts.format
        };
        let air = air(7, 1, &opts);
        assert_eq!(num_parts(&air), 4);
        let proof = prove(&air, trace(7, 1, 21));
        assert!(verifies(&air, &proof), "{one_row}");
        let mut bad = proof.clone();
        bad.proofs[0].composition_poly_parts_ood_evaluation[3] += FE3::one();
        assert!(!verifies(&air, &bad), "{one_row}: a tampered part");
    }
}

/// Where the rule keeps pairs — a table too small for wider groups to pay, or
/// a blowup that cannot hold their degree — the proof is today's, byte for
/// byte.
#[test]
fn where_the_rule_keeps_pairs_the_bytes_are_todays() {
    for (pairs, idle, blowup) in [(3, 0, 4), (6, 1, 2), (7, 0, 2)] {
        let pair = options(blowup, LogUpPolicy::Pair);
        let n = 2 * pairs + idle;
        let today = proof_bytes(&prove(&air(pairs, idle, &pair), trace(pairs, idle, 7)));
        for policy in [LogUpPolicy::K3, LogUpPolicy::K4, LogUpPolicy::Best] {
            let wide = air(pairs, idle, &options(blowup, policy));
            assert_eq!(
                wide.trace_layout().1,
                n.div_ceil(2),
                "N={n} blowup {blowup}"
            );
            assert!(
                proof_bytes(&prove(&wide, trace(pairs, idle, 7))) == today,
                "N={n} blowup {blowup} {policy}: the pair layout must prove today's bytes"
            );
        }
    }
}

/// ★ T0e: the host aux build under k = 4 writes, per committed group, the
/// sum of its interactions' `s·m/f`, and an accumulator whose step is the
/// row's total over EVERY interaction (groups plus the absorbed tail) minus
/// `L/N`, from `acc[0] = 0` around the cycle; `L` is the table's total.
#[test]
fn the_host_aux_build_sums_every_interaction_k4() {
    let (pairs, idle) = (6, 1);
    let opts = options(4, LogUpPolicy::K4);
    let air = air(pairs, idle, &opts);
    let mut t = trace(pairs, idle, 99);
    // Give the idle sender multiplicities, so L ≠ 0.
    for row in 0..ROWS {
        t.set_main(row, 2 * pairs, FE::from(row as u64 % 3));
    }
    let main: Vec<Vec<FE>> = t.columns_main();
    let challenges = vec![FE3::from(0x1234_5678u64), FE3::from(0x9abc_def0u64)];
    let bus = air.build_auxiliary_trace(&mut t, &challenges).expect("aux");
    let its = interactions(pairs, idle);
    let frac = |it: &BusInteraction, row: usize| -> FE3 {
        let (m_col, v_col) = match (&it.multiplicity, &it.values[0]) {
            (Multiplicity::Column(m), BusValue::Linear(terms)) => match terms[0] {
                crate::lookup::LinearTerm::Column { column, .. } => (*m, column),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        let f = challenges[0]
            - FE3::from(it.bus_id)
            - challenges[1] * main[v_col][row].to_extension::<E>();
        let q = main[m_col][row].to_extension::<E>() * f.inv().unwrap();
        if it.is_sender { q } else { -q }
    };
    let groups = 3; // N = 13 at k = 4: 3 groups, 1 absorbed
    assert_eq!(air.trace_layout().1, groups + 1);
    let row_totals: Vec<FE3> = (0..ROWS)
        .map(|row| {
            its.iter()
                .map(|it| frac(it, row))
                .fold(FE3::zero(), |a, b| a + b)
        })
        .collect();
    for row in 0..ROWS {
        for g in 0..groups {
            let want = its[4 * g..4 * g + 4]
                .iter()
                .map(|it| frac(it, row))
                .fold(FE3::zero(), |a, b| a + b);
            assert_eq!(*t.get_aux(row, g), want, "row {row} group {g}");
        }
    }
    let total = row_totals.iter().fold(FE3::zero(), |a, b| a + *b);
    assert_eq!(bus.table_contribution, total, "L is the table's total");
    assert_ne!(total, FE3::zero(), "the idle sender makes L non-zero");
    let offset = total * FE3::from(ROWS as u64).inv().unwrap();
    assert_eq!(*t.get_aux(0, groups), FE3::zero(), "acc[0] = 0");
    for (row, row_total) in row_totals.iter().enumerate() {
        let step = *t.get_aux((row + 1) % ROWS, groups) - *t.get_aux(row, groups);
        assert_eq!(step, *row_total - offset, "accumulator step at row {row}");
    }
}

/// A valid k = 4 proof (N = 13) and its AIR, for the negatives below.
fn valid_k4() -> (Air<EmptyConstraints>, MultiProof<F, E, ()>) {
    let air = air(6, 1, &options(4, LogUpPolicy::K4));
    let proof = prove(&air, trace(6, 1, 13));
    assert!(verifies(&air, &proof), "the control proof verifies");
    (air, proof)
}

/// N1: one committed k = 4 term cell, at the out-of-domain point.
#[test]
fn a_tampered_k4_term_cell_is_rejected() {
    let (air, proof) = valid_k4();
    let main = air.trace_layout().0;
    for term in 0..air.trace_layout().1 {
        let mut bad = proof.clone();
        let ood = &mut bad.proofs[0].trace_ood_evaluations;
        let v = *ood.get(0, main + term) + FE3::one();
        ood.set(0, main + term, v);
        assert!(!verifies(&air, &bad), "aux column {term}");
    }
}

/// N2: the multiplicity of an absorbed interaction, in the trace. The prover
/// then commits a consistent aux trace, but the table no longer sums to zero
/// and the bus check refuses it. The control: an edit the bus does not see
/// (the idle sender's value, at multiplicity 0) still verifies.
#[test]
fn an_absorbed_multiplicity_unbalances_the_bus() {
    let (pairs, idle) = (6, 1);
    let air = air(pairs, idle, &options(4, LogUpPolicy::K4));
    // N = 13, k = 4: the one absorbed interaction is the idle sender.
    assert_eq!(air.trace_layout().1, 4);
    let mut unseen = trace(pairs, idle, 13);
    unseen.set_main(3, 2 * pairs + 1, FE::from(77u64));
    assert!(verifies(&air, &prove(&air, unseen)));
    let mut t = trace(pairs, idle, 13);
    t.set_main(5, 2 * pairs, FE::one());
    assert!(!verifies(&air, &prove(&air, t)));
}

/// N3: a k = 4 proof checked against the pair-layout AIR (the verifier's
/// format, not the proof's) is refused on shape.
#[test]
fn a_k4_proof_is_refused_under_the_pair_layout() {
    let (_, proof) = valid_k4();
    let pair = air(6, 1, &options(4, LogUpPolicy::Pair));
    assert_eq!(pair.trace_layout().1, 7);
    assert!(!verifies(&pair, &proof));
}

/// N4: a proof carrying 3 or 5 part OODs for a 4-part AIR.
#[test]
fn a_wrong_composition_part_count_is_refused() {
    let (air, proof) = valid_k4();
    assert_eq!(num_parts(&air), 4);
    let mut fewer = proof.clone();
    fewer.proofs[0].composition_poly_parts_ood_evaluation.pop();
    assert!(!verifies(&air, &fewer));
    let mut more = proof;
    more.proofs[0]
        .composition_poly_parts_ood_evaluation
        .push(FE3::zero());
    assert!(!verifies(&air, &more));
}

/// N5: one composition-part OOD value.
#[test]
fn a_tampered_composition_part_ood_is_refused() {
    let (air, proof) = valid_k4();
    for part in 0..4 {
        let mut bad = proof.clone();
        bad.proofs[0].composition_poly_parts_ood_evaluation[part] += FE3::one();
        assert!(!verifies(&air, &bad), "part {part}");
    }
}

/// A constraint set that declares degree 4 but emits nothing.
#[derive(Clone, Copy)]
struct DeclaresDegreeFour;

impl ConstraintSet<F, E> for DeclaresDegreeFour {
    fn eval<B: ConstraintBuilder<F, E>>(&self, _b: &mut B) {}
    fn max_degree(&self) -> usize {
        4
    }
}

/// ★ The verifier's own refusal of `parts > blowup`, and that it is
/// load-bearing. A table declaring degree 4 has 3 parts; at blowup 2 its
/// quotient is not determined by the LDE coset, so no proof at those options
/// is sound. Its constraints are in fact low-degree, so the proof is
/// otherwise consistent: only the new check refuses it, and the same table
/// at blowup 4 (3 parts ≤ 4) verifies.
#[test]
fn the_verifier_refuses_more_parts_than_the_blowup() {
    let mk = |blowup: u8| {
        Air::new(
            2,
            AuxiliaryTraceBuildData {
                interactions: interactions(1, 0),
            },
            &options(blowup, LogUpPolicy::Pair),
            1,
            DeclaresDegreeFour,
        )
    };
    let fits = mk(4);
    assert_eq!(num_parts(&fits), 3);
    assert!(verifies(&fits, &prove(&fits, trace(1, 0, 3))));
    let over = mk(2);
    assert_eq!(num_parts(&over), 3);
    let proof = prove(&over, trace(1, 0, 3));
    assert_eq!(
        proof.proofs[0].composition_poly_parts_ood_evaluation.len(),
        3
    );
    assert!(
        !verifies(&over, &proof),
        "3 composition parts at blowup 2 must be refused"
    );
}
