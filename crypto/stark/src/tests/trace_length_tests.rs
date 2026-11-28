//! The host verifier's check of a proof's claimed trace length.
//!
//! A proof states its own `trace_length`, and the verifier derives the LDE
//! size, the blowup and the FRI fold layout from it. Without a bound a claim of
//! `2^62 + 2^j` rows at blowup 4 makes `trace_length · blowup` wrap (release
//! builds) to `2^(j+2)`, the honest LDE size of a `2^j`-row trace. The blowup
//! then reads as `0`, so its log is 64: nothing folds, and the final
//! polynomial carries as many coefficients as the codeword has points, so it
//! interpolates any DEEP function and the low-degree test is vacuous.
//!
//! `forgery::forged_fibonacci_proof` (release builds) builds that proof: an honest Fibonacci proof's
//! trees and openings, a false public input, a composition value that
//! satisfies the out-of-domain identity at the claimed length, and a final
//! polynomial interpolating the resulting DEEP codeword. The verifier must
//! refuse it on the trace length alone.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
    goldilocks::GoldilocksField,
};

use crate::domain::{checked_lde_length, try_new_verifier_domain};
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs};
use crate::proof::options::{CapPolicy, FriMode, ProofFormat, ProofOptions};
use crate::proof::stark::{MultiProof, StarkProof};
use crate::proof::view::{MultiProofView, StarkProofView};
use crate::prover::{IsStarkProver, Prover};
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type Felt = FieldElement<F>;
type Pi = FibonacciPublicInputs<F>;
type V = Verifier<F, F, Pi>;
type Proof = StarkProof<F, F, Pi>;

const ROWS_LOG: u32 = 3;
const ROWS: usize = 1 << ROWS_LOG;
const BLOWUP: usize = 4;
const COSET_OFFSET: u64 = 3;
/// Enough queries that the honest proof opens every one of the 16 row pairs
/// of its 32-point LDE (checked below), so the forger holds an opening for
/// any query index the forged transcript draws.
const QUERIES: usize = 200;

/// The claimed length: `4 · (2^62 + 2^3)` wraps to `2^5`, the honest LDE size.
const WRAPPED_ROWS: usize = (1 << 62) + ROWS;

fn options() -> ProofOptions {
    ProofOptions {
        blowup_factor: BLOWUP as u8,
        fri_number_of_queries: QUERIES,
        coset_offset: COSET_OFFSET,
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

fn honest_pi() -> Pi {
    Pi {
        a0: Felt::one(),
        a1: Felt::one(),
    }
}

fn transcript() -> DefaultTranscript<F> {
    DefaultTranscript::<F>::new(&[])
}

/// The verifier's transcript after round 1 of a single-table, no-LogUp proof.
fn round_1_transcript(proof: &Proof) -> DefaultTranscript<F> {
    let mut t = transcript();
    t.append_bytes(&proof.lde_trace_main_merkle_root);
    t
}

/// An honest proof of the 8-row Fibonacci run from (1, 1).
fn honest_proof(air: &FibonacciAIR<F>) -> Proof {
    let mut trace = simple_fibonacci::fibonacci_trace([Felt::one(), Felt::one()], ROWS);
    Prover::prove(air, &mut trace, &honest_pi(), &mut transcript()).expect("honest prove")
}

/// The forgery and the tests that need it. Release builds only: the forger
/// replays the verifier at the wrapped length, and a debug build's overflow
/// checks and debug assertions panic there.
#[cfg(not(debug_assertions))]
mod forgery {
    use super::*;
    use crate::domain::VerifierDomain;
    use crate::fri::fri_decommit::FriDecommitment;
    use crate::leaf_layout::LeafLayout;
    use crate::proof::stark::DeepPolynomialOpening;
    use crate::verifier::Challenges;
    use math::field::traits::IsFFTField;

    /// The false statement: a Fibonacci run starting from (5, 7), proved with the
    /// trace of the run from (1, 1).
    fn forged_pi() -> Pi {
        Pi {
            a0: Felt::from(5u64),
            a1: Felt::from(7u64),
        }
    }

    /// The domain the unbounded verifier derives from `trace_length`: the product
    /// with the blowup wraps, exactly as a release build computes it.
    fn wrapped_domain(trace_length: usize) -> VerifierDomain<F> {
        let lde_length = trace_length.wrapping_mul(BLOWUP);
        let root =
            |len: usize| F::get_primitive_root_of_unity(len.trailing_zeros() as u64).unwrap();
        VerifierDomain {
            root_order: trace_length.trailing_zeros(),
            trace_length,
            lde_length,
            trace_primitive_root: root(trace_length),
            lde_primitive_root: root(lde_length),
            coset_offset: Felt::from(COSET_OFFSET),
        }
    }

    fn replay(air: &FibonacciAIR<F>, proof: &Proof, domain: &VerifierDomain<F>) -> Challenges<F> {
        V::replay_rounds_after_round_1(
            air,
            StarkProofView::Owned(proof),
            &proof.public_inputs,
            domain,
            &mut round_1_transcript(proof),
            Vec::new(),
            &V::ood_layout(air),
        )
    }

    /// `Σ βᵢ (t(z) − vᵢ) / (z − g^{stepᵢ})`: the boundary half of the composition
    /// polynomial at `z` (step 2's sum), for public inputs `pi` at `trace_length`.
    fn boundary_at_z(
        air: &FibonacciAIR<F>,
        pi: &Pi,
        trace_length: usize,
        domain: &VerifierDomain<F>,
        ood_row: &[Felt],
        challenges: &Challenges<F>,
    ) -> Felt {
        air.boundary_constraints(pi, &[], None, trace_length)
            .constraints
            .iter()
            .zip(&challenges.boundary_coeffs)
            .map(|(c, beta)| {
                let den = (challenges.z - domain.trace_primitive_root.pow(c.step as u64))
                    .inv()
                    .unwrap();
                (ood_row[c.col] - c.value) * den * beta
            })
            .fold(Felt::zero(), |acc, x| acc + x)
    }

    /// The forged proof of [`forged_pi`], claiming [`WRAPPED_ROWS`] rows. Every
    /// tree root and every opening is the honest proof's.
    fn forged_fibonacci_proof(air: &FibonacciAIR<F>) -> Proof {
        let honest = honest_proof(air);
        assert!(
            V::verify(&honest, air, &mut transcript()),
            "setup: the honest proof verifies"
        );
        let honest_domain = wrapped_domain(ROWS);
        let domain = wrapped_domain(WRAPPED_ROWS);
        assert_eq!(
            domain.lde_length, honest_domain.lde_length,
            "setup: the claimed length's LDE wraps to the honest one"
        );
        assert_eq!(domain.root_order, honest_domain.root_order);
        let pairs = domain.lde_length / 2;

        // Every row pair's honest opening, by query index.
        let honest_iotas = replay(air, &honest, &honest_domain).iotas;
        let mut by_iota: Vec<Option<DeepPolynomialOpening<F, F>>> = vec![None; pairs];
        for (q, &iota) in honest_iotas.iter().enumerate() {
            by_iota[iota] = Some(honest.deep_poly_openings[q].clone());
        }
        let by_iota: Vec<_> = by_iota
            .into_iter()
            .map(|o| o.expect("setup: the honest proof opens every row pair"))
            .collect();

        let mut forged = honest.clone();
        forged.trace_length = WRAPPED_ROWS;
        forged.public_inputs = forged_pi();
        forged.fri_layers_merkle_roots.clear();
        forged.fri_final_poly_coeffs.clear();
        forged.nonce = None;

        // The composition value at z that the out-of-domain identity needs at
        // the claimed length and the false public inputs. One transition
        // constraint: its term at the honest length is the honest claim minus the
        // honest boundary sum, and only its zerofier depends on the length.
        let ch = replay(air, &forged, &domain);
        assert_eq!(
            ch.z,
            replay(air, &honest, &honest_domain).z,
            "setup: same z"
        );
        assert_eq!(air.constraints_meta().len(), 1);
        assert_eq!(forged.composition_poly_parts_ood_evaluation.len(), 1);
        let ood_row = forged.trace_ood_evaluations.get_row(0).to_vec();
        let zerofier = |n: usize| {
            crate::constraints::zerofier::evaluate_zerofier(
                &air.constraints_meta()[0],
                &ch.z,
                &domain.trace_primitive_root,
                n,
            )
        };
        let honest_claim = honest.composition_poly_parts_ood_evaluation[0];
        let transition_honest =
            honest_claim - boundary_at_z(air, &honest_pi(), ROWS, &domain, &ood_row, &ch);
        let transition_claimed =
            transition_honest * zerofier(WRAPPED_ROWS) * zerofier(ROWS).inv().unwrap();
        forged.composition_poly_parts_ood_evaluation[0] =
            boundary_at_z(air, &forged_pi(), WRAPPED_ROWS, &domain, &ood_row, &ch)
                + transition_claimed;

        // The DEEP codeword at every point under the forged challenges, from the
        // honest openings, and the final polynomial that interpolates it: as many
        // coefficients as points.
        let mut ch = replay(air, &forged, &domain);
        ch.iotas = (0..pairs).collect();
        let mut every_pair = forged.clone();
        every_pair.deep_poly_openings = by_iota.clone();
        let layout = V::ood_layout(air);
        let view = StarkProofView::Owned(&every_pair);
        let ood_full = layout.reconstruct_full(
            view.trace_ood_evaluations().row_major_data(),
            view.trace_ood_evaluations().width(),
            view.trace_ood_next_evaluations().row_major_data(),
        );
        let (deep, deep_sym) = V::reconstruct_deep_composition_poly_evaluations_for_layout(
            &ch,
            &domain,
            view,
            &ood_full,
            layout.next_row_cols(),
            layout.step_size(),
            LeafLayout::RowPair,
        )
        .expect("DEEP at every pair");
        let codeword: Vec<Felt> = deep
            .iter()
            .zip(&deep_sym)
            .flat_map(|(x, sym)| [*x, *sym])
            .collect();
        forged.fri_final_poly_coeffs = crate::fri::terminal::coeffs_from_terminal_codeword::<F, F>(
            &codeword,
            &Felt::from(COSET_OFFSET),
            domain.lde_length.trailing_zeros(),
        );

        // The query indices now drawn, each answered by the honest opening there;
        // no FRI layer to open.
        let iotas = replay(air, &forged, &domain).iotas;
        forged.deep_poly_openings = iotas.iter().map(|&i| by_iota[i].clone()).collect();
        forged.query_list = vec![
            FriDecommitment {
                layers_auth_paths: Vec::new(),
                layers_evaluations_sym: Vec::new(),
            };
            QUERIES
        ];
        forged
    }

    /// A proof of a false Fibonacci statement whose claimed trace length wraps
    /// the LDE size is refused. Without the length check a release build accepts
    /// it (the forger's replay itself trips a debug assertion, so the forgery is
    /// only built in release).
    ///
    /// ★ The control: with the check switched off for this one length
    /// (`domain::wrapped_length`, the product wrapped as the verifier computed it
    /// before), the same forged proof is ACCEPTED. So the refusal is the length
    /// check's, not some other check's.
    #[test]
    fn wrapped_trace_length_forgery_is_refused() {
        let air = FibonacciAIR::<F>::new(&options());
        let forged = forged_fibonacci_proof(&air);
        assert!(
            crate::domain::wrapped_length::with(WRAPPED_ROWS, || V::verify(
                &forged,
                &air,
                &mut transcript()
            )),
            "the control: with the length let through wrapped the forgery is accepted"
        );
        assert!(
            !V::verify(&forged, &air, &mut transcript()),
            "a proof claiming 2^62 + 2^3 rows (LDE wraps to 2^5) of a false statement was accepted"
        );
    }

    /// The check in `verify_rounds_2_to_4` alone (its domain through
    /// `try_new_verifier_domain`), reached directly past Phase A: the forged
    /// proof is refused there, and accepted there under the control.
    #[test]
    fn rounds_2_to_4_alone_refuse_the_wrapped_forgery() {
        let air = FibonacciAIR::<F>::new(&options());
        let forged = forged_fibonacci_proof(&air);
        let rounds = || {
            V::verify_rounds_2_to_4(
                &air,
                StarkProofView::Owned(&forged),
                &forged.public_inputs,
                &mut round_1_transcript(&forged),
                Vec::new(),
                None,
            )
        };
        assert!(
            crate::domain::wrapped_length::with(WRAPPED_ROWS, rounds),
            "the control: rounds 2 to 4 accept the forgery at the wrapped domain"
        );
        assert!(!rounds(), "rounds 2 to 4 accepted the wrapped forgery");
    }
}

/// The honest proof with only its claimed length changed: refused in every
/// build — out of range before any arithmetic on the length (a debug build
/// used to panic on the wrapping multiply instead), in range as a false
/// length.
#[test]
fn claimed_lengths_no_proof_can_have_are_refused() {
    let air = FibonacciAIR::<F>::new(&options());
    let honest = honest_proof(&air);
    assert!(V::verify(&honest, &air, &mut transcript()));
    for claimed in [
        0,
        ROWS - 1,
        ROWS + 1,
        3 * ROWS,
        WRAPPED_ROWS,
        1 << 31,
        1 << 62,
        1 << 63,
        usize::MAX,
        // In range, but not this proof's: refused as false, without a panic
        // or an allocation at the largest domains.
        1,
        2,
        4,
        16,
        1 << 29,
        1 << 30,
    ] {
        let mut proof = honest.clone();
        proof.trace_length = claimed;
        assert!(
            !V::verify(&proof, &air, &mut transcript()),
            "claimed trace length {claimed} was accepted"
        );
    }
}

/// The bound itself: a power of two whose LDE fits Goldilocks' 2^32 subgroup.
#[test]
fn checked_lde_length_bounds() {
    // The largest LDE: 2^32 points, at every blowup.
    assert_eq!(checked_lde_length::<F>(1 << 30, 4), Some(1 << 32));
    assert_eq!(checked_lde_length::<F>(1 << 31, 2), Some(1 << 32));
    assert_eq!(checked_lde_length::<F>(1 << 32, 1), Some(1 << 32));
    assert_eq!(checked_lde_length::<F>(1, 4), Some(4));
    // One past it.
    assert_eq!(checked_lde_length::<F>(1 << 31, 4), None);
    assert_eq!(checked_lde_length::<F>(1 << 32, 2), None);
    assert_eq!(checked_lde_length::<F>(1 << 33, 1), None);
    // Not a power of two.
    assert_eq!(checked_lde_length::<F>(0, 4), None);
    assert_eq!(checked_lde_length::<F>(12, 4), None);
    assert_eq!(checked_lde_length::<F>((1 << 30) - 1, 4), None);
    assert_eq!(checked_lde_length::<F>(WRAPPED_ROWS, 4), None);
    assert_eq!(checked_lde_length::<F>(8, 3), None);
    assert_eq!(checked_lde_length::<F>(8, 0), None);
    // Products that overflow `usize`.
    assert_eq!(checked_lde_length::<F>(1 << 62, 4), None);
    assert_eq!(checked_lde_length::<F>(1 << 63, 2), None);
}

/// The verifier's domain at `air`'s blowup (4): the largest claim gets its
/// 2^32-point LDE; one past it, and any wrapping claim, gets none.
#[test]
fn verifier_domain_refuses_out_of_range_lengths() {
    let air = FibonacciAIR::<F>::new(&options());
    let max = try_new_verifier_domain(&air, 1 << 30).expect("2^30 rows at blowup 4");
    assert_eq!(max.lde_length, 1 << 32);
    assert_eq!(max.trace_length, 1 << 30);
    assert!(try_new_verifier_domain(&air, 1 << 31).is_none());
    assert!(try_new_verifier_domain(&air, WRAPPED_ROWS).is_none());
    assert!(try_new_verifier_domain(&air, 12).is_none());
    assert!(try_new_verifier_domain(&air, 0).is_none());
}

type E3 = Degree3GoldilocksExtensionField;
type Air3 = dyn AIR<Field = F, FieldExtension = E3, PublicInputs = ()>;

/// Lengths that are out of range for a table of the multi-table example at
/// blowup 2 (its test options): non-powers of two, one doubling past the
/// largest LDE, a wrapping claim, and the top bit.
const OUT_OF_RANGE: [usize; 6] = [0, 24, 1 << 32, (1 << 63) + 4, 1 << 63, usize::MAX];

/// ★ Phase A's own check, in release: a claimed length out of range is
/// refused before Phase A absorbs anything, so the verifier's transcript is
/// still a fresh one when it returns. Phase A weakened back to `== 0` would
/// absorb the table's root first and leave the refusal to rounds 2 to 4.
#[test]
fn phase_a_refuses_an_out_of_range_length_before_it_absorbs_anything() {
    let air = FibonacciAIR::<F>::new(&options());
    let honest = honest_proof(&air);
    for claimed in [24, 1 << 31, WRAPPED_ROWS] {
        let mut proof = honest.clone();
        proof.trace_length = claimed;
        let mut t = transcript();
        assert!(
            !V::verify(&proof, &air, &mut t),
            "length {claimed} was accepted"
        );
        assert_eq!(
            t.state(),
            transcript().state(),
            "length {claimed}: Phase A absorbed before refusing"
        );
    }
    let mut t = transcript();
    assert!(V::verify(&honest, &air, &mut t), "the control");
    assert_ne!(
        t.state(),
        transcript().state(),
        "the control: an accepted proof's Phase A absorbs its roots"
    );
}

/// `verify_rounds_2_to_4` alone, past Phase A: the honest proof's rounds
/// verify at its own length and refuse out-of-range ones, with no panic.
#[test]
fn rounds_2_to_4_alone_refuse_out_of_range_lengths() {
    let air = FibonacciAIR::<F>::new(&options());
    let honest = honest_proof(&air);
    let rounds = |proof: &Proof| {
        V::verify_rounds_2_to_4(
            &air,
            StarkProofView::Owned(proof),
            &proof.public_inputs,
            &mut round_1_transcript(proof),
            Vec::new(),
            None,
        )
    };
    assert!(rounds(&honest), "the control: the honest rounds verify");
    for claimed in [0, 24, WRAPPED_ROWS, 1 << 31, usize::MAX] {
        let mut proof = honest.clone();
        proof.trace_length = claimed;
        assert!(!rounds(&proof), "rounds 2 to 4 accepted length {claimed}");
    }
}

/// `replay_phase_a_bound` alone: its challenges at the honest length, none at
/// lengths whose LDE size `commit_ood::point_domain`'s own `checked_mul`
/// would still take (24 rows, 2^31 rows).
#[test]
fn the_bound_phase_a_replay_refuses_out_of_range_lengths() {
    let air = FibonacciAIR::<F>::new(&options());
    let honest = honest_proof(&air);
    let airs: [&dyn AIR<Field = F, FieldExtension = F, PublicInputs = Pi>; 1] = [&air];
    let answers = vec![vec![
        Felt::zero();
        crate::commit_ood::answered_columns(&air).len()
    ]];
    let replay = |proof: &Proof| {
        V::replay_phase_a_bound(
            &airs,
            &[StarkProofView::Owned(proof)][..],
            &answers,
            &mut transcript(),
        )
    };
    assert!(replay(&honest).is_some(), "the control: the honest replay");
    for claimed in [24, 1 << 31, WRAPPED_ROWS] {
        let mut proof = honest.clone();
        proof.trace_length = claimed;
        assert!(replay(&proof).is_none(), "the replay took length {claimed}");
    }
}

/// The multi-table example (LogUp bus, three tables) proved, and its AIRs.
fn multi_table() -> MultiProof<F, E3, ()> {
    let (mut cpu, mut add, mut mul) = super::residency_mode_tests::traces();
    let opts = ProofOptions::default_test_options();
    let (cpu_air, add_air, mul_air) = (
        new_cpu_air_with_lookup(&opts),
        new_add_air_with_lookup(&opts),
        new_mul_air_with_lookup(&opts),
    );
    let pairs: Vec<(&Air3, _, _)> = vec![
        (&cpu_air, &mut cpu, &()),
        (&add_air, &mut add, &()),
        (&mul_air, &mut mul, &()),
    ];
    crate::test_utils::multi_prove_ram(pairs, &mut DefaultTranscript::<E3>::new(&[]))
        .expect("the instance proves")
}

/// ★ The archived (in-place, the recursion guest's) entry point, on a
/// multi-table proof with a bus: the honest archive verifies; each table's
/// length patched out of range before archiving is refused, no panic.
#[test]
fn the_archived_verifier_refuses_out_of_range_lengths() {
    let opts = ProofOptions::default_test_options();
    let (cpu_air, add_air, mul_air) = (
        new_cpu_air_with_lookup(&opts),
        new_add_air_with_lookup(&opts),
        new_mul_air_with_lookup(&opts),
    );
    let airs: Vec<&Air3> = vec![&cpu_air, &add_air, &mul_air];
    let honest = multi_table();
    let verify_archived = |proof: &MultiProof<F, E3, ()>| {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(proof).expect("archives");
        let archived = rkyv::access::<
            crate::proof::stark::ArchivedMultiProof<F, E3, ()>,
            rkyv::rancor::Error,
        >(&bytes)
        .expect("reads");
        Verifier::multi_verify_archived(
            &airs,
            archived,
            &mut DefaultTranscript::<E3>::new(&[]),
            &FieldElement::zero(),
        )
    };
    assert!(verify_archived(&honest), "the control: the honest archive");
    let owned = |proof: &MultiProof<F, E3, ()>| {
        Verifier::multi_verify_views(
            &airs,
            MultiProofView::Owned(proof),
            &mut DefaultTranscript::<E3>::new(&[]),
            &FieldElement::zero(),
        )
    };
    assert!(owned(&honest), "the control: the honest owned proof");
    for table in 0..honest.proofs.len() {
        for claimed in OUT_OF_RANGE {
            let mut proof = honest.clone();
            proof.proofs[table].trace_length = claimed;
            assert!(
                !verify_archived(&proof),
                "archived: table {table} at length {claimed} was accepted"
            );
            assert!(
                !owned(&proof),
                "owned: table {table} at length {claimed} was accepted"
            );
        }
    }
}

/// ★ The bound entry point (`multi_verify_views_bound`, the LFM proofs' and
/// the block tree top's): a bound proof verifies with its answers, and each
/// table's length patched out of range is refused, no panic.
#[test]
fn the_bound_verifier_refuses_out_of_range_lengths() {
    use super::commit_ood_tests::{prove_bound, verify_bound};
    use crate::commit_ood::faults::Faults;
    let (honest, answers) = prove_bound(Faults::default());
    assert!(
        verify_bound(&honest, &answers, Faults::default()),
        "the control: the honest bound proof"
    );
    for table in 0..honest.proofs.len() {
        for claimed in OUT_OF_RANGE {
            let mut proof = honest.clone();
            proof.proofs[table].trace_length = claimed;
            assert!(
                !verify_bound(&proof, &answers, Faults::default()),
                "bound: table {table} at length {claimed} was accepted"
            );
        }
    }
}

/// The production format (`cap=auto fri=dp`): the honest proof verifies and
/// out-of-range lengths are refused, no panic.
#[test]
fn the_production_format_refuses_out_of_range_lengths() {
    let opts = ProofOptions {
        format: ProofFormat {
            merkle_cap: CapPolicy::Auto,
            fri_mode: FriMode::Dp,
            ..ProofFormat::DEFAULT
        },
        ..options()
    };
    let air = FibonacciAIR::<F>::new(&opts);
    let honest = honest_proof(&air);
    assert!(
        V::verify(&honest, &air, &mut transcript()),
        "the control: the honest proof at the production format"
    );
    for claimed in [0, 24, WRAPPED_ROWS, 1 << 31, 1 << 63, usize::MAX, 16] {
        let mut proof = honest.clone();
        proof.trace_length = claimed;
        assert!(
            !V::verify(&proof, &air, &mut transcript()),
            "production format: length {claimed} was accepted"
        );
    }
}
