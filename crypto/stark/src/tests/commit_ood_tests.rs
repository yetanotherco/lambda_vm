//! Commitment-time out-of-domain answers on a multi-table proof
//! (`crate::commit_ood`), on the bus-balanced CPU/ADD/MUL instance of
//! [`super::residency_mode_tests`]: a bound proof verifies with its answers;
//! Phase A absorbs every answer between the roots and the shared LogUp
//! challenges; an answer altered, missing, extra or moved is refused; a false
//! answer the prover carries into its DEEP polynomial is refused; and with the
//! answers' DEEP quotients left out on both sides that same false answer is
//! accepted, so the quotients are what refuses it.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::commit_ood::faults::{self, Faults};
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::proof::view::MultiProofView;
use crate::prover::{IsStarkProver, Prover};
use crate::residency_mode::ResidencyMode;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type EE = FieldElement<E>;
type Answers = Vec<Vec<EE>>;

/// Blowup 4 at 40 queries and no grinding: a false answer's DEEP quotient is
/// far from every codeword, so FRI misses it with probability below 2^-80,
/// and the proof is deterministic.
fn options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 40,
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

/// The instance proved bound under `faults`, its answers beside it.
fn prove_bound(faults: Faults) -> (MultiProof<F, E, ()>, Answers) {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = super::residency_mode_tests::traces();
    let opts = options();
    let cpu_air = new_cpu_air_with_lookup(&opts);
    let add_air = new_add_air_with_lookup(&opts);
    let mul_air = new_mul_air_with_lookup(&opts);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    faults::with(faults, || {
        Prover::multi_prove_bound(
            pairs,
            &mut DefaultTranscript::<E>::new(&[]),
            ResidencyMode::Retain,
        )
    })
    .expect("the instance proves")
}

fn verify_bound(proof: &MultiProof<F, E, ()>, answers: &[Vec<EE>], faults: Faults) -> bool {
    let opts = options();
    let cpu_air = new_cpu_air_with_lookup(&opts);
    let add_air = new_add_air_with_lookup(&opts);
    let mul_air = new_mul_air_with_lookup(&opts);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    faults::with(faults, || {
        Verifier::multi_verify_views_bound(
            &airs,
            MultiProofView::Owned(proof),
            answers,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        )
    })
}

/// ★ Every table answers once per main column (none of these AIRs is
/// preprocessed), and the bound proof verifies with its answers — and only
/// with the bound verifier: the plain one replays another transcript.
#[test]
fn a_bound_proof_verifies_with_its_answers() {
    let (proof, answers) = prove_bound(Faults::default());
    let widths: Vec<usize> = answers.iter().map(Vec::len).collect();
    assert_eq!(widths, vec![5, 4, 4], "one answer per main column");
    assert!(verify_bound(&proof, &answers, Faults::default()));

    let opts = options();
    let cpu_air = new_cpu_air_with_lookup(&opts);
    let add_air = new_add_air_with_lookup(&opts);
    let mul_air = new_mul_air_with_lookup(&opts);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    assert!(
        !Verifier::multi_verify(
            &airs,
            &proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "a bound proof is not a plain one"
    );
}

/// The answers are `t_c(z0)`: each one equals its column's interpolant at the
/// `z0` Phase A draws, evaluated independently here by Lagrange over the
/// table's trace domain.
#[test]
fn every_answer_is_its_column_at_the_drawn_point() {
    let (proof, answers) = prove_bound(Faults::default());
    let mut t = DefaultTranscript::<E>::new(&[]);
    for p in &proof.proofs {
        t.append_bytes(&p.lde_trace_main_merkle_root);
    }
    let z0: EE = t.sample_z_ood_with_domain_params(
        8,
        8 * options().blowup_factor as usize,
        &FieldElement::<F>::from(options().coset_offset),
    );
    let (cpu, add, mul) = super::residency_mode_tests::traces();
    for (table, (trace, answers)) in [cpu, add, mul].iter().zip(&answers).enumerate() {
        let (data, width) = trace.main_data_row_major();
        let rows = data.len() / width;
        let omega = <F as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
            rows.trailing_zeros() as u64,
        )
        .unwrap();
        let points: Vec<FieldElement<F>> = (0..rows).map(|k| omega.pow(k)).collect();
        for (c, answer) in answers.iter().enumerate() {
            let mut value = EE::zero();
            for k in 0..rows {
                let mut basis = EE::one();
                for (j, xj) in points.iter().enumerate() {
                    if j != k {
                        let num = &z0 - &(*xj).to_extension::<E>();
                        let den = (&points[k] - xj).inv().unwrap();
                        basis = basis * num * den.to_extension::<E>();
                    }
                }
                value += basis * data[k * width + c].to_extension::<E>();
            }
            assert_eq!(*answer, value, "table {table} column {c}");
        }
    }
}

/// ★ Phase A spelled out: every root, `z0`, every answer, THEN the shared
/// LogUp challenges. Each answer moves the first challenge, and the same
/// draw with the answers left out (or absorbed after it) is another value —
/// the placement no round trip can see, since prover and verifier would
/// share it.
#[test]
fn phase_a_binds_every_answer_before_the_shared_challenges() {
    let (proof, answers) = prove_bound(Faults::default());
    let opts = options();
    let cpu_air = new_cpu_air_with_lookup(&opts);
    let add_air = new_add_air_with_lookup(&opts);
    let mul_air = new_mul_air_with_lookup(&opts);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    let challenges = |answers: &[Vec<EE>]| {
        Verifier::replay_phase_a_bound(
            &airs,
            MultiProofView::Owned(&proof),
            answers,
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .expect("the replay accepts the counts")
    };
    let drawn = challenges(&answers);

    let mut by_hand = DefaultTranscript::<E>::new(&[]);
    for p in &proof.proofs {
        by_hand.append_bytes(&p.lde_trace_main_merkle_root);
    }
    let lde = 8 * opts.blowup_factor as usize;
    let offset = FieldElement::<F>::from(opts.coset_offset);
    let _z0: EE = by_hand.sample_z_ood_with_domain_params(8, lde, &offset);
    for answer in answers.iter().flatten() {
        by_hand.append_field_element(answer);
    }
    let spelled: Vec<EE> = (0..2).map(|_| by_hand.sample_field_element()).collect();
    assert_eq!(drawn, spelled);

    for (t, table) in answers.iter().enumerate() {
        for i in 0..table.len() {
            let mut moved = answers.clone();
            moved[t][i] += EE::one();
            assert_ne!(
                challenges(&moved)[0],
                drawn[0],
                "table {t} answer {i} moves z"
            );
        }
    }
    let mut late = DefaultTranscript::<E>::new(&[]);
    for p in &proof.proofs {
        late.append_bytes(&p.lde_trace_main_merkle_root);
    }
    let _z0: EE = late.sample_z_ood_with_domain_params(8, lde, &offset);
    let late_z: EE = late.sample_field_element();
    assert_ne!(
        late_z, drawn[0],
        "answers absorbed after the draw bind nothing"
    );
}

/// An answer altered, a table's answers short or long, a table's answers
/// missing or extra, or two tables' answers swapped: each refused.
#[test]
fn an_altered_missing_extra_or_moved_answer_is_refused() {
    let (proof, answers) = prove_bound(Faults::default());
    let refused = |answers: Answers, what: &str| {
        assert!(
            !verify_bound(&proof, &answers, Faults::default()),
            "{what} must be refused"
        );
    };
    for (t, table) in answers.iter().enumerate() {
        for i in 0..table.len() {
            let mut altered = answers.clone();
            altered[t][i] += EE::one();
            refused(altered, &format!("table {t} answer {i} altered"));
        }
        let mut short = answers.clone();
        short[t].pop();
        refused(short, &format!("table {t} one answer short"));
        let mut long = answers.clone();
        long[t].push(EE::zero());
        refused(long, &format!("table {t} one answer long"));
    }
    refused(answers[..2].to_vec(), "a table's answers missing");
    let mut extra = answers.clone();
    extra.push(Vec::new());
    refused(extra, "an extra table of answers");
    let mut swapped = answers.clone();
    swapped.swap(1, 2);
    refused(swapped, "two tables' answers swapped");
    refused(Vec::new(), "no answers at all");
}

/// A false answer the prover absorbs AND carries into its DEEP polynomial (a
/// consistent proof of the false value) is refused: the DEEP quotient at `z0`
/// is not low degree.
#[test]
fn a_false_answer_its_deep_carries_is_refused() {
    for (table, index) in [(0, 0), (0, 4), (1, 2), (2, 3)] {
        let fault = Faults {
            false_answer: Some((table, index)),
            without_deep_term: false,
            z0_meets_z: false,
        };
        let (proof, answers) = prove_bound(fault);
        assert!(
            !verify_bound(&proof, &answers, Faults::default()),
            "a false answer at table {table} column {index} must be refused"
        );
    }
}

/// ★ Load-bearing: with the answers' DEEP quotients left out on both sides,
/// the same false answer is ACCEPTED. So the quotients are what refuses it in
/// [`a_false_answer_its_deep_carries_is_refused`], not some other check.
#[test]
fn the_deep_quotients_are_what_refuse_a_false_answer() {
    let fault = Faults {
        false_answer: Some((1, 2)),
        without_deep_term: true,
        z0_meets_z: false,
    };
    let (proof, answers) = prove_bound(fault);
    assert!(
        verify_bound(&proof, &answers, fault),
        "without the quotients nothing checks an answer"
    );
    assert!(
        !verify_bound(&proof, &answers, Faults::default()),
        "with them the false answer is refused"
    );
}

/// The faults are per thread and off by default: a bound proof proved and
/// verified with none set is the production path.
#[test]
fn no_fault_is_set_by_default() {
    assert_eq!(faults::current(), Faults::default());
    let (proof, answers) = prove_bound(Faults::default());
    assert!(verify_bound(&proof, &answers, Faults::default()));
}

/// `z0` meets `z·gʳ` for each OOD row and `z^P`, and no other point.
#[test]
fn the_point_check_finds_every_deep_point() {
    use math::field::traits::IsFFTField;
    let g = F::get_primitive_root_of_unity(3).expect("a root of order 8");
    let z = EE::new([
        FieldElement::from(7u64),
        FieldElement::from(11u64),
        FieldElement::from(13u64),
    ]);
    let meets = |z0: &EE| crate::commit_ood::meets_deep_point(z0, &z, &g, 2, 3);
    assert!(meets(&z), "z");
    assert!(meets(&(&g * &z)), "g·z");
    assert!(meets(&z.pow(3usize)), "z^P");
    assert!(
        !meets(&(&g * &(&g * &z))),
        "g²·z is no DEEP point at two rows"
    );
    assert!(!meets(&z.pow(2usize)), "z² is no DEEP point at three parts");
    assert!(!meets(&(&z + EE::one())), "an unrelated point");
}

/// A `z0` that meets a DEEP point is refused: the same honest proof passes,
/// and fails once the verifier's point check sees `z` in place of `z0`.
#[test]
fn a_z0_that_meets_a_deep_point_is_refused() {
    let (proof, answers) = prove_bound(Faults::default());
    assert!(verify_bound(&proof, &answers, Faults::default()));
    let fault = Faults {
        z0_meets_z: true,
        ..Faults::default()
    };
    assert!(
        !verify_bound(&proof, &answers, fault),
        "z0 = z must be refused"
    );
}
