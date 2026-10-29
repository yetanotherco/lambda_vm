//! Commitment-time answers on the card: the answers a bound prove takes off
//! the resident main LDE equal the host's barycentric evaluation of the same
//! columns, the proof verifies with them, and a false answer is refused. Box
//! only (`--features cuda`, ignored); it fails loudly when the card arm did not
//! run, so it cannot pass by falling back to the host:
//!
//! ```text
//! LAMBDA_VM_GPU_LDE_THRESHOLD=2 LAMBDA_VM_GPU_BARY_THRESHOLD=2 \
//!     cargo test -p stark --release --features cuda --lib \
//!     tests::commit_ood_device_tests -- --ignored --test-threads=1
//! ```

use std::sync::atomic::Ordering;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
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
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;
type EE = FieldElement<E>;

/// The CPU/ADD/MUL instance at `2^log_n` operations of each kind: the CPU
/// table sends every ADD then every MUL, and each receiving table holds its
/// triples once.
fn traces(log_n: u32) -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    let n = 1usize << log_n;
    let a: Vec<FE> = (0..n as u64).map(|i| FE::from(3 * i + 1)).collect();
    let b: Vec<FE> = (0..n as u64).map(|i| FE::from(7 * i + 2)).collect();
    let sum: Vec<FE> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let prod: Vec<FE> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    let ones = vec![FE::one(); n];
    let zeros = vec![FE::zero(); n];
    let cat = |x: &[FE], y: &[FE]| [x, y].concat();
    let cpu = TraceTable::from_columns_main(
        vec![
            cat(&ones, &zeros),
            cat(&zeros, &ones),
            cat(&a, &a),
            cat(&b, &b),
            cat(&sum, &prod),
        ],
        1,
    );
    let add = TraceTable::from_columns_main(vec![a.clone(), b.clone(), sum, ones.clone()], 1);
    let mul = TraceTable::from_columns_main(vec![a, b, prod, ones], 1);
    (cpu, add, mul)
}

fn options() -> ProofOptions {
    ProofOptions::default_test_options()
}

/// A bound proof, its answers, and the traces it proved.
type Proved = (MultiProof<F, E, ()>, Vec<Vec<EE>>, [TraceTable<F, E>; 3]);

fn prove(log_n: u32, faults: Faults) -> Proved {
    let (mut cpu, mut add, mut mul) = traces(log_n);
    let opts = options();
    let (cpu_air, add_air, mul_air) = (
        new_cpu_air_with_lookup(&opts),
        new_add_air_with_lookup(&opts),
        new_mul_air_with_lookup(&opts),
    );
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu, &()),
        (&add_air, &mut add, &()),
        (&mul_air, &mut mul, &()),
    ];
    let (proof, answers) = faults::with(faults, || {
        Prover::multi_prove_bound(
            pairs,
            &mut DefaultTranscript::<E>::new(&[]),
            ResidencyMode::Retain,
        )
    })
    .expect("the instance proves");
    (proof, answers, [cpu, add, mul])
}

fn verifies(proof: &MultiProof<F, E, ()>, answers: &[Vec<EE>]) -> bool {
    let opts = options();
    let (cpu_air, add_air, mul_air) = (
        new_cpu_air_with_lookup(&opts),
        new_add_air_with_lookup(&opts),
        new_mul_air_with_lookup(&opts),
    );
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    Verifier::multi_verify_views_bound(
        &airs,
        MultiProofView::Owned(proof),
        answers,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// ★ The card's answers are the host's, the proof verifies, and the card arm
/// ran for every table (`COMMIT OOD DEVICE` line: tables off the card).
#[test]
#[ignore = "box: needs a GPU"]
fn the_cards_answers_are_the_hosts_and_the_proof_verifies() {
    for log_n in [14u32, 16] {
        let before = crate::gpu_lde::COMMIT_OOD_GPU_CALLS.load(Ordering::Relaxed);
        let (proof, answers, traces) = prove(log_n, Faults::default());
        let on_card = crate::gpu_lde::COMMIT_OOD_GPU_CALLS.load(Ordering::Relaxed) - before;
        println!("COMMIT OOD DEVICE log_n {log_n}: {on_card} of 3 tables off the card");
        assert_eq!(on_card, 3, "every table's answers came off the card");
        // `z0` again, as the verifier draws it.
        let mut t = DefaultTranscript::<E>::new(&[]);
        for p in &proof.proofs {
            crypto::fiat_shamir::is_transcript::IsTranscript::append_bytes(
                &mut t,
                &p.lde_trace_main_merkle_root,
            );
        }
        let rows = 2usize << log_n;
        let z0 = crate::commit_ood::draw_point::<F, E>(
            &mut t,
            (
                rows,
                rows * options().blowup_factor as usize,
                options().coset_offset,
            ),
        );
        for (table, (trace, answers)) in traces.iter().zip(&answers).enumerate() {
            let (data, width) = trace.main_data_row_major();
            let host = crate::commit_ood::evaluate_trace_columns(data, width, 0..width, &z0)
                .expect("z0 is out of the domain");
            assert_eq!(*answers, host, "log_n {log_n} table {table}");
        }
        assert!(verifies(&proof, &answers), "log_n {log_n}");
        println!("COMMIT OOD DEVICE log_n {log_n}: answers equal the host's, proof verifies");
    }
}

/// A false answer carried into DEEP on the card's path is refused.
#[test]
#[ignore = "box: needs a GPU"]
fn a_false_answer_on_the_cards_path_is_refused() {
    let (proof, answers, _) = prove(
        14,
        Faults {
            false_answer: Some((0, 2)),
            without_deep_term: false,
            z0_meets_z: false,
        },
    );
    assert!(!verifies(&proof, &answers));
    println!("COMMIT OOD DEVICE false answer: refused");
}
