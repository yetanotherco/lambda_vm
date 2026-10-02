//! STARK round trips under ZisK's Poseidon1 configuration (`P1StarkHash`:
//! arity-4 trees, ZisK's leaf hash, width-8 grinding), on the host, and the
//! tamper checks that make the verifier's new path walk load-bearing.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use stark::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use stark::proof::options::{FriMode, ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::prover::{GenericProver, IsStarkProver};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{GenericVerifier, IsStarkVerifier};

use crate::lfm::p1_commit::{P1StarkHash, P1Transcript};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;

fn options(blowup: u8, queries: usize, grinding: u8, format: ProofFormat) -> ProofOptions {
    ProofOptions {
        blowup_factor: blowup,
        fri_number_of_queries: queries,
        coset_offset: 3,
        grinding_factor: grinding,
        fri_final_poly_log_degree: 1,
        format,
    }
}

fn formats() -> [(&'static str, ProofFormat); 3] {
    [
        ("legacy", ProofFormat::LEGACY),
        (
            "dp",
            ProofFormat {
                fri_mode: FriMode::Dp,
                ..ProofFormat::LEGACY
            },
        ),
        (
            "one-row",
            ProofFormat {
                one_row: stark::proof::options::OneRowMode::Auto,
                ..ProofFormat::LEGACY
            },
        ),
    ]
}

fn prove_addition(
    rows: usize,
    o: &ProofOptions,
) -> (
    SimpleAdditionAIR<F>,
    StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) {
    let air = SimpleAdditionAIR::<F>::new(o);
    let pi = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    let mut trace = simple_addition_trace::<F>(rows);
    let proof = GenericProver::<F, F, _, P1StarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

fn verify_addition(
    air: &SimpleAdditionAIR<F>,
    proof: &StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, F, _, P1StarkHash>::verify(
        proof,
        air,
        &mut DefaultTranscript::<F>::new(&[]),
    )
}

fn prove_logup(
    rows: usize,
    o: &ProofOptions,
) -> (
    LogReadOnlyRAP<F, E>,
    StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) {
    let addr: Vec<Felt> = (0..rows).map(|i| Felt::from((i % 5) as u64 + 1)).collect();
    let val: Vec<Felt> = (0..rows)
        .map(|i| Felt::from(((i % 5) as u64 + 1) * 10))
        .collect();
    let mut trace: TraceTable<F, E> = read_only_logup_trace(addr, val);
    let cols = trace.columns_main();
    let pi = LogReadOnlyPublicInputs {
        a0: cols[0][0],
        v0: cols[1][0],
        a_sorted_0: cols[2][0],
        v_sorted_0: cols[3][0],
        m0: cols[4][0],
    };
    let air = LogReadOnlyRAP::<F, E>::new(o);
    let proof = GenericProver::<F, E, _, P1StarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut P1Transcript::new(),
    )
    .expect("proving must succeed");
    (air, proof)
}

fn verify_logup(
    air: &LogReadOnlyRAP<F, E>,
    proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, E, _, P1StarkHash>::verify(proof, air, &mut P1Transcript::new())
}

#[test]
fn p1_proofs_round_trip_at_every_tree_shape() {
    // Rows 2^8 and 2^9 at blowup 4: LDE depths 10 and 11, even and odd, so
    // both the full and the zero-padded top level are walked.
    for (name, format) in formats() {
        for rows in [256usize, 512] {
            let o = options(4, 9, 4, format);
            let (air, proof) = prove_addition(rows, &o);
            assert!(verify_addition(&air, &proof), "{name} rows {rows}");
        }
        let o = options(4, 9, 4, format);
        let (air, proof) = prove_logup(128, &o);
        assert!(verify_logup(&air, &proof), "{name} logup");
    }
}

#[test]
fn the_production_cap_policy_runs_uncapped_at_arity_4() {
    // `cap = auto` would cap a tree opened this often; at arity 4 both sides
    // run uncapped (`effective_cap_policy`), and the proof is the uncapped one.
    let capped = ProofFormat {
        merkle_cap: stark::proof::options::CapPolicy::Auto,
        ..ProofFormat::LEGACY
    };
    // No grinding: the two proofs must draw the same queries.
    let (air, proof) = prove_logup(128, &options(4, 30, 0, capped));
    assert!(verify_logup(&air, &proof));
    let (_, uncapped) = prove_logup(128, &options(4, 30, 0, ProofFormat::LEGACY));
    assert_eq!(
        proof.deep_poly_openings[0]
            .main_trace_polys
            .proof
            .merkle_path,
        uncapped.deep_poly_openings[0]
            .main_trace_polys
            .proof
            .merkle_path
    );
}

#[test]
fn a_tampered_p1_proof_is_rejected() {
    let o = options(4, 9, 4, ProofFormat::LEGACY);
    let (air, proof) = prove_logup(128, &o);
    assert!(verify_logup(&air, &proof));
    // 128 rows at blowup 4: 512 LDE rows in 256 row-pair leaves, four 4-ary
    // levels of three siblings (a binary path would be 8 nodes).
    assert_eq!(
        proof.deep_poly_openings[0]
            .main_trace_polys
            .proof
            .merkle_path
            .len(),
        12
    );
    // A trace opening's authentication path, one byte of one sibling.
    let mut bad = proof.clone();
    bad.deep_poly_openings[0].main_trace_polys.proof.merkle_path[0][0] ^= 1;
    assert!(!verify_logup(&air, &bad), "trace path sibling");
    // A FRI layer's path.
    let mut bad = proof.clone();
    bad.query_list[0].layers_auth_paths[0].merkle_path[1][5] ^= 1;
    assert!(!verify_logup(&air, &bad), "FRI path sibling");
    // A root.
    let mut bad = proof.clone();
    bad.fri_layers_merkle_roots[0][31] ^= 1;
    assert!(!verify_logup(&air, &bad), "FRI root");
    // The nonce.
    let mut bad = proof.clone();
    bad.nonce = bad.nonce.map(|n| n ^ 1);
    assert!(!verify_logup(&air, &bad), "nonce");
}
