//! A full STARK prove → verify under [`Blake3StarkHash`], and the FRI-layer
//! evidence that makes it work.
//!
//! The whole commitment path is named by one configuration: the prover builds
//! FRI layer trees with `H::Pair` and the verifier authenticates those openings
//! with `H::Batched`, which are one hash by [`StarkHash`]'s two-element
//! invariant. A configuration that broke that agreement would reject every
//! honest proof at its first FRI query, so these tests are what says it holds
//! for a hash other than the default.
//!
//! Everything here is `cfg(not(cuda))`, because [`Blake3StarkHash`] is: under
//! `cuda` the tree entries hash on the device with the keccak kernels and only
//! label the result, so there is no second configuration to name.
#![cfg(not(feature = "cuda"))]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::config::{Blake3StarkHash, KeccakStarkHash, StarkHash};
use crate::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::StarkProof;
use crate::prover::{GenericProver, IsStarkProver};
use crate::traits::AIR;
use crate::verifier::{GenericVerifier, IsStarkVerifier};

type F = GoldilocksField;
type FE = FieldElement<F>;
type PI = SimpleAdditionPublicInputs<F>;

type Prove<H> = GenericProver<F, F, PI, H>;
type Verify<H> = GenericVerifier<F, F, PI, H>;

/// 1024 rows: `trace_bits = 10` against the default `k = 7`, so FRI actually
/// folds and commits layers. A trace that terminates immediately would make
/// every assertion below vacuous, which is why the layer count is asserted.
const TRACE_ROWS: usize = 1024;

fn air_and_inputs() -> (SimpleAdditionAIR<F>, PI) {
    let proof_options = ProofOptions::default_test_options();
    let air = SimpleAdditionAIR::<F>::new(&proof_options);
    let pub_inputs = SimpleAdditionPublicInputs {
        a: FE::from(1u64),
        b: FE::from(2u64),
    };
    (air, pub_inputs)
}

fn prove_with<H: StarkHash>(air: &SimpleAdditionAIR<F>, pub_inputs: &PI) -> StarkProof<F, F, PI> {
    let mut trace = simple_addition_trace::<F>(TRACE_ROWS);
    Prove::<H>::prove(
        air,
        &mut trace,
        pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed")
}

/// ★ The Stage-2 oracle: a real STARK proves and verifies end to end under the
/// BLAKE3 configuration.
///
/// This is a **same-reference** claim — this build's prover and this build's
/// verifier agree — and that is exactly what it is for. It makes no
/// cross-version claim; `scripts/cross_verify_vm.sh` is what covers the default
/// keccak path across refs.
#[test]
fn a_blake3_stark_proof_verifies() {
    let (air, pub_inputs) = air_and_inputs();
    let proof = prove_with::<Blake3StarkHash>(&air, &pub_inputs);

    // Non-vacuity: the proof must actually contain committed FRI layers, or it
    // would verify without ever exercising the trees this stage threads `H`
    // through.
    assert!(
        !proof.fri_layers_merkle_roots.is_empty(),
        "the test trace must fold; otherwise this proves nothing about fri/"
    );

    assert!(
        Verify::<Blake3StarkHash>::verify(&proof, &air, &mut DefaultTranscript::<F>::new(&[])),
        "an honest BLAKE3-committed proof must verify under the BLAKE3 verifier"
    );
}

/// HONEST-PATH CONTROL: the keccak configuration still round-trips.
///
/// The refactor rewrote the code path the default prover runs through; this
/// says it still proves and verifies. Its stronger sibling is the cross-ref
/// gate, which checks the actual proof BYTES did not move.
#[test]
fn the_keccak_stark_proof_still_verifies() {
    let (air, pub_inputs) = air_and_inputs();
    let proof = prove_with::<KeccakStarkHash>(&air, &pub_inputs);

    assert!(!proof.fri_layers_merkle_roots.is_empty());
    assert!(
        Verify::<KeccakStarkHash>::verify(&proof, &air, &mut DefaultTranscript::<F>::new(&[])),
        "an honest keccak-committed proof must verify under the keccak verifier"
    );
}

/// FALSIFICATION: tampering a FRI layer root must break verification.
///
/// A positive round trip alone cannot distinguish "the verifier checks the FRI
/// layer openings" from "the verifier reached the end without looking". This is
/// aimed at the precise bytes this stage changed the producer of.
#[test]
fn a_tampered_blake3_fri_layer_root_is_rejected() {
    let (air, pub_inputs) = air_and_inputs();
    let honest = prove_with::<Blake3StarkHash>(&air, &pub_inputs);

    for layer in 0..honest.fri_layers_merkle_roots.len() {
        let mut tampered = honest.clone();
        tampered.fri_layers_merkle_roots[layer][0] ^= 1;
        assert!(
            !Verify::<Blake3StarkHash>::verify(
                &tampered,
                &air,
                &mut DefaultTranscript::<F>::new(&[])
            ),
            "a proof with FRI layer root {layer} flipped must be rejected"
        );
    }
}

/// FALSIFICATION: a tampered FRI layer *opening* must break verification.
///
/// The root tamper above also moves every challenge drawn after it, so it would
/// be caught by a verifier that only replayed Fiat-Shamir. This one leaves the
/// transcript untouched and corrupts the authenticated value instead, so only
/// the Merkle check can catch it — and that check is `H::Batched` against a tree
/// the prover built with `H::Pair`.
#[test]
fn a_tampered_blake3_fri_layer_opening_is_rejected() {
    let (air, pub_inputs) = air_and_inputs();
    let honest = prove_with::<Blake3StarkHash>(&air, &pub_inputs);

    assert!(
        !honest.query_list.is_empty() && !honest.query_list[0].layers_evaluations_sym.is_empty(),
        "the test proof must carry FRI query openings"
    );

    let mut tampered = honest.clone();
    tampered.query_list[0].layers_evaluations_sym[0] += FE::one();
    assert!(
        !Verify::<Blake3StarkHash>::verify(&tampered, &air, &mut DefaultTranscript::<F>::new(&[])),
        "a tampered FRI symmetric evaluation must fail its Merkle authentication"
    );
}

/// ★ CONTROL — the stark-proof-level analog of
/// `the_blake3_and_keccak_configurations_commit_differently`.
///
/// Without this, every test above would pass just as well if `Blake3StarkHash`
/// still resolved to the keccak backends, or if `fri/` had kept building keccak
/// layer trees under a BLAKE3 `H`. Both proofs are over the same trace with the
/// same transcript seed, so nothing but the commitment hash can move these
/// bytes — and each verifier must reject the other configuration's proof.
#[test]
fn the_two_configurations_produce_mutually_unverifiable_proofs() {
    let (air, pub_inputs) = air_and_inputs();
    let blake3_proof = prove_with::<Blake3StarkHash>(&air, &pub_inputs);
    let keccak_proof = prove_with::<KeccakStarkHash>(&air, &pub_inputs);

    assert_ne!(
        blake3_proof.lde_trace_main_merkle_root, keccak_proof.lde_trace_main_merkle_root,
        "the two configurations must commit the same trace to different roots"
    );
    assert_ne!(
        blake3_proof.fri_layers_merkle_roots, keccak_proof.fri_layers_merkle_roots,
        "the two configurations must commit FRI layers to different roots"
    );

    assert!(
        !Verify::<KeccakStarkHash>::verify(
            &blake3_proof,
            &air,
            &mut DefaultTranscript::<F>::new(&[])
        ),
        "the keccak verifier must reject a BLAKE3-committed proof"
    );
    assert!(
        !Verify::<Blake3StarkHash>::verify(
            &keccak_proof,
            &air,
            &mut DefaultTranscript::<F>::new(&[])
        ),
        "the BLAKE3 verifier must reject a keccak-committed proof"
    );
}

/// ★ The FRI layer tree IS the configuration's tree over that layer's
/// evaluations — checked directly, at both configurations.
///
/// The round trip above says prover and verifier agree; it does not say *which*
/// hash they agree on, and a `commit_phase_from_evaluations` that ignored `H`
/// and used keccak for both would still round-trip under a keccak verifier.
/// This rebuilds each committed layer's tree from the layer's own evaluations
/// with `H::Pair` and demands the roots match, so the threading is pinned at the
/// producer rather than inferred from the consumer.
#[test]
fn fri_layer_trees_are_built_with_the_configurations_pair_backend() {
    use crate::fri::commit_phase_from_evaluations;
    use crate::fri::fri_functions::compute_coset_twiddles_inv;

    /// Returns each committed layer's root, and layer 0's folded codeword.
    fn check<H: StarkHash>(
        offset: &FE,
        len: usize,
        blowup_log: u32,
        k: u32,
    ) -> (Vec<[u8; 32]>, Vec<FE>) {
        let codeword: Vec<FE> = (0..len as u64).map(|i| FE::from(i * 7 + 1)).collect();
        let inv_twiddles = compute_coset_twiddles_inv::<F>(offset, len);
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let (_coeffs, layers) = commit_phase_from_evaluations::<F, F, _, H>(
            codeword,
            &mut transcript,
            offset,
            len,
            blowup_log,
            k,
            &inv_twiddles,
        );
        assert!(!layers.is_empty(), "the input must fold");

        for (i, layer) in layers.iter().enumerate() {
            let leaves: Vec<[FE; 2]> = layer
                .evaluation
                .chunks_exact(2)
                .map(|c| [c[0], c[1]])
                .collect();
            let rebuilt = MerkleTree::<H::Pair<F>>::build(&leaves).expect("rebuild layer tree");
            assert_eq!(
                rebuilt.root, layer.merkle_tree.root,
                "layer {i}'s committed root must be the H::Pair tree over its own evaluations"
            );
        }
        (
            layers.iter().map(|l| l.merkle_tree.root).collect(),
            layers[0].evaluation.clone(),
        )
    }

    let offset = FE::from(3u64);
    let (len, blowup_log, k) = (1usize << 10, 1u32, 5u32);

    let (keccak_roots, keccak_layer0) = check::<KeccakStarkHash>(&offset, len, blowup_log, k);
    let (blake3_roots, blake3_layer0) = check::<Blake3StarkHash>(&offset, len, blowup_log, k);

    // ζ₀ is drawn before anything is appended, so both configurations fold the
    // same input with the same challenge and layer 0's codeword is identical.
    // Checked rather than argued, because it is what makes the root comparison
    // below mean "the hash differs" instead of "the input differs".
    assert_eq!(
        keccak_layer0, blake3_layer0,
        "layer 0 must fold identically under both configurations"
    );
    assert_ne!(
        keccak_roots[0], blake3_roots[0],
        "over one identical codeword, the layer root must differ only because \
         the hash does"
    );
}
