//! The adversarial review REV-P1-B's tests for the Poseidon1 base path (the
//! subset REV-P1-JUDGE §7 keeps), flipped where the judge's fixes change their
//! expectation: a zero-padded opening is refused by the P1 leaf hash itself
//! (the width tag) as well as by the verifier, the statement tag names the cap
//! policy, every cap node is bound by the cap root, the two bases do not
//! cross-verify, and the width-8 grind credits its bits.

use crypto::hash::poseidon1_w8;
use crypto::merkle_tree::traits::IsStreamingLeafBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::config::StarkHash;
use stark::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use stark::proof::options::{BaseFormat, CapPolicy, ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::prover::{GenericProver, IsStarkProver};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{GenericVerifier, IsStarkVerifier};

use crate::lfm::algebraic_commit::{RpxStarkHash, commitment_to_digest};
use crate::lfm::algebraic_transcript::AlgebraicTranscript;
use crate::lfm::hash::HasherKind;
use crate::lfm::p1_commit::{P1BatchBackend, P1GrindDigest, P1StarkHash, P1Transcript};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Fe = FieldElement<F>;
type P1B = P1BatchBackend<F>;
type RpxB = <RpxStarkHash as StarkHash>::Batched<F>;

fn fe(x: u64) -> Fe {
    Fe::from(x)
}

// =====================================================================
// 3. The verifier, the bases and the grind
// =====================================================================

fn options(format: ProofFormat, grinding: u8) -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 30,
        coset_offset: 3,
        grinding_factor: grinding,
        fri_final_poly_log_degree: 1,
        format,
    }
}

fn p1_format(cap: CapPolicy) -> ProofFormat {
    ProofFormat {
        base: BaseFormat {
            arity4_cap: cap,
            ..BaseFormat::P1
        },
        ..ProofFormat::LEGACY
    }
}

type Proof = StarkProof<F, E, LogReadOnlyPublicInputs<F>>;

fn logup_trace(rows: usize) -> (TraceTable<F, E>, LogReadOnlyPublicInputs<F>) {
    let addr: Vec<Fe> = (0..rows).map(|i| fe((i % 5) as u64 + 1)).collect();
    let val: Vec<Fe> = (0..rows).map(|i| fe(((i % 5) as u64 + 1) * 10)).collect();
    let trace: TraceTable<F, E> = read_only_logup_trace(addr, val);
    let cols = trace.columns_main();
    let pi = LogReadOnlyPublicInputs {
        a0: cols[0][0],
        v0: cols[1][0],
        a_sorted_0: cols[2][0],
        v_sorted_0: cols[3][0],
        m0: cols[4][0],
    };
    (trace, pi)
}

fn prove_p1(o: &ProofOptions) -> Proof {
    let (mut trace, pi) = logup_trace(128);
    let air = LogReadOnlyRAP::<F, E>::new(o);
    GenericProver::<F, E, _, P1StarkHash>::prove(&air, &mut trace, &pi, &mut P1Transcript::new())
        .expect("P1 proves")
}

fn verify_p1(o: &ProofOptions, proof: &Proof) -> bool {
    GenericVerifier::<F, E, _, P1StarkHash>::verify(
        proof,
        &LogReadOnlyRAP::<F, E>::new(o),
        &mut P1Transcript::new(),
    )
}

fn prove_rpx(o: &ProofOptions) -> Proof {
    let (mut trace, pi) = logup_trace(128);
    let air = LogReadOnlyRAP::<F, E>::new(o);
    GenericProver::<F, E, _, RpxStarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut AlgebraicTranscript::new(HasherKind::Rpx),
    )
    .expect("RPX proves")
}

fn verify_rpx(o: &ProofOptions, proof: &Proof) -> bool {
    GenericVerifier::<F, E, _, RpxStarkHash>::verify(
        proof,
        &LogReadOnlyRAP::<F, E>::new(o),
        &mut AlgebraicTranscript::new(HasherKind::Rpx),
    )
}

/// One extra zero on the symmetric main row (10 → 11 felts, still one block)
/// moves the P1 leaf hash, through the width tag in its first capacity, as it
/// moves the RPX one; and the verifier's AIR-derived width checks refuse the
/// proof as before. Two bindings, not one.
#[test]
fn rev_p1b_a_zero_padded_opening_is_refused_by_the_p1_leaf_and_the_verifier() {
    let o = options(p1_format(CapPolicy::Off), 0);
    let proof = prove_p1(&o);
    assert!(verify_p1(&o, &proof));

    let main = &proof.deep_poly_openings[0].main_trace_polys;
    let (ev, sym) = (main.evaluations.clone(), main.evaluations_sym.clone());
    assert_eq!(ev.len() + sym.len(), 10, "five main columns, a row pair");
    let mut sym0 = sym.clone();
    sym0.push(Fe::zero());
    assert_ne!(
        <P1B as IsStreamingLeafBackend<F>>::hash_data_from_slices(&ev, &sym),
        <P1B as IsStreamingLeafBackend<F>>::hash_data_from_slices(&ev, &sym0),
        "the P1 leaf hash sees the extra zero (the width tag)"
    );
    assert_ne!(
        <RpxB as IsStreamingLeafBackend<F>>::hash_data_from_slices(&ev, &sym),
        <RpxB as IsStreamingLeafBackend<F>>::hash_data_from_slices(&ev, &sym0),
        "the RPX leaf hash does"
    );

    let mut bad = proof.clone();
    for q in &mut bad.deep_poly_openings {
        q.main_trace_polys.evaluations_sym.push(Fe::zero());
    }
    assert!(!verify_p1(&o, &bad), "the width checks refuse it");
}

/// The bases do not cross-verify: a P1 proof under the RPX configuration and
/// an RPX proof under P1 are refused, and so is a P1 proof replayed on the
/// RPX transcript (the half-flip `hash_pin` warns about).
#[test]
fn rev_p1b_the_bases_do_not_cross_verify() {
    let p1o = options(p1_format(CapPolicy::Fixed(2)), 4);
    let rpxo = options(ProofFormat::LEGACY, 4);
    let p1 = prove_p1(&p1o);
    let rpx = prove_rpx(&rpxo);
    assert!(verify_p1(&p1o, &p1));
    assert!(verify_rpx(&rpxo, &rpx));
    assert!(!verify_rpx(&rpxo, &p1), "P1 proof, RPX verifier");
    assert!(
        !verify_rpx(&p1o, &p1),
        "P1 proof, RPX verifier at the P1 format"
    );
    assert!(!verify_p1(&p1o, &rpx), "RPX proof, P1 verifier");
    let half_flip = GenericVerifier::<F, E, _, P1StarkHash>::verify(
        &p1,
        &LogReadOnlyRAP::<F, E>::new(&p1o),
        &mut AlgebraicTranscript::new(HasherKind::Rpx),
    );
    assert!(!half_flip, "P1 commitments on the RPX transcript");
}

/// The statement tag names the cap policy: `Off` and `Auto` read apart
/// (REV-P1-JUDGE F2). The tag is domain separation, not the binding — a proof
/// made under `Auto` (a 4-ary cap of height 3 at 30 openings) is refused by a
/// verifier at `Off`, by the exact path lengths.
#[test]
fn rev_p1b_the_tag_names_auto_and_the_geometry_binds() {
    use crate::hash_pin::p1_statement_tag;
    assert_ne!(
        p1_statement_tag(CapPolicy::Off),
        p1_statement_tag(CapPolicy::Auto),
        "Off and Auto read apart"
    );
    let auto = options(p1_format(CapPolicy::Auto), 0);
    let off = options(p1_format(CapPolicy::Off), 0);
    let proof = prove_p1(&auto);
    assert!(verify_p1(&auto, &proof));
    assert!(!verify_p1(&off, &proof), "an Auto-capped proof under Off");
}

/// The width-8 grind credits its bits: `is_valid_nonce` under the P1 digest
/// is exactly "lane 0 of W8(inner ‖ nonce ‖ 0³), canonical, below
/// 2^(64 − g)", the inner digest binds the seed and the factor, and the pass
/// rate over nonces is 2^−g.
#[test]
fn rev_p1b_the_width8_grind_credits_its_bits() {
    use crypto::grinding::is_valid_nonce;
    use digest::Digest;

    let seed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0x5a);
    let g = 6u8;
    // The definition, recomputed from the permutation.
    let inner: [u8; 32] = {
        let mut d = P1GrindDigest::new();
        Digest::update(&mut d, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed]);
        Digest::update(&mut d, seed);
        Digest::update(&mut d, [g]);
        d.finalize().into()
    };
    let lane0 = |nonce: u64| {
        let i = commitment_to_digest(&inner);
        let mut s = [Fe::zero(); poseidon1_w8::STATE_FELTS];
        s[..4].copy_from_slice(&i);
        s[4] = Fe::from(nonce);
        poseidon1_w8::permute(s)[0].canonical()
    };
    let n = 1u64 << 14;
    let mut passes = 0u64;
    for nonce in 0..n {
        let ok = is_valid_nonce::<P1GrindDigest>(&seed, nonce, g);
        assert_eq!(ok, lane0(nonce) < 1u64 << (64 - g), "nonce {nonce}");
        passes += u64::from(ok);
    }
    // Binomial(2^14, 2^-6): mean 256, sd ≈ 15.9; ±6 sd.
    assert!((160..=352).contains(&passes), "{passes} passes of {n}");
    // The inner digest binds the seed: one flipped seed bit moves the
    // passing set.
    let mut other = seed;
    other[31] ^= 1;
    assert!(
        (0..1024u64).any(|k| is_valid_nonce::<P1GrindDigest>(&other, k, g)
            != is_valid_nonce::<P1GrindDigest>(&seed, k, g)),
        "the seed moves the passing set"
    );
}

/// The RPX control of the zero-padded opening: the same tamper on an RPX
/// proof is refused by the leaf hash too, so it stays refused when the width
/// checks are mutated away (REV-P1-B mutation M1), as the P1 one now is.
#[test]
fn rev_p1b_rpx_refuses_the_zero_padded_opening_by_its_leaf_hash() {
    let o = options(ProofFormat::LEGACY, 0);
    let proof = prove_rpx(&o);
    assert!(verify_rpx(&o, &proof));
    let mut bad = proof.clone();
    for q in &mut bad.deep_poly_openings {
        q.main_trace_polys.evaluations_sym.push(Fe::zero());
    }
    assert!(!verify_rpx(&o, &bad));
}

/// Every cap node is bound, including the ones no query lands on: those are
/// refused only by the cap-to-root check (`verify_cap_shaped`), which the
/// crate's capped tamper test does not reach (its flipped nodes are all hit
/// by some opening). 256 rows at blowup 4: 512 row-pair leaves, binary depth
/// 9, five 4-ary levels; at height 4 the owner path keeps 3 siblings and
/// carries the 2·4³ = 128 cap nodes, of which 30 queries reach at most 30.
#[test]
fn rev_p1b_every_cap_node_is_bound_by_the_cap_root() {
    let o = options(p1_format(CapPolicy::Fixed(4)), 0);
    let (mut trace, pi) = logup_trace(256);
    let air = LogReadOnlyRAP::<F, E>::new(&o);
    let proof = GenericProver::<F, E, _, P1StarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut P1Transcript::new(),
    )
    .expect("P1 proves");
    assert!(verify_p1(&o, &proof));
    let owner = &proof.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path;
    assert_eq!(owner.len(), 3 + 128, "3 kept siblings and the 128-node cap");
    let refused = (3..owner.len())
        .filter(|&k| {
            let mut bad = proof.clone();
            bad.deep_poly_openings[0].main_trace_polys.proof.merkle_path[k][31] ^= 1;
            !verify_p1(&o, &bad)
        })
        .count();
    assert_eq!(refused, 128, "every cap node of the main tree is bound");
}
