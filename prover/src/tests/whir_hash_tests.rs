//! ★★ The hash seam, end to end, on a real VM table — the four propositions the
//! byte gate is made of, in the one place they can all be checked in process.
//!
//! This is the miniature of the A/B the box will run, and it is deliberately
//! shaped the same way: **one process, the trace built once, proved once per
//! hash.** Two processes could not be compared, because six of this VM's trace
//! generators lay their rows out in `HashMap` iteration order
//! (`prover/src/tables/eq.rs:128` and five siblings), so the same program gives
//! different traces run to run for reasons that have nothing to do with the
//! hash.
//!
//! | | proposition | why it can fail |
//! |---|---|---|
//! | **G1** | the keccak arm is the proof PR #988 produces | a threading mistake would move a byte |
//! | **G2** | the RPX arm proves and verifies, and its proof DIFFERS | a seam that silently kept hashing keccak would pass a verify and fail this |
//! | **G3** | prove under one hash, verify under the other ⇒ REJECTED | this is what makes G2 non-vacuous: prover and verifier agree on a wrong hash too |
//! | **B3** | the two arms serialize to the SAME LENGTH, to the byte | a digest-width or field change would move it; a hash change must not |
//!
//! G3 is the load-bearing one. "Prover and verifier agree" is worth nothing on
//! its own — they would agree on a hash that returned its input. What says the
//! hash is really in the proof is that a verifier told to expect the other one
//! rejects, and that the KATs in `crypto::hash::rpx::tests` pin the digest to
//! numbers this repository did not produce.

use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, MultiProof, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::eq::{EqOperation, generate_eq_trace};
use crate::test_utils::{ConcreteVmAir, create_eq_air};

type Proof = MultiProof<Fp, Ext>;
type Columns = Vec<Vec<FieldElement<Fp>>>;

/// Grinding ON: the grind is one of the three hash consumers the seam names, so
/// an arm that left it on keccak would be a half-flip and this fixture has to
/// be able to see it. Four bits, so the search costs nothing.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::uniform(4),
        format: multilinear::whir_chain::ChainFormat::DEFAULT,
    }
}

fn fixture_columns() -> Columns {
    let ops = vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(3, 3, true),
        EqOperation::new(3, 5, true),
    ];
    generate_eq_trace(&ops).columns_main()
}

fn air() -> ConcreteVmAir<impl stark::constraints::builder::ConstraintSet<Fp, Ext>> {
    create_eq_air(&ProofOptions::default_test_options())
}

fn layout(columns: &Columns) -> TableLayout<'static, Fp, Ext> {
    // The AIR is rebuilt per call so the layout borrows nothing from a temporary.
    let a = Box::leak(Box::new(air()));
    TableLayout::<Fp, Ext>::new(
        a.constraint_program(),
        a.constraints_meta(),
        a.bus_interactions(),
        columns.len(),
        columns[0].len().trailing_zeros() as usize,
        Uniforms::default(),
    )
    .expect("layout")
}

/// Prove the fixed trace under `H`.
fn prove<H: WhirHash>(columns: &Columns) -> Proof {
    let table = CommittedTable::from_layout(layout(columns), |col| columns[col as usize].clone())
        .expect("committed table");
    let committed = CommittedTables::<_, _, H>::commit(vec![table], &config()).expect("commit");
    let mut transcript = DefaultTranscript::<Ext, H::Transcript>::new(b"whir-hash-seam");
    multilinear_table::multi_prove(&committed, &config(), &mut transcript, None).expect("prove")
}

/// Verify `proof` under `H`. Returns the verifier's verdict rather than
/// unwrapping, because half these calls are supposed to fail.
fn verify<H: WhirHash>(proof: &Proof, columns: &Columns) -> Result<(), multilinear::Error> {
    let table = CommittedTable::from_layout(layout(columns), |col| columns[col as usize].clone())
        .expect("committed table");
    let committed = CommittedTables::<_, _, H>::commit(vec![table], &config()).expect("commit");

    let owed = multilinear_table::contribution(&proof.tables[0].bus_output)
        .ok_or(multilinear::Error::BusImbalance)?;
    let verifier_layout = layout(columns);
    let statement = verifier_layout.statement();
    let mut transcript = DefaultTranscript::<Ext, H::Transcript>::new(b"whir-hash-seam");
    multilinear_table::multi_verify::<_, _, _, H>(
        proof,
        &[statement],
        std::slice::from_ref(committed.groups()[0].layout()),
        std::slice::from_ref(committed.groups()[0].domain()),
        committed.sizes(),
        &owed,
        &config(),
        &mut transcript,
        None,
    )
}

fn serialized(proof: &Proof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// (G2, first half) The RPX arm proves and verifies. On its own this says very
/// little — see the module header — which is why it is one line of four.
#[test]
fn the_rpx_arm_proves_and_verifies() {
    let columns = fixture_columns();
    let proof = prove::<RpxWhir>(&columns);
    verify::<RpxWhir>(&proof, &columns).expect("an RPX proof must verify under RPX");
}

/// The keccak arm still does, unchanged.
#[test]
fn the_keccak_arm_proves_and_verifies() {
    let columns = fixture_columns();
    let proof = prove::<KeccakWhir>(&columns);
    verify::<KeccakWhir>(&proof, &columns).expect("a keccak proof must verify under keccak");
}

/// (G2, second half) ★ The two arms produce DIFFERENT proofs.
///
/// Checked at the root, not at the whole blob: a root is the one field whose
/// difference can only come from the Merkle hash, so this distinguishes "the
/// hash moved" from "some challenge moved".
#[test]
fn the_two_hashes_produce_different_roots() {
    let columns = fixture_columns();
    let keccak = prove::<KeccakWhir>(&columns);
    let rpx = prove::<RpxWhir>(&columns);

    assert_eq!(keccak.roots.len(), 1, "the fixture commits one group");
    assert_ne!(
        keccak.roots, rpx.roots,
        "a seam that kept hashing keccak under RpxWhir would land here"
    );
}

/// (G3) ★★ **The arm that must FAIL.** A proof made under one hash must be
/// rejected by a verifier expecting the other, in both directions.
#[test]
fn a_proof_made_under_one_hash_is_rejected_under_the_other() {
    let columns = fixture_columns();

    let keccak = prove::<KeccakWhir>(&columns);
    assert!(
        verify::<RpxWhir>(&keccak, &columns).is_err(),
        "an RPX verifier accepted a keccak proof"
    );

    let rpx = prove::<RpxWhir>(&columns);
    assert!(
        verify::<KeccakWhir>(&rpx, &columns).is_err(),
        "a keccak verifier accepted an RPX proof"
    );
}

/// (B3) ★ The serialized length is EQUAL to the byte.
///
/// The sharper of the two byte statements: the digest is 32 bytes under either
/// hash and no proof struct gains a field, so the length is not allowed to move
/// even though every byte inside it does. A difference here is a defect in the
/// seam, not a property of the hash.
#[test]
fn the_two_hashes_serialize_to_the_same_length() {
    let columns = fixture_columns();
    let keccak = serialized(&prove::<KeccakWhir>(&columns));
    let rpx = serialized(&prove::<RpxWhir>(&columns));

    assert_eq!(
        keccak.len(),
        rpx.len(),
        "a hash swap must not be a proof-format change"
    );
    assert_ne!(keccak, rpx, "…but the bytes themselves must differ");
}

/// ✓ The configurations name themselves, and differently — the string the
/// banner prints and the KATs are filed under.
#[test]
fn the_two_configurations_have_distinct_names() {
    assert_eq!(KeccakWhir::NAME, "keccak256");
    assert_eq!(RpxWhir::NAME, "rpx256");
    assert_ne!(KeccakWhir::NAME, RpxWhir::NAME);
}

/// ★ The GRIND follows the configuration, not a default.
///
/// The grind is the seam's third consumer and the easiest to leave behind,
/// because it is reached through a free function rather than through a type. A
/// nonce valid under one configuration's digest is invalid under the other's
/// with overwhelming probability, so this is a direct read of which hash the
/// proof-of-work actually ran on.
#[test]
fn the_grind_follows_the_configuration() {
    use crypto::grinding::{generate_nonce_smallest, is_valid_nonce};
    use multilinear::whir_hash::GrindingDigest;

    let seed = [7u8; 32];
    let factor = 12u8;

    let k = generate_nonce_smallest::<GrindingDigest<KeccakWhir>>(&seed, factor).expect("nonce");
    let r = generate_nonce_smallest::<GrindingDigest<RpxWhir>>(&seed, factor).expect("nonce");

    assert!(is_valid_nonce::<GrindingDigest<KeccakWhir>>(
        &seed, k, factor
    ));
    assert!(is_valid_nonce::<GrindingDigest<RpxWhir>>(&seed, r, factor));
    assert_ne!(
        k, r,
        "the same seed must not grind to the same nonce under two different hashes"
    );
    assert!(
        !is_valid_nonce::<GrindingDigest<RpxWhir>>(&seed, k, factor),
        "keccak's nonce must not satisfy RPX's predicate"
    );
    assert!(
        !is_valid_nonce::<GrindingDigest<KeccakWhir>>(&seed, r, factor),
        "RPX's nonce must not satisfy keccak's predicate"
    );
}

/// ★ The two transcript TYPES draw different challenges.
///
/// ⚠ RENAMED. This was called `the_transcript_follows_the_configuration`, which
/// is a claim about the PROVER — and this body never mentions the prover. It
/// constructs both transcript types itself and compares them, so it was true
/// for the whole period in which no WHIR call site built an RPX transcript at
/// all, and it would have stayed true if none ever did.
///
/// What it does check is worth keeping: that the two configurations are not
/// accidentally the same sponge. The claim its old name made is now checked two
/// ways — by the compiler, via the `HasTranscriptHash` bound on `multi_prove`
/// and `multi_verify`, which makes a mismatched transcript unspellable rather
/// than merely untested; and at runtime by
/// `prover/tests/whir_transcript_configuration.rs`, which runs a real prove and
/// reads the Fiat-Shamir counters afterwards.
#[test]
fn the_two_transcript_types_draw_different_challenges() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use crypto::fiat_shamir::transcript_hash::TranscriptHash;

    type KeccakT = DefaultTranscript<Ext, <KeccakWhir as WhirHash>::Transcript>;
    type RpxT = DefaultTranscript<Ext, <RpxWhir as WhirHash>::Transcript>;

    let mut k = KeccakT::new(b"same-seed");
    let mut r = RpxT::new(b"same-seed");
    k.append_bytes(b"same-absorbed-bytes");
    r.append_bytes(b"same-absorbed-bytes");

    assert_ne!(k.state(), r.state(), "two hashes, two sponge states");
    assert_ne!(
        k.sample_field_element(),
        r.sample_field_element(),
        "two hashes, two challenge streams"
    );
    assert_eq!(<KeccakWhir as WhirHash>::Transcript::NAME, "keccak256");
    assert_eq!(<RpxWhir as WhirHash>::Transcript::NAME, "rpx256");
}

// ===========================================================================
// ZisK's Poseidon1 (D-WHIR-P1 S1): the same seam at a third hash, whose trees
// are 4-ary and whose sponge is a field sponge, through each configuration's
// own `WhirHash::sponge`.
// ===========================================================================

/// The configuration's own sponge with the fixture's tag absorbed.
fn sponge<H: WhirHash>() -> H::Sponge {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    let mut t = H::sponge();
    t.append_bytes(b"whir-hash-seam");
    t
}

fn prove_sponged<H: WhirHash>(columns: &Columns) -> Proof {
    let table = CommittedTable::from_layout(layout(columns), |col| columns[col as usize].clone())
        .expect("committed table");
    let committed = CommittedTables::<_, _, H>::commit(vec![table], &config()).expect("commit");
    multilinear_table::multi_prove(&committed, &config(), &mut sponge::<H>(), None).expect("prove")
}

fn verify_sponged<H: WhirHash>(proof: &Proof, columns: &Columns) -> Result<(), multilinear::Error> {
    let table = CommittedTable::from_layout(layout(columns), |col| columns[col as usize].clone())
        .expect("committed table");
    let committed = CommittedTables::<_, _, H>::commit(vec![table], &config()).expect("commit");
    let owed = multilinear_table::contribution(&proof.tables[0].bus_output)
        .ok_or(multilinear::Error::BusImbalance)?;
    let verifier_layout = layout(columns);
    let statement = verifier_layout.statement();
    multilinear_table::multi_verify::<_, _, _, H>(
        proof,
        &[statement],
        std::slice::from_ref(committed.groups()[0].layout()),
        std::slice::from_ref(committed.groups()[0].domain()),
        committed.sizes(),
        &owed,
        &config(),
        &mut sponge::<H>(),
        None,
    )
}

/// The RPX and keccak sponges ARE the transcripts the block always built
/// (`DefaultTranscript::new(&[])`): equal states and challenges after the same
/// absorbs, so routing the block through `WhirHash::sponge` moves no byte.
#[test]
fn the_rpx_and_keccak_sponges_are_the_default_transcripts() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    fn same<H: WhirHash>() {
        let mut a = H::sponge();
        let mut b = DefaultTranscript::<Ext, H::Transcript>::new(&[]);
        for t in [&mut a as &mut dyn IsTranscript<Ext>, &mut b] {
            t.append_bytes(b"statement");
            t.append_field_element(&FieldElement::<Ext>::from(7u64));
        }
        assert_eq!(a.state(), b.state(), "{}", H::NAME);
        assert_eq!(a.sample_field_element(), b.sample_field_element());
        assert_eq!(a.sample_u64(1 << 20), b.sample_u64(1 << 20));
    }
    same::<KeccakWhir>();
    same::<RpxWhir>();
}

/// (G2 at P1) The Poseidon1 arm proves and verifies on the real table, and its
/// roots and bytes differ from RPX's under the same fixture.
#[test]
fn the_p1_arm_proves_and_verifies_and_differs_from_rpx() {
    use multilinear::whir_hash::P1Whir;
    let columns = fixture_columns();
    let p1 = prove_sponged::<P1Whir>(&columns);
    verify_sponged::<P1Whir>(&p1, &columns).expect("a P1 proof must verify under P1");
    let rpx = prove_sponged::<RpxWhir>(&columns);
    verify_sponged::<RpxWhir>(&rpx, &columns).expect("an RPX proof must verify under RPX");
    assert_ne!(p1.roots, rpx.roots, "the P1 seam must hash with P1");
    assert_ne!(serialized(&p1), serialized(&rpx));
}

/// (G3 at P1) ★★ Prove under one, verify under the other: rejected, both ways,
/// and against keccak too.
#[test]
fn a_p1_proof_and_an_rpx_proof_do_not_cross_verify() {
    use multilinear::whir_hash::P1Whir;
    let columns = fixture_columns();
    let p1 = prove_sponged::<P1Whir>(&columns);
    assert!(verify_sponged::<RpxWhir>(&p1, &columns).is_err());
    assert!(verify_sponged::<KeccakWhir>(&p1, &columns).is_err());
    let rpx = prove_sponged::<RpxWhir>(&columns);
    assert!(verify_sponged::<P1Whir>(&rpx, &columns).is_err());
}

/// The P1 grind is its own: its nonce fails RPX's predicate and the reverse.
#[test]
fn the_p1_grind_follows_its_configuration() {
    use crypto::grinding::{generate_nonce_smallest, is_valid_nonce};
    use multilinear::whir_hash::{GrindingDigest, P1Whir};
    let seed = [7u8; 32];
    let factor = 12u8;
    let p = generate_nonce_smallest::<GrindingDigest<P1Whir>>(&seed, factor).expect("nonce");
    let r = generate_nonce_smallest::<GrindingDigest<RpxWhir>>(&seed, factor).expect("nonce");
    assert!(is_valid_nonce::<GrindingDigest<P1Whir>>(&seed, p, factor));
    assert_ne!(p, r);
    assert!(!is_valid_nonce::<GrindingDigest<RpxWhir>>(&seed, p, factor));
    assert!(!is_valid_nonce::<GrindingDigest<P1Whir>>(&seed, r, factor));
    assert_eq!(P1Whir::NAME, "poseidon1-w16");
}
