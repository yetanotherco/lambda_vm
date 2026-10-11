//! STARK round trips under ZisK's Poseidon1 configuration
//! (`stark::config::P1StarkHashCapped`: arity-4 trees, ZisK's tagged leaf hash,
//! width-8 grinding), on the host, and the tamper checks that make the
//! verifier's arity-4 walk load-bearing; the arity-4 Merkle caps at fixed
//! heights, set by the configuration's type (never by the format, the
//! environment or the proof).
//!
//! Ported from #1013's `p1_stark_tests` (P1 base STARK proofs), whose cap there
//! was a format field: here each height is its own configuration, so a
//! "verifier at height `c`" is the verifier of `P1StarkHashCapped<c>`.
//!
//! The goldens at the end pin the host bytes of P1 proofs at the production
//! cap (`P1StarkHash`); `zf_p1_device_tests` proves the same cases on the card
//! and requires the same bytes. Regenerate them only for a deliberate format
//! change: `cargo test -p lambda-vm-prover --lib
//! tests::p1_stark_tests::print_p1_goldens -- --ignored --nocapture`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::traits::IsStreamingLeafBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use sha2::{Digest, Sha256};
use stark::config::StarkHash;
use stark::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use stark::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use stark::proof::options::{CapPolicy, FriMode, ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::prover::{GenericProver, IsStarkProver};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{GenericVerifier, IsStarkVerifier};

use crate::lfm::algebraic_commit::RpxStarkHash;
use crate::lfm::algebraic_transcript::AlgebraicTranscript;
use crate::lfm::hash::HasherKind;
use crate::lfm::p1_commit::{P1BatchBackend, P1StarkHash, P1StarkHashCapped, P1Transcript};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type LogupProof = StarkProof<F, E, LogReadOnlyPublicInputs<F>>;

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

fn prove_addition<H: StarkHash>(
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
    let proof = GenericProver::<F, F, _, H>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

fn verify_addition<H: StarkHash>(
    air: &SimpleAdditionAIR<F>,
    proof: &StarkProof<F, F, SimpleAdditionPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, F, _, H>::verify(proof, air, &mut DefaultTranscript::<F>::new(&[]))
}

fn logup_trace(rows: usize) -> (TraceTable<F, E>, LogReadOnlyPublicInputs<F>) {
    let addr: Vec<Felt> = (0..rows).map(|i| Felt::from((i % 5) as u64 + 1)).collect();
    let val: Vec<Felt> = (0..rows)
        .map(|i| Felt::from(((i % 5) as u64 + 1) * 10))
        .collect();
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

/// A logup proof under the configuration `H` on ZisK's transcript.
pub(crate) fn prove_logup_under<H: StarkHash>(rows: usize, o: &ProofOptions) -> LogupProof {
    let (mut trace, pi) = logup_trace(rows);
    let air = LogReadOnlyRAP::<F, E>::new(o);
    GenericProver::<F, E, _, H>::prove(&air, &mut trace, &pi, &mut P1Transcript::new())
        .expect("proving must succeed")
}

/// The verifier of `H` on ZisK's transcript, with its own AIR at `o`.
pub(crate) fn verify_logup_under<H: StarkHash>(o: &ProofOptions, proof: &LogupProof) -> bool {
    GenericVerifier::<F, E, _, H>::verify(
        proof,
        &LogReadOnlyRAP::<F, E>::new(o),
        &mut P1Transcript::new(),
    )
}

/// [`prove_logup_under`] under the P1 configuration capped at 4-ary height
/// `c` (0: uncapped).
fn prove_logup_at(c: u8, rows: usize, o: &ProofOptions) -> LogupProof {
    match c {
        0 => prove_logup_under::<P1StarkHashCapped<0>>(rows, o),
        1 => prove_logup_under::<P1StarkHashCapped<1>>(rows, o),
        2 => prove_logup_under::<P1StarkHashCapped<2>>(rows, o),
        3 => prove_logup_under::<P1StarkHashCapped<3>>(rows, o),
        4 => prove_logup_under::<P1StarkHashCapped<4>>(rows, o),
        5 => prove_logup_under::<P1StarkHashCapped<5>>(rows, o),
        _ => panic!("no P1 configuration at cap {c} in these tests"),
    }
}

/// The verifier of the P1 configuration capped at 4-ary height `c`.
fn verify_logup_at(c: u8, o: &ProofOptions, proof: &LogupProof) -> bool {
    match c {
        0 => verify_logup_under::<P1StarkHashCapped<0>>(o, proof),
        1 => verify_logup_under::<P1StarkHashCapped<1>>(o, proof),
        2 => verify_logup_under::<P1StarkHashCapped<2>>(o, proof),
        3 => verify_logup_under::<P1StarkHashCapped<3>>(o, proof),
        4 => verify_logup_under::<P1StarkHashCapped<4>>(o, proof),
        5 => verify_logup_under::<P1StarkHashCapped<5>>(o, proof),
        _ => panic!("no P1 configuration at cap {c} in these tests"),
    }
}

/// Every Merkle path node a proof carries (trace, composition and FRI
/// openings): what a cap changes.
fn path_nodes(proof: &LogupProof) -> usize {
    let trace: usize = proof
        .deep_poly_openings
        .iter()
        .map(|o| {
            o.main_trace_polys.proof.merkle_path.len()
                + o.aux_trace_polys
                    .as_ref()
                    .map_or(0, |p| p.proof.merkle_path.len())
                + o.composition_poly.proof.merkle_path.len()
        })
        .sum();
    let fri: usize = proof
        .query_list
        .iter()
        .flat_map(|q| q.layers_auth_paths.iter().map(|p| p.merkle_path.len()))
        .sum();
    trace + fri
}

/// A logup proof at cap height `c` under every format, verified at `c` and
/// refused by the verifiers of the neighbouring heights `lo` and `hi`.
fn capped_round_trip(c: u8, lo: u8, hi: u8) {
    for (name, format) in formats() {
        for rows in [128usize, 256] {
            let o = options(4, 30, 0, format);
            let proof = prove_logup_at(c, rows, &o);
            assert!(verify_logup_at(c, &o, &proof), "{name} rows {rows} c {c}");
            assert!(
                !verify_logup_at(lo, &o, &proof),
                "{name} rows {rows} c {c} by {lo}"
            );
            // A height at or past every tree's levels clamps to them: the
            // next height up is the same configuration there.
            let trace_levels = (rows.trailing_zeros() as usize + 1).div_ceil(2);
            if (c as usize) < trace_levels {
                assert!(
                    !verify_logup_at(hi, &o, &proof),
                    "{name} rows {rows} c {c} by {hi}"
                );
            }
        }
    }
}

#[test]
fn p1_proofs_round_trip_at_every_tree_shape() {
    // Rows 2^8 and 2^9 at blowup 4: LDE depths 10 and 11, even and odd, so
    // both the full and the zero-padded top level are walked; uncapped and at
    // the production cap, on the byte transcript and on ZisK's.
    for (name, format) in formats() {
        for rows in [256usize, 512] {
            let o = options(4, 9, 4, format);
            let (air, proof) = prove_addition::<P1StarkHashCapped<0>>(rows, &o);
            assert!(
                verify_addition::<P1StarkHashCapped<0>>(&air, &proof),
                "{name} rows {rows}"
            );
            let (air, proof) = prove_addition::<P1StarkHash>(rows, &o);
            assert!(
                verify_addition::<P1StarkHash>(&air, &proof),
                "{name} rows {rows} capped"
            );
        }
        let o = options(4, 9, 4, format);
        let proof = prove_logup_at(0, 128, &o);
        assert!(verify_logup_at(0, &o, &proof), "{name} logup");
        let proof = prove_logup_under::<P1StarkHash>(128, &o);
        assert!(
            verify_logup_under::<P1StarkHash>(&o, &proof),
            "{name} logup capped"
        );
    }
}

#[test]
fn the_formats_binary_cap_policy_does_not_reach_arity_4() {
    // `cap = auto` caps a binary tree opened this often; an arity-4
    // configuration runs its own cap whatever the format says, so `auto` and
    // `off` give the same proof under every P1 cap.
    let auto = ProofFormat {
        merkle_cap: CapPolicy::Auto,
        ..ProofFormat::LEGACY
    };
    // No grinding: the proofs must draw the same queries.
    let (oa, ol) = (
        options(4, 30, 0, auto),
        options(4, 30, 0, ProofFormat::LEGACY),
    );
    let a = prove_logup_at(0, 128, &oa);
    let b = prove_logup_at(0, 128, &ol);
    assert!(verify_logup_at(0, &oa, &a));
    assert_eq!(
        a.deep_poly_openings[0].main_trace_polys.proof.merkle_path,
        b.deep_poly_openings[0].main_trace_polys.proof.merkle_path
    );
    let a = prove_logup_at(2, 128, &oa);
    let b = prove_logup_at(2, 128, &ol);
    assert_eq!(
        a.deep_poly_openings[1].main_trace_polys.proof.merkle_path,
        b.deep_poly_openings[1].main_trace_polys.proof.merkle_path
    );
    // The production format's own binary policy (`ZfFormat::DEFAULT`, cap
    // auto) leaves a P1 proof as the uncapped-format one.
    let prod = options(4, 30, 0, crate::zf_format::ZfFormat::DEFAULT.proof_format());
    let dp = options(
        4,
        30,
        0,
        ProofFormat {
            merkle_cap: CapPolicy::Off,
            ..prod.format
        },
    );
    let a = prove_logup_under::<P1StarkHash>(128, &prod);
    let b = prove_logup_under::<P1StarkHash>(128, &dp);
    assert!(verify_logup_under::<P1StarkHash>(&dp, &a));
    assert_eq!(
        rkyv::to_bytes::<rkyv::rancor::Error>(&a).expect("rkyv")[..],
        rkyv::to_bytes::<rkyv::rancor::Error>(&b).expect("rkyv")[..]
    );
}

#[test]
fn p1_caps_round_trip_at_every_height_and_bind_their_height() {
    // 128 and 256 rows at blowup 4: trace trees of 256 and 512 row-pair leaves
    // (binary depths 8 and 9, four and five 4-ary levels), FRI layers below.
    capped_round_trip(1, 0, 2);
    capped_round_trip(2, 1, 3);
    capped_round_trip(3, 2, 4);
    capped_round_trip(4, 3, 5);
}

#[test]
fn a_cap_shortens_the_paths_by_its_height() {
    // 256 rows: the trace trees are 9 binary levels deep, five 4-ary levels,
    // the last a two-node group. Uncapped: 15 siblings an opening; at height
    // 2: 9 kept, and query 0 carries the 2·4 = 8 cap nodes.
    let o = options(4, 30, 0, ProofFormat::LEGACY);
    let open = prove_logup_at(0, 256, &o);
    let capped = prove_logup_at(2, 256, &o);
    assert!(verify_logup_at(2, &o, &capped));
    let main = |p: &LogupProof, q: usize| {
        p.deep_poly_openings[q]
            .main_trace_polys
            .proof
            .merkle_path
            .len()
    };
    assert_eq!(main(&open, 0), 15);
    assert_eq!(main(&open, 1), 15);
    assert_eq!(main(&capped, 0), 9 + 8);
    assert_eq!(main(&capped, 1), 9);
    assert!(path_nodes(&capped) < path_nodes(&open));
    // The opened values and roots are the same: a cap moves path nodes only.
    assert_eq!(
        open.lde_trace_main_merkle_root,
        capped.lde_trace_main_merkle_root
    );
    assert_eq!(open.fri_layers_merkle_roots, capped.fri_layers_merkle_roots);
}

#[test]
fn a_tampered_capped_p1_proof_is_rejected() {
    let o = options(4, 30, 0, ProofFormat::LEGACY);
    let proof = prove_logup_at(2, 256, &o);
    assert!(verify_logup_at(2, &o, &proof));
    let ok = |p: &LogupProof| verify_logup_at(2, &o, p);
    // A cap node on the owner path (query 0's main opening: 9 siblings, then
    // the 8 cap nodes), of the main, aux and composition trees.
    for at in [9usize, 12, 16] {
        let mut bad = proof.clone();
        bad.deep_poly_openings[0].main_trace_polys.proof.merkle_path[at][0] ^= 1;
        assert!(!ok(&bad), "main cap node {}", at - 9);
    }
    let mut bad = proof.clone();
    bad.deep_poly_openings[0]
        .aux_trace_polys
        .as_mut()
        .unwrap()
        .proof
        .merkle_path[10][3] ^= 1;
    assert!(!ok(&bad), "aux cap node");
    let mut bad = proof.clone();
    bad.deep_poly_openings[0].composition_poly.proof.merkle_path[11][7] ^= 1;
    assert!(!ok(&bad), "composition cap node");
    // A kept sibling of another opening; a FRI owner path's cap node.
    let mut bad = proof.clone();
    bad.deep_poly_openings[3].main_trace_polys.proof.merkle_path[4][1] ^= 1;
    assert!(!ok(&bad), "kept sibling");
    let mut bad = proof.clone();
    let fri0 = &mut bad.query_list[0].layers_auth_paths[0].merkle_path;
    let last = fri0.len() - 1;
    fri0[last][2] ^= 1;
    assert!(!ok(&bad), "FRI cap node");
    // The owner path one node short, another opening one node long, the cap
    // moved to the second opening.
    let mut bad = proof.clone();
    bad.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path
        .pop();
    assert!(!ok(&bad), "owner short");
    let mut bad = proof.clone();
    bad.deep_poly_openings[1]
        .main_trace_polys
        .proof
        .merkle_path
        .push([0u8; 32]);
    assert!(!ok(&bad), "opening long");
    let mut bad = proof.clone();
    let cap: Vec<_> = bad.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path
        .split_off(9);
    bad.deep_poly_openings[1]
        .main_trace_polys
        .proof
        .merkle_path
        .extend(cap);
    assert!(!ok(&bad), "cap moved");
}

#[test]
fn a_tampered_p1_proof_is_rejected() {
    let o = options(4, 9, 4, ProofFormat::LEGACY);
    let proof = prove_logup_at(0, 128, &o);
    let ok = |p: &LogupProof| verify_logup_at(0, &o, p);
    assert!(ok(&proof));
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
    assert!(!ok(&bad), "trace path sibling");
    // A FRI layer's path.
    let mut bad = proof.clone();
    bad.query_list[0].layers_auth_paths[0].merkle_path[1][5] ^= 1;
    assert!(!ok(&bad), "FRI path sibling");
    // A root.
    let mut bad = proof.clone();
    bad.fri_layers_merkle_roots[0][31] ^= 1;
    assert!(!ok(&bad), "FRI root");
    let mut bad = proof.clone();
    bad.lde_trace_main_merkle_root[0] ^= 1;
    assert!(!ok(&bad), "main root");
    // The nonce.
    let mut bad = proof.clone();
    bad.nonce = bad.nonce.map(|n| n ^ 1);
    assert!(!ok(&bad), "nonce");
}

// =====================================================================
// REV-P1-B's configuration-level tests (#1013 `rev_p1_b_tests`, the subset
// REV-P1-JUDGE §7 keeps that does not touch the base's statement).
// =====================================================================

fn prove_rpx(o: &ProofOptions) -> LogupProof {
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

fn verify_rpx(o: &ProofOptions, proof: &LogupProof) -> bool {
    GenericVerifier::<F, E, _, RpxStarkHash>::verify(
        proof,
        &LogReadOnlyRAP::<F, E>::new(o),
        &mut AlgebraicTranscript::new(HasherKind::Rpx),
    )
}

/// One extra zero on the symmetric main row (10 → 11 felts, still one block)
/// moves the P1 leaf hash, through the width tag in its first capacity; and
/// the verifier's AIR-derived width checks refuse the proof as before. Two
/// bindings, not one.
#[test]
fn a_zero_padded_opening_is_refused_by_the_p1_leaf_and_the_verifier() {
    let o = options(4, 30, 0, ProofFormat::LEGACY);
    let proof = prove_logup_at(0, 128, &o);
    assert!(verify_logup_at(0, &o, &proof));

    let main = &proof.deep_poly_openings[0].main_trace_polys;
    let (ev, sym) = (main.evaluations.clone(), main.evaluations_sym.clone());
    assert_eq!(ev.len() + sym.len(), 10, "five main columns, a row pair");
    let mut sym0 = sym.clone();
    sym0.push(Felt::zero());
    type P1B = P1BatchBackend<F>;
    assert_ne!(
        <P1B as IsStreamingLeafBackend<F>>::hash_data_from_slices(&ev, &sym),
        <P1B as IsStreamingLeafBackend<F>>::hash_data_from_slices(&ev, &sym0),
        "the P1 leaf hash sees the extra zero (the width tag)"
    );

    let mut bad = proof.clone();
    for q in &mut bad.deep_poly_openings {
        q.main_trace_polys.evaluations_sym.push(Felt::zero());
    }
    assert!(!verify_logup_at(0, &o, &bad), "the width checks refuse it");
}

/// The configurations do not cross-verify: a P1 proof under RPX and an RPX
/// proof under P1 are refused, and so is a P1 proof replayed on the RPX
/// transcript (the half flip), or under the P1 configuration at another cap.
#[test]
fn p1_and_rpx_proofs_do_not_cross_verify() {
    let o = options(4, 30, 4, ProofFormat::LEGACY);
    let p1 = prove_logup_under::<P1StarkHash>(128, &o);
    let rpx = prove_rpx(&o);
    assert!(verify_logup_under::<P1StarkHash>(&o, &p1));
    assert!(verify_rpx(&o, &rpx));
    assert!(!verify_rpx(&o, &p1), "P1 proof, RPX verifier");
    assert!(
        !verify_logup_under::<P1StarkHash>(&o, &rpx),
        "RPX proof, P1 verifier"
    );
    let half_flip = GenericVerifier::<F, E, _, P1StarkHash>::verify(
        &p1,
        &LogReadOnlyRAP::<F, E>::new(&o),
        &mut AlgebraicTranscript::new(HasherKind::Rpx),
    );
    assert!(!half_flip, "P1 commitments on the RPX transcript");
    assert!(
        !verify_logup_under::<P1StarkHashCapped<0>>(&o, &p1),
        "the production cap's proof under the uncapped configuration"
    );
}

/// Every cap node is bound, including the ones no query lands on: those are
/// refused only by the cap-to-root check (`verify_cap_shaped`), which the
/// capped tamper test above does not reach (its flipped nodes are all hit by
/// some opening). 256 rows at blowup 4: 512 row-pair leaves, binary depth 9,
/// five 4-ary levels; at height 4 the owner path keeps 3 siblings and carries
/// the 2·4³ = 128 cap nodes, of which 30 queries reach at most 30.
#[test]
fn every_cap_node_is_bound_by_the_cap_root() {
    let o = options(4, 30, 0, ProofFormat::LEGACY);
    let proof = prove_logup_at(4, 256, &o);
    assert!(verify_logup_at(4, &o, &proof));
    let owner = &proof.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path;
    assert_eq!(owner.len(), 3 + 128, "3 kept siblings and the 128-node cap");
    let refused = (3..owner.len())
        .filter(|&k| {
            let mut bad = proof.clone();
            bad.deep_poly_openings[0].main_trace_polys.proof.merkle_path[k][31] ^= 1;
            !verify_logup_at(4, &o, &bad)
        })
        .count();
    assert_eq!(refused, 128, "every cap node of the main tree is bound");
}

// =====================================================================
// Goldens: the host bytes of P1 proofs at the production cap.
// =====================================================================

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The production configuration's goldens: logup proofs on ZisK's transcript
/// at grinding 0 (reproducible bytes), at LDE `2^12` and `2^13` (trace trees
/// of odd and even binary depth, so the padded and the full top are both
/// committed and capped) under the legacy pair format and the production
/// format (`fri = dp`). Every case is at or above an LDE of 4096, so the card
/// commits it once `LAMBDA_VM_GPU_LDE_THRESHOLD` is lowered to 4096 or below.
pub(crate) fn p1_golden_cases() -> Vec<(String, ProofOptions, usize)> {
    let prod = crate::zf_format::ZfFormat::DEFAULT.proof_format();
    let mut out = Vec::new();
    for (fname, format) in [("legacy", ProofFormat::LEGACY), ("prod", prod)] {
        for rows in [1024usize, 2048] {
            out.push((
                format!("logup/p1c{}/{fname}/rows{rows}/blowup4", P1_CAP),
                options(4, 30, 0, format),
                rows,
            ));
        }
    }
    out
}

const P1_CAP: u8 = crate::lfm::p1_commit::P1_STARK_CAP;

/// One line per golden case: the case's name and the SHA-256 of the proof's
/// rkyv bytes.
pub(crate) fn p1_goldens() -> Vec<(String, String)> {
    p1_golden_cases()
        .into_iter()
        .map(|(name, o, rows)| {
            let proof = prove_logup_under::<P1StarkHash>(rows, &o);
            assert!(verify_logup_under::<P1StarkHash>(&o, &proof), "{name}");
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("rkyv");
            (name, hex(&bytes))
        })
        .collect()
}

/// The pins, generated by `print_p1_goldens` on the host.
pub(crate) const P1_GOLDENS: &[(&str, &str)] = &[
    (
        "logup/p1c1/legacy/rows1024/blowup4",
        "9f4eb8f28ea070fce726208eade3c38f09f010e705d05e15463ebfdf044ad7dd",
    ),
    (
        "logup/p1c1/legacy/rows2048/blowup4",
        "9bc536cb872465b8e4d6aaa8a34f6e7671eaa9da42a305f2cfa69c1546d00379",
    ),
    (
        "logup/p1c1/prod/rows1024/blowup4",
        "3810809f52f8250ac1d127380dcb5abf2c71ad4e13ad792a6f81b55e5168650e",
    ),
    (
        "logup/p1c1/prod/rows2048/blowup4",
        "4fdd8c01466e20a4ac1b33591561b75f1c96d46f2b2c9e747308ff966e61ee0a",
    ),
];

#[test]
fn p1_goldens_are_byte_identical() {
    let got = p1_goldens();
    assert_eq!(got.len(), P1_GOLDENS.len(), "one pin per case");
    for ((name, digest), (pin_name, pin)) in got.iter().zip(P1_GOLDENS) {
        assert_eq!(name, pin_name);
        assert_eq!(digest, pin, "{name}: the P1 proof's bytes moved");
    }
}

#[test]
#[ignore = "generator for P1_GOLDENS"]
fn print_p1_goldens() {
    for (name, digest) in p1_goldens() {
        println!("P1 GOLDEN (\"{name}\", \"{digest}\"),");
    }
}
