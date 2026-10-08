//! STARK round trips under ZisK's Poseidon1 configuration (`P1StarkHash`:
//! arity-4 trees, ZisK's leaf hash, width-8 grinding), on the host, and the
//! tamper checks that make the verifier's new path walk load-bearing; the
//! arity-4 Merkle caps at fixed heights, set by the format's `base`.

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

use stark::config::StarkHash;
use stark::proof::options::{BaseFormat, CapPolicy};

use crate::lfm::p1_commit::{P1StarkHash, P1Transcript};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;

/// `format` with its base the P1 base at 4-ary cap height `c` (0: uncapped).
fn at(c: u8, format: ProofFormat) -> ProofFormat {
    ProofFormat {
        base: BaseFormat {
            arity4_cap: if c == 0 {
                CapPolicy::Off
            } else {
                CapPolicy::Fixed(c)
            },
            ..BaseFormat::P1
        },
        ..format
    }
}

/// [`prove_logup_under`] under P1 with the options' format at cap height `c`.
fn prove_logup_at(
    c: u8,
    rows: usize,
    o: &ProofOptions,
) -> (
    LogReadOnlyRAP<F, E>,
    StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) {
    let o = ProofOptions {
        format: at(c, o.format),
        ..o.clone()
    };
    prove_logup_under::<P1StarkHash>(rows, &o)
}

/// The P1 verifier of a logup proof whose format's cap height is `c`: the
/// verifier's own AIR, at its own format.
fn verify_logup_at(
    c: u8,
    o: &ProofOptions,
    proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    let o = ProofOptions {
        format: at(c, o.format),
        ..o.clone()
    };
    verify_logup_under::<P1StarkHash>(&LogReadOnlyRAP::<F, E>::new(&o), proof)
}

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
    prove_logup_under::<P1StarkHash>(rows, o)
}

fn prove_logup_under<H: StarkHash>(
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
    let proof = GenericProver::<F, E, _, H>::prove(&air, &mut trace, &pi, &mut P1Transcript::new())
        .expect("proving must succeed");
    (air, proof)
}

fn verify_logup(
    air: &LogReadOnlyRAP<F, E>,
    proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    verify_logup_under::<P1StarkHash>(air, proof)
}

fn verify_logup_under<H: StarkHash>(
    air: &LogReadOnlyRAP<F, E>,
    proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    GenericVerifier::<F, E, _, H>::verify(proof, air, &mut P1Transcript::new())
}

/// Every Merkle path node a proof carries (trace, composition and FRI
/// openings): what a cap changes.
fn path_nodes(proof: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>) -> usize {
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
            let (_, proof) = prove_logup_at(c, rows, &o);
            assert!(verify_logup_at(c, &o, &proof), "{name} rows {rows} c {c}");
            assert!(
                !verify_logup_at(lo, &o, &proof),
                "{name} rows {rows} c {c} by {lo}"
            );
            // A height at or past every tree's levels clamps to them: the
            // next height up is the same format there.
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
    let (_, a) = prove_logup_at(0, 128, &oa);
    let (_, b) = prove_logup_at(0, 128, &ol);
    assert!(verify_logup_at(0, &oa, &a));
    assert_eq!(
        a.deep_poly_openings[0].main_trace_polys.proof.merkle_path,
        b.deep_poly_openings[0].main_trace_polys.proof.merkle_path
    );
    let (_, a) = prove_logup_at(2, 128, &oa);
    let (_, b) = prove_logup_at(2, 128, &ol);
    assert_eq!(
        a.deep_poly_openings[1].main_trace_polys.proof.merkle_path,
        b.deep_poly_openings[1].main_trace_polys.proof.merkle_path
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
    let (_, open) = prove_logup_at(0, 256, &o);
    let (_, capped) = prove_logup_at(2, 256, &o);
    assert!(verify_logup_at(2, &o, &capped));
    let main = |p: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>, q: usize| {
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
    let (_, proof) = prove_logup_at(2, 256, &o);
    assert!(verify_logup_at(2, &o, &proof));
    let ok = |p: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>| verify_logup_at(2, &o, p);
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
    let (air, proof) = prove_logup_at(0, 128, &o);
    // The verifier's own AIR at the proof's format (the prover's `air` is the
    // same), at cap height 0.
    let verify_logup = |_: &LogReadOnlyRAP<F, E>,
                        p: &StarkProof<F, E, LogReadOnlyPublicInputs<F>>| {
        verify_logup_at(0, &o, p)
    };
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
