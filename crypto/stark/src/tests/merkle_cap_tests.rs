//! Merkle caps on univariate STARK proofs (lever S1).
//!
//! Every tree of a proof — main, precomputed, aux, composition, each committed
//! FRI layer — gets a height-`c` cap under a cap policy. The cap rides at the
//! end of the tree's first opening (the owner path); every path is cut to
//! `D − c` siblings. These tests pin:
//! - round trips at every policy, over the owned and the archived (rkyv) path;
//! - the default (`Off`) is byte-identical to a zero-height policy;
//! - the transcript does not move: an `Off` and an `Auto` proof of one witness
//!   differ only in their Merkle paths;
//! - tampers of every tree class, of the owner split, and of the policy;
//! - load-bearing checks at the verifier level: an unreached cap node that only the
//!   cap-to-root check rejects, and an internal node passed off as a leaf that
//!   only the exact-length check rejects.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::cap::{CapPolicy, verify_cap};
use crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::config::{Commitment, DefaultStarkHash, StarkHash};
use crate::domain::new_verifier_domain;
use crate::examples::fibonacci_2_columns::compute_trace;
use crate::examples::fibonacci_rap::{FibonacciRAP, FibonacciRAPPublicInputs, fibonacci_rap_trace};
use crate::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use crate::examples::simple_fibonacci::FibonacciPublicInputs;
use crate::merkle_caps::StarkCaps;
use crate::proof::options::ProofOptions;
use crate::proof::stark::{MultiProof, StarkProof};
use crate::proof::view::StarkProofView;
use crate::prover::{IsStarkProver, Prover};
use crate::tests::opening_width_tests::FibonacciSplitAIR;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type FE = FieldElement<F>;
type PI = SimpleAdditionPublicInputs<F>;
type Proof = StarkProof<F, F, PI>;
/// The leaf backend the default prover commits the trace trees with.
type Leaf = <DefaultStarkHash as StarkHash>::Batched<F>;

/// 1024 rows at blowup 2: trace trees 10 deep, 2 committed FRI layers.
const ROWS: usize = 1024;

fn options(policy: CapPolicy, queries: usize, blowup: u8) -> ProofOptions {
    let mut o = ProofOptions::default_test_options();
    o.blowup_factor = blowup;
    o.fri_number_of_queries = queries;
    // Grinding off: the nonce is then absent, and two proofs of one witness
    // are comparable byte for byte.
    o.grinding_factor = 0;
    o.format.merkle_cap = policy;
    o
}

fn prove(opts: &ProofOptions) -> (SimpleAdditionAIR<F>, Proof) {
    let air = SimpleAdditionAIR::<F>::new(opts);
    let pub_inputs = SimpleAdditionPublicInputs {
        a: FE::from(1u64),
        b: FE::from(2u64),
    };
    let mut trace = simple_addition_trace::<F>(ROWS);
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

fn verifies(air: &SimpleAdditionAIR<F>, proof: &Proof) -> bool {
    Verifier::verify(proof, air, &mut DefaultTranscript::<F>::new(&[]))
}

/// The same proof over the wire: rkyv, then `multi_verify_archived` (the
/// read-in-place path host continuation verification uses).
fn verifies_archived(air: &SimpleAdditionAIR<F>, proof: &Proof) -> bool {
    let multi = MultiProof {
        proofs: vec![proof.clone()],
    };
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&multi).unwrap();
    let archived = rkyv::access::<
        crate::proof::stark::ArchivedMultiProof<F, F, PI>,
        rkyv::rancor::Error,
    >(&bytes)
    .unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = PI>> = vec![air];
    Verifier::multi_verify_archived(
        &airs,
        archived,
        &mut DefaultTranscript::<F>::new(&[]),
        &FE::zero(),
    )
}

fn caps_of(air: &SimpleAdditionAIR<F>, proof: &Proof) -> StarkCaps {
    let o = air.options();
    StarkCaps::new(
        o.format.merkle_cap,
        o.fri_number_of_queries,
        (o.blowup_factor as usize * proof.trace_length).trailing_zeros() as usize,
        proof.fri_layers_merkle_roots.len(),
    )
}

/// Every path of one tree: the owner carries `D − c + 2^c` nodes (the cap at
/// its end, hashing to `root`), every other opening `D − c`.
fn assert_tree_shape(paths: &[&Vec<Commitment>], root: &Commitment, depth: usize, c: usize) {
    let owner_len = if c == 0 { depth } else { depth - c + (1 << c) };
    assert_eq!(paths[0].len(), owner_len, "owner path, D={depth} c={c}");
    for (q, p) in paths.iter().enumerate().skip(1) {
        assert_eq!(p.len(), depth - c, "query {q}, D={depth} c={c}");
    }
    if c > 0 {
        assert!(
            verify_cap::<Leaf>(&paths[0][depth - c..], root, c),
            "the owner's cap must hash to the root"
        );
    }
}

fn assert_proof_shape(air: &SimpleAdditionAIR<F>, proof: &Proof) {
    let caps = caps_of(air, proof);
    let main: Vec<_> = proof
        .deep_poly_openings
        .iter()
        .map(|o| &o.main_trace_polys.proof.merkle_path)
        .collect();
    assert_tree_shape(
        &main,
        &proof.lde_trace_main_merkle_root,
        caps.trace_depth,
        caps.trace,
    );
    let comp: Vec<_> = proof
        .deep_poly_openings
        .iter()
        .map(|o| &o.composition_poly.proof.merkle_path)
        .collect();
    assert_tree_shape(
        &comp,
        &proof.composition_poly_root,
        caps.trace_depth,
        caps.trace,
    );
    for (i, root) in proof.fri_layers_merkle_roots.iter().enumerate() {
        let layer: Vec<_> = proof
            .query_list
            .iter()
            .map(|q| &q.layers_auth_paths[i].merkle_path)
            .collect();
        assert_tree_shape(&layer, root, caps.fri_depths[i], caps.fri[i]);
    }
}

// ------------------------------------------------------------------ round trips

#[test]
fn every_policy_round_trips_owned_and_archived() {
    for blowup in [2u8, 4] {
        for (policy, queries) in [
            (CapPolicy::Fixed(1), 3),
            (CapPolicy::Fixed(2), 3),
            (CapPolicy::Fixed(3), 3),
            (CapPolicy::Fixed(4), 3),
            (CapPolicy::Auto, 3),
            (CapPolicy::Auto, 8),
            (CapPolicy::Auto, 30),
        ] {
            let (air, proof) = prove(&options(policy, queries, blowup));
            assert!(
                proof.fri_layers_merkle_roots.len() >= 2,
                "the FRI arm must commit layers"
            );
            let caps = caps_of(&air, &proof);
            if policy != CapPolicy::Auto || queries >= 4 {
                assert!(caps.any(), "{policy} Q={queries}: some tree must be capped");
            }
            assert_proof_shape(&air, &proof);
            assert!(
                verifies(&air, &proof),
                "{policy} Q={queries} blowup {blowup}"
            );
            assert!(
                verifies_archived(&air, &proof),
                "{policy} Q={queries} blowup {blowup}: archived"
            );
        }
    }
}

/// A preprocessed table (precomputed + main trees) and a RAP table (main + aux
/// trees) round-trip under a cap, and a cap node flip in each of those trees is
/// rejected.
#[test]
fn preprocessed_and_aux_trees_are_capped() {
    // Preprocessed: 1 precomputed column, 1 main column, 1024 rows.
    let opts = options(CapPolicy::Fixed(3), 3, 2);
    let mut trace = compute_trace([FE::one(), FE::one()], ROWS);
    let reference = FibonacciSplitAIR::<F>::honest(&opts, None);
    let commitment = Prover::compute_precomputed_commitment_for_testing(&trace, &reference, 1)
        .expect("precomputed commitment");
    let air = FibonacciSplitAIR::<F>::preprocessed_declaring(&opts, None, 1, commitment);
    let pi = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };
    let proof =
        Prover::prove(&air, &mut trace, &pi, &mut DefaultTranscript::<F>::new(&[])).expect("prove");
    let verify = |p: &StarkProof<F, F, FibonacciPublicInputs<F>>| {
        Verifier::verify(p, &air, &mut DefaultTranscript::<F>::new(&[]))
    };
    assert!(verify(&proof), "capped preprocessed proof");
    let depth = 10;
    let pre = proof.deep_poly_openings[0]
        .precomputed_trace_polys
        .as_ref()
        .expect("precomputed opening");
    assert_eq!(pre.proof.merkle_path.len(), depth - 3 + 8);
    for k in 0..8 {
        let mut bad = proof.clone();
        bad.deep_poly_openings[0]
            .precomputed_trace_polys
            .as_mut()
            .unwrap()
            .proof
            .merkle_path[depth - 3 + k][0] ^= 1;
        assert!(!verify(&bad), "precomputed cap node {k}");
    }

    // RAP: 2 main + 1 aux column, 16 steps (the AIR's constraints are fixed to
    // 16 steps), capped at 3.
    let opts = options(CapPolicy::Fixed(3), 3, 2);
    let mut trace = fibonacci_rap_trace([FE::one(), FE::one()], 16);
    let air = FibonacciRAP::<F>::new(&opts);
    let pi = FibonacciRAPPublicInputs {
        steps: 16,
        a0: FE::one(),
        a1: FE::one(),
    };
    let proof =
        Prover::prove(&air, &mut trace, &pi, &mut DefaultTranscript::<F>::new(&[])).expect("prove");
    let verify = |p: &StarkProof<F, F, FibonacciRAPPublicInputs<F>>| {
        Verifier::verify(p, &air, &mut DefaultTranscript::<F>::new(&[]))
    };
    assert!(verify(&proof), "capped RAP proof");
    let depth = StarkCaps::trace_tree_depth((2 * proof.trace_length).trailing_zeros() as usize);
    assert!(depth >= 3);
    let aux = proof.deep_poly_openings[0]
        .aux_trace_polys
        .as_ref()
        .expect("aux opening");
    assert_eq!(aux.proof.merkle_path.len(), depth - 3 + 8);
    assert_eq!(
        proof.deep_poly_openings[1]
            .aux_trace_polys
            .as_ref()
            .unwrap()
            .proof
            .merkle_path
            .len(),
        depth - 3
    );
    for k in 0..8 {
        let mut bad = proof.clone();
        bad.deep_poly_openings[0]
            .aux_trace_polys
            .as_mut()
            .unwrap()
            .proof
            .merkle_path[depth - 3 + k][5] ^= 0x40;
        assert!(!verify(&bad), "aux cap node {k}");
    }
}

// ------------------------------------------------------------ default identity

/// `Off`, `Fixed(0)` and a policy whose every height is 0 (`Auto` at 3
/// queries) produce the same bytes: the default format is unchanged.
#[test]
fn a_zero_height_policy_is_byte_identical_to_off() {
    let bytes = |policy| {
        let (air, proof) = prove(&options(policy, 3, 2));
        assert!(!caps_of(&air, &proof).any());
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .unwrap()
            .to_vec()
    };
    let off = bytes(CapPolicy::Off);
    assert_eq!(off, bytes(CapPolicy::Fixed(0)));
    assert_eq!(off, bytes(CapPolicy::Auto));
}

/// The transcript does not change under a cap. One witness
/// proved at `Off` and at `Auto` (grinding off) gives equal roots, OOD values,
/// FRI final coefficients, nonces and opened values; only the Merkle paths
/// differ, and each capped path is exactly its full path cut to `D − c`, with
/// the cap appended on the owner.
#[test]
fn the_transcript_is_the_same_with_and_without_a_cap() {
    let (_, off) = prove(&options(CapPolicy::Off, 30, 2));
    let (air, on) = prove(&options(CapPolicy::Auto, 30, 2));
    let caps = caps_of(&air, &on);
    assert_eq!(caps.trace, 3);
    assert!(caps.fri.iter().all(|&c| c == 3));

    assert_eq!(
        off.lde_trace_main_merkle_root,
        on.lde_trace_main_merkle_root
    );
    assert_eq!(off.lde_trace_aux_merkle_root, on.lde_trace_aux_merkle_root);
    assert_eq!(
        off.lde_trace_precomputed_merkle_root,
        on.lde_trace_precomputed_merkle_root
    );
    assert_eq!(off.composition_poly_root, on.composition_poly_root);
    assert_eq!(off.fri_layers_merkle_roots, on.fri_layers_merkle_roots);
    assert_eq!(off.trace_ood_evaluations, on.trace_ood_evaluations);
    assert_eq!(
        off.trace_ood_next_evaluations,
        on.trace_ood_next_evaluations
    );
    assert_eq!(
        off.composition_poly_parts_ood_evaluation,
        on.composition_poly_parts_ood_evaluation
    );
    assert_eq!(off.fri_final_poly_coeffs, on.fri_final_poly_coeffs);
    assert_eq!(off.nonce, on.nonce);
    assert_eq!(off.trace_length, on.trace_length);

    let cut = |full: &Vec<Commitment>, capped: &Vec<Commitment>, q: usize, d: usize, c: usize| {
        assert_eq!(full.len(), d);
        assert_eq!(&capped[..d - c], &full[..d - c], "query {q}: the siblings");
        let tail = if q == 0 { 1usize << c } else { 0 };
        assert_eq!(capped.len(), d - c + tail, "query {q}");
    };
    let d = caps.trace_depth;
    for (q, (a, b)) in off
        .deep_poly_openings
        .iter()
        .zip(&on.deep_poly_openings)
        .enumerate()
    {
        assert_eq!(
            a.main_trace_polys.evaluations,
            b.main_trace_polys.evaluations
        );
        assert_eq!(
            a.main_trace_polys.evaluations_sym,
            b.main_trace_polys.evaluations_sym
        );
        assert_eq!(
            a.composition_poly.evaluations,
            b.composition_poly.evaluations
        );
        assert_eq!(
            a.composition_poly.evaluations_sym,
            b.composition_poly.evaluations_sym
        );
        cut(
            &a.main_trace_polys.proof.merkle_path,
            &b.main_trace_polys.proof.merkle_path,
            q,
            d,
            3,
        );
        cut(
            &a.composition_poly.proof.merkle_path,
            &b.composition_poly.proof.merkle_path,
            q,
            d,
            3,
        );
    }
    for (q, (a, b)) in off.query_list.iter().zip(&on.query_list).enumerate() {
        assert_eq!(a.layers_evaluations_sym, b.layers_evaluations_sym);
        for i in 0..caps.fri.len() {
            cut(
                &a.layers_auth_paths[i].merkle_path,
                &b.layers_auth_paths[i].merkle_path,
                q,
                caps.fri_depths[i],
                caps.fri[i],
            );
        }
    }
}

// --------------------------------------------------------------------- tampers

type PathOf = fn(&mut Proof) -> &mut Vec<Commitment>;
type PathFn = dyn Fn(&mut Proof) -> &mut Vec<Commitment>;

fn main_path(q: usize) -> impl Fn(&mut Proof) -> &mut Vec<Commitment> {
    move |p| &mut p.deep_poly_openings[q].main_trace_polys.proof.merkle_path
}
fn comp_path(q: usize) -> impl Fn(&mut Proof) -> &mut Vec<Commitment> {
    move |p| &mut p.deep_poly_openings[q].composition_poly.proof.merkle_path
}
fn fri_path(q: usize, layer: usize) -> impl Fn(&mut Proof) -> &mut Vec<Commitment> {
    move |p| &mut p.query_list[q].layers_auth_paths[layer].merkle_path
}

fn rejected_after(
    air: &SimpleAdditionAIR<F>,
    honest: &Proof,
    tamper: impl FnOnce(&mut Proof),
) -> bool {
    let mut p = honest.clone();
    tamper(&mut p);
    !verifies(air, &p) && !verifies_archived(air, &p)
}

#[test]
fn every_cap_node_of_every_tree_class_is_bound() {
    let (air, honest) = prove(&options(CapPolicy::Auto, 30, 2));
    assert!(verifies(&air, &honest));
    let caps = caps_of(&air, &honest);
    let last = honest.fri_layers_merkle_roots.len() - 1;
    let trees: Vec<(&str, Box<PathFn>, usize, usize)> = vec![
        ("main", Box::new(main_path(0)), caps.trace_depth, caps.trace),
        (
            "composition",
            Box::new(comp_path(0)),
            caps.trace_depth,
            caps.trace,
        ),
        (
            "FRI layer 0",
            Box::new(fri_path(0, 0)),
            caps.fri_depths[0],
            caps.fri[0],
        ),
        (
            "last FRI layer",
            Box::new(fri_path(0, last)),
            caps.fri_depths[last],
            caps.fri[last],
        ),
    ];
    for (what, path_of, d, c) in &trees {
        assert_eq!(*c, 3, "{what}");
        for k in 0..(1usize << c) {
            assert!(
                rejected_after(&air, &honest, |p| path_of(p)[d - c + k][7] ^= 1),
                "{what}: cap node {k} flipped"
            );
        }
    }
}

#[test]
fn a_path_node_of_a_later_query_is_bound() {
    let (air, honest) = prove(&options(CapPolicy::Auto, 30, 2));
    let paths: [(&str, PathOf); 3] = [
        ("main", |p| {
            &mut p.deep_poly_openings[5].main_trace_polys.proof.merkle_path
        }),
        ("composition", |p| {
            &mut p.deep_poly_openings[5].composition_poly.proof.merkle_path
        }),
        ("FRI layer 1", |p| {
            &mut p.query_list[5].layers_auth_paths[1].merkle_path
        }),
    ];
    for (what, path_of) in paths {
        let len = path_of(&mut honest.clone()).len();
        for k in 0..len {
            assert!(
                rejected_after(&air, &honest, |p| path_of(p)[k][0] ^= 0x10),
                "{what}: query 5 node {k}"
            );
        }
    }
}

#[test]
fn the_owner_split_is_exact() {
    let (air, honest) = prove(&options(CapPolicy::Auto, 30, 2));
    // The owner path one node short (the last cap node dropped) or long.
    assert!(rejected_after(&air, &honest, |p| {
        main_path(0)(p).pop();
    }));
    assert!(rejected_after(&air, &honest, |p| {
        let path = main_path(0)(p);
        path.push(path[0]);
    }));
    // A non-owner path carrying the cap too.
    assert!(rejected_after(&air, &honest, |p| {
        let cap: Vec<Commitment> = main_path(0)(p)[7..].to_vec();
        main_path(1)(p).extend(cap);
    }));
    // The cap moved from query 0 to query 1.
    assert!(rejected_after(&air, &honest, |p| {
        let cap: Vec<Commitment> = main_path(0)(p).split_off(7);
        main_path(1)(p).extend(cap);
    }));
    // The same for a FRI layer.
    assert!(rejected_after(&air, &honest, |p| {
        let path = fri_path(0, 0);
        let d = path(p).len() - 8;
        let cap: Vec<Commitment> = path(p).split_off(d);
        fri_path(1, 0)(p).extend(cap);
    }));
}

/// The cap height is a verifier constant: a proof made under one policy fails
/// under any other, in both directions.
#[test]
fn a_proof_made_under_one_policy_fails_under_another() {
    let (_, fixed3) = prove(&options(CapPolicy::Fixed(3), 30, 2));
    let (air_off, off) = prove(&options(CapPolicy::Off, 30, 2));
    let air_at = |policy| SimpleAdditionAIR::<F>::new(&options(policy, 30, 2));
    assert!(verifies(&air_at(CapPolicy::Fixed(3)), &fixed3));
    assert!(!verifies(&air_at(CapPolicy::Fixed(2)), &fixed3));
    assert!(!verifies(&air_at(CapPolicy::Fixed(4)), &fixed3));
    assert!(!verifies(&air_off, &fixed3));
    assert!(!verifies(&air_at(CapPolicy::Auto), &off));
    assert!(verifies(&air_off, &off));
}

// ---------------------------------------------------- M1 at the verifier level

/// The transcript's index of query `q` of the main tree, recovered from its
/// capped opening (the only index whose fold lands on the cap), and the cap
/// node it reaches.
fn main_query_index(proof: &Proof, q: usize, d: usize, c: usize) -> (usize, usize) {
    let owner = &proof.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path;
    let cap = &owner[d - c..];
    let opening = &proof.deep_poly_openings[q].main_trace_polys;
    let siblings = &opening.proof.merkle_path[..d - c];
    let leaf = Leaf::hash_data_from_slices(&opening.evaluations, &opening.evaluations_sym);
    let hits: Vec<usize> = (0..1usize << d)
        .filter(|&i| {
            verify_merkle_path_from_leaf_hash::<Leaf>(siblings, &cap[i >> (d - c)], i, leaf)
        })
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "query {q}: exactly one index folds onto the cap"
    );
    (hits[0], hits[0] >> (d - c))
}

/// M1(b): with 3 queries and a height-3 cap, at least 5 of the 8 main-tree cap
/// nodes are reached by no query. Flipping one leaves every per-query check
/// green (each query still folds onto its own, unchanged, cap node), so only
/// the cap-to-root check rejects the proof. Deleting `verify_cap` from
/// `CappedRoot::from_owner` makes this test fail.
#[test]
fn an_unreached_cap_node_is_rejected_by_the_cap_to_root_check_alone() {
    let (air, honest) = prove(&options(CapPolicy::Fixed(3), 3, 2));
    let (d, c) = (10, 3);
    assert!(verifies(&air, &honest));
    let reached: Vec<usize> = (0..3)
        .map(|q| main_query_index(&honest, q, d, c).1)
        .collect();
    let unreached: Vec<usize> = (0..8).filter(|k| !reached.contains(k)).collect();
    assert!(unreached.len() >= 5, "3 queries reach at most 3 of 8 nodes");
    for k in unreached {
        let mut bad = honest.clone();
        main_path(0)(&mut bad)[d - c + k][3] ^= 1;
        // Precondition: the per-query folds are untouched by the flip.
        for (q, &node) in reached.iter().enumerate() {
            assert_eq!(main_query_index(&bad, q, d, c).1, node);
        }
        assert!(!verifies(&air, &bad), "unreached cap node {k}");
        assert!(
            !verifies_archived(&air, &bad),
            "unreached cap node {k}: archived"
        );
    }
}

/// M1(a): the verifier's own per-tree check (`table_tree_checks`) refuses the
/// real internal node one level above a queried leaf, presented as a leaf hash
/// with the path from that node up — which the length-agnostic fold accepts.
/// Only the exact-length check stands between the two; deleting it from the
/// cap primitive makes this test fail. Run at the default (`c = 0`) and
/// under a cap.
#[test]
fn an_internal_node_passed_as_a_leaf_is_rejected_by_the_length_check_alone() {
    for policy in [CapPolicy::Off, CapPolicy::Fixed(3)] {
        let (air, proof) = prove(&options(policy, 3, 2));
        let c = if policy == CapPolicy::Off { 0 } else { 3 };
        let d = 10;
        let view = StarkProofView::Owned(&proof);
        let domain = new_verifier_domain(&air, proof.trace_length);
        let checks = Verifier::table_tree_checks(&air, view, &domain).expect("honest shape");
        for q in 1..3 {
            let iota = if c == 0 {
                // Uncapped: fold against the root directly.
                let opening = &proof.deep_poly_openings[q].main_trace_polys;
                let leaf =
                    Leaf::hash_data_from_slices(&opening.evaluations, &opening.evaluations_sym);
                (0..1usize << d)
                    .find(|&i| {
                        verify_merkle_path_from_leaf_hash::<Leaf>(
                            &opening.proof.merkle_path,
                            &proof.lde_trace_main_merkle_root,
                            i,
                            leaf,
                        )
                    })
                    .expect("the honest index")
            } else {
                main_query_index(&proof, q, d, c).0
            };
            let opening = &proof.deep_poly_openings[q].main_trace_polys;
            let path = &opening.proof.merkle_path;
            let leaf = Leaf::hash_data_from_slices(&opening.evaluations, &opening.evaluations_sym);
            // The real node one level up, and the position it sits at.
            let node = if iota & 1 == 0 {
                Leaf::hash_new_parent(&leaf, &path[0])
            } else {
                Leaf::hash_new_parent(&path[0], &leaf)
            };
            let forged = &path[1..];
            let target = if c == 0 {
                proof.lde_trace_main_merkle_root
            } else {
                proof.deep_poly_openings[0]
                    .main_trace_polys
                    .proof
                    .merkle_path[d - c + (iota >> (d - c))]
            };
            assert!(
                verify_merkle_path_from_leaf_hash::<Leaf>(forged, &target, iota >> 1, node),
                "{policy} q={q}: precondition, the fold alone accepts the forgery"
            );
            assert!(
                !checks.main.verify::<Leaf>(q, forged, iota >> 1, node),
                "{policy} q={q}: an internal node passed for a leaf"
            );
            // The honest opening passes the same check.
            assert!(checks.main.verify::<Leaf>(q, path, iota, leaf));
        }
    }
}

// ------------------------------------------------------------- device trees

/// A device-resident tree (a root-only host tree) whose cap has
/// no device read is a hard `Err` naming the tree — never a skipped cap, which
/// would ship full-length paths the verifier rejects with no pointer to the
/// cause. And a device read that fails is an `Err` too, not a panic.
#[test]
fn a_device_resident_tree_without_a_cap_read_is_an_error() {
    use crate::prover::ProvingError;
    use crypto::merkle_tree::merkle::MerkleTree;
    type P = Prover<F, F, PI>;
    let root_only = MerkleTree::<Leaf>::from_root([7u8; 32]);
    match <P as IsStarkProver<F, F, PI, DefaultStarkHash>>::tree_cap(
        &root_only,
        10,
        3,
        "main",
        |_| None,
    ) {
        Err(ProvingError::DevicePath(msg)) => {
            assert!(
                msg.contains("main") && msg.contains("device-resident"),
                "{msg}"
            )
        }
        other => panic!("expected a DevicePath error, got {other:?}"),
    }
    match <P as IsStarkProver<F, F, PI, DefaultStarkHash>>::tree_cap(
        &root_only,
        10,
        3,
        "FRI layer 2",
        |_| Some(Err("cudarc said no".to_string())),
    ) {
        Err(ProvingError::DevicePath(msg)) => {
            assert!(
                msg.contains("FRI layer 2") && msg.contains("cudarc said no"),
                "{msg}"
            )
        }
        other => panic!("expected a DevicePath error, got {other:?}"),
    }
    // A host tree whose depth is not the format's is refused too.
    let data: Vec<Vec<FE>> = (0..16u64)
        .map(|i| vec![FE::from(i), FE::from(i + 1)])
        .collect();
    let host = MerkleTree::<Leaf>::build(&data).expect("tree");
    assert!(
        <P as IsStarkProver<F, F, PI, DefaultStarkHash>>::tree_cap(&host, 5, 2, "aux", |_| None)
            .is_err()
    );
    let cap =
        <P as IsStarkProver<F, F, PI, DefaultStarkHash>>::tree_cap(&host, 4, 2, "aux", |_| None)
            .expect("a full host tree serves its cap");
    assert!(verify_cap::<Leaf>(&cap, &host.root, 2));
}

/// The device cap read on a real device (box only; `--features cuda -- --ignored`): a LogUp
/// table over the cubic extension, big enough that its main, aux,
/// composition and FRI trees are committed on the device (host trees
/// root-only), proved under `Auto` at 30 queries. The caps must come off the
/// device (`gpu_cap_read_calls` moves), the proof must verify owned and
/// archived, and against an `Off` proof of the same witness the transcript is
/// unchanged and every capped path is the full device-gathered path cut to
/// `D − c` (the cap on the owner).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a GPU; run with --features cuda -- --ignored"]
fn device_trees_serve_their_caps() {
    // An `AirWithBuses` table: the device composition arm needs the AIR's
    // constraint program, which the hand-written example AIRs do not supply
    // (`LogReadOnlyRAP` here panicked in `constraint_program` on the box).
    use crate::examples::bus_permutation::{bus_permutation_air, bus_permutation_trace};
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as E;
    type Pi = ();

    let rows = 1usize << 14;
    let prove_at = |policy| {
        let opts = options(policy, 30, 2);
        let mut trace = bus_permutation_trace(rows);
        let air = bus_permutation_air(&opts);
        let proof = Prover::prove(&air, &mut trace, &(), &mut DefaultTranscript::<E>::new(&[]))
            .expect("prove");
        (air, proof)
    };

    let (_, off) = prove_at(CapPolicy::Off);
    let before = crate::gpu_lde::gpu_cap_read_calls();
    let (air, on) = prove_at(CapPolicy::Auto);
    let reads = crate::gpu_lde::gpu_cap_read_calls() - before;
    println!("CAPDEV device cap reads: {reads}");
    assert!(
        reads > 0,
        "no cap came off the device: the trees were host trees, the test proves nothing"
    );
    assert!(
        Verifier::verify(&on, &air, &mut DefaultTranscript::<E>::new(&[])),
        "a device-proved capped proof must verify"
    );
    let multi = MultiProof {
        proofs: vec![on.clone()],
    };
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&multi).unwrap();
    let archived = rkyv::access::<
        crate::proof::stark::ArchivedMultiProof<F, E, Pi>,
        rkyv::rancor::Error,
    >(&bytes)
    .unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = Pi>> = vec![&air];
    assert!(Verifier::multi_verify_archived(
        &airs,
        archived,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::<E>::zero(),
    ));

    assert_eq!(
        off.lde_trace_main_merkle_root,
        on.lde_trace_main_merkle_root
    );
    assert_eq!(off.lde_trace_aux_merkle_root, on.lde_trace_aux_merkle_root);
    assert_eq!(off.composition_poly_root, on.composition_poly_root);
    assert_eq!(off.fri_layers_merkle_roots, on.fri_layers_merkle_roots);
    assert_eq!(off.fri_final_poly_coeffs, on.fri_final_poly_coeffs);
    let lde_log = (2 * on.trace_length).trailing_zeros() as usize;
    let caps = StarkCaps::new(
        CapPolicy::Auto,
        30,
        lde_log,
        on.fri_layers_merkle_roots.len(),
    );
    assert_eq!(caps.trace, 3);
    let check = |full: &Vec<Commitment>, capped: &Vec<Commitment>, q: usize, d: usize, c: usize| {
        assert_eq!(full.len(), d, "query {q}: full path");
        assert_eq!(&capped[..d - c], &full[..d - c], "query {q}: siblings");
        assert_eq!(
            capped.len(),
            d - c + if q == 0 { 1 << c } else { 0 },
            "query {q}"
        );
    };
    let d = caps.trace_depth;
    for (q, (a, b)) in off
        .deep_poly_openings
        .iter()
        .zip(&on.deep_poly_openings)
        .enumerate()
    {
        check(
            &a.main_trace_polys.proof.merkle_path,
            &b.main_trace_polys.proof.merkle_path,
            q,
            d,
            3,
        );
        check(
            &a.composition_poly.proof.merkle_path,
            &b.composition_poly.proof.merkle_path,
            q,
            d,
            3,
        );
        let (aa, bb) = (
            a.aux_trace_polys.as_ref().unwrap(),
            b.aux_trace_polys.as_ref().unwrap(),
        );
        check(&aa.proof.merkle_path, &bb.proof.merkle_path, q, d, 3);
    }
    for (q, (a, b)) in off.query_list.iter().zip(&on.query_list).enumerate() {
        for i in 0..caps.fri.len() {
            check(
                &a.layers_auth_paths[i].merkle_path,
                &b.layers_auth_paths[i].merkle_path,
                q,
                caps.fri_depths[i],
                caps.fri[i],
            );
        }
    }
}
