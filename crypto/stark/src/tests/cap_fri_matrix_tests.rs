//! Merkle caps (S1) composed with group-leaf FRI layers (S3) on the host path:
//! a round-trip matrix {cap off, fixed, auto} × {pair, dp,
//! dp with an uneven override}, at a query count where `auto` caps (Q ≥ 20).
//!
//! Under a fold schedule a committed FRI layer is a GROUP tree whose depth is
//! the layout's (`FriFoldLayout::layer_depth`), not today's
//! `log2(lde) − j − 2`. The cap of each layer is taken at that depth, by the
//! prover (`StarkCaps::from_layout` in round 4) and by the verifier (the
//! per-tree `TreeCheck` the group path authenticates with). These tests pin:
//! - every cell of the matrix proves and verifies, owned and archived;
//! - every FRI layer's paths have the capped shape at the LAYOUT's depth,
//!   computed here independently from the schedule;
//! - every cap node of a capped group layer is bound, and an unreached one is
//!   rejected by the cap-to-root check alone (on a group tree);
//! - a proof made under one (cap, fri) format fails under the others.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::cap::{CapPolicy, verify_cap};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::config::{Commitment, DefaultStarkHash, StarkHash};
use crate::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use crate::fri::capture::{FriCapture, capture};
use crate::fri::schedule::FriFormat;
use crate::merkle_caps::StarkCaps;
use crate::proof::options::{FriMode, FriScheduleOverride, ProofFormat, ProofOptions};
use crate::proof::stark::{MultiProof, StarkProof};
use crate::prover::{IsStarkProver, Prover};
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type FE = FieldElement<F>;
type PI = SimpleAdditionPublicInputs<F>;
type Proof = StarkProof<F, F, PI>;
/// The leaf backend the FRI layer trees are verified with (the FRI values
/// live in the proof's extension, which is `F` for this AIR).
type Leaf = <DefaultStarkHash as StarkHash>::Batched<F>;

/// 1024 rows at blowup 2 with `k = 2`: LDE 2^11, terminal 2^3, so the
/// committed chain covers 10 → 3 (seven bits).
const ROWS: usize = 1024;
const LDE_LOG: u32 = 11;
const TERMINAL_LOG: u32 = 3;
/// `auto` caps at height 3 from 20 openings on.
const QUERIES: usize = 24;

fn options(cap: CapPolicy, fri: FriMode, over: Option<&[u8]>, queries: usize) -> ProofOptions {
    let mut o = ProofOptions::default_test_options();
    o.blowup_factor = 2;
    o.fri_number_of_queries = queries;
    o.grinding_factor = 0;
    o.fri_final_poly_log_degree = 2;
    o.format = ProofFormat {
        merkle_cap: cap,
        fri_mode: fri,
        fri_schedule_override: over.and_then(FriScheduleOverride::new),
        ..ProofFormat::DEFAULT
    };
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

/// The committed layers' tree depths, from the schedule alone: the chain
/// starts at `lde_log − 1` and layer `j`'s tree is its length over `2^{d_j}`
/// leaves. Independent of `FriFoldLayout::layer_depth`.
fn schedule_depths(opts: &ProofOptions) -> (Vec<u8>, Vec<usize>) {
    let schedule = FriFormat::from_options(opts, false).schedule(LDE_LOG, TERMINAL_LOG);
    let mut b = LDE_LOG as usize - 1;
    let depths = schedule
        .iter()
        .map(|&d| {
            b -= d as usize;
            b
        })
        .collect();
    (schedule, depths)
}

fn fri_paths(proof: &Proof, layer: usize) -> Vec<&Vec<Commitment>> {
    proof
        .query_list
        .iter()
        .map(|q| &q.layers_auth_paths[layer].merkle_path)
        .collect()
}

const FORMATS: [(&str, FriMode, Option<&[u8]>); 3] = [
    ("pair", FriMode::Pair, None),
    ("dp", FriMode::Dp, None),
    ("dp [3,1,3]", FriMode::Dp, Some(&[3, 1, 3])),
];

#[test]
fn the_cap_and_fri_matrix_round_trips_owned_and_archived() {
    for cap in [CapPolicy::Off, CapPolicy::Fixed(2), CapPolicy::Auto] {
        for (name, fri, over) in FORMATS {
            let opts = options(cap, fri, over, QUERIES);
            let (air, proof) = prove(&opts);
            let (schedule, depths) = schedule_depths(&opts);
            let caps = StarkCaps::for_options(&opts, LDE_LOG as usize, false).expect("layout");
            assert_eq!(
                caps.fri_depths, depths,
                "cap={cap} fri={name}: the caps' FRI depths are the layout's"
            );
            assert_eq!(proof.fri_layers_merkle_roots.len(), schedule.len());
            for (j, root) in proof.fri_layers_merkle_roots.iter().enumerate() {
                let (d, c) = (depths[j], caps.fri[j]);
                assert_eq!(c, cap.height(QUERIES, d), "cap={cap} fri={name} layer {j}");
                let paths = fri_paths(&proof, j);
                let owner = if c == 0 { d } else { d - c + (1 << c) };
                assert_eq!(
                    paths[0].len(),
                    owner,
                    "cap={cap} fri={name} layer {j} owner"
                );
                for p in &paths[1..] {
                    assert_eq!(p.len(), d - c, "cap={cap} fri={name} layer {j}");
                }
                if c > 0 {
                    assert!(
                        verify_cap::<Leaf>(&paths[0][d - c..], root, c),
                        "cap={cap} fri={name} layer {j}: the owner's cap hashes to the root"
                    );
                }
            }
            if cap != CapPolicy::Off {
                assert!(
                    caps.fri.iter().any(|&c| c > 0),
                    "cap={cap} fri={name}: some FRI layer must be capped"
                );
            }
            assert!(verifies(&air, &proof), "cap={cap} fri={name}");
            assert!(
                verifies_archived(&air, &proof),
                "cap={cap} fri={name}: archived"
            );
        }
    }
}

#[test]
fn every_cap_node_of_a_capped_group_layer_is_bound() {
    let opts = options(CapPolicy::Auto, FriMode::Dp, Some(&[3, 1, 3]), QUERIES);
    let (air, honest) = prove(&opts);
    assert!(verifies(&air, &honest));
    let (_, depths) = schedule_depths(&opts);
    // Layer 0 folds by 8 (depth 7) and layer 2 by 8 (depth 3): both capped at 3.
    for j in [0usize, 2] {
        let (d, c) = (depths[j], CapPolicy::Auto.height(QUERIES, depths[j]));
        assert_eq!(c, 3, "layer {j}");
        for k in 0..(1usize << c) {
            let mut bad = honest.clone();
            bad.query_list[0].layers_auth_paths[j].merkle_path[d - c + k][5] ^= 1;
            assert!(
                !verifies(&air, &bad) && !verifies_archived(&air, &bad),
                "group layer {j}: cap node {k} flipped"
            );
        }
        // A later query's path, cut at the cap (layer 2's tree is all cap:
        // depth 3 at c = 3, so its paths are empty).
        assert_eq!(
            honest.query_list[7].layers_auth_paths[j].merkle_path.len(),
            d - c
        );
        if d > c {
            let mut bad = honest.clone();
            bad.query_list[7].layers_auth_paths[j].merkle_path[0][0] ^= 1;
            assert!(!verifies(&air, &bad), "group layer {j}: query 7 sibling");
        }
    }
}

/// An unreached cap node on a group tree: with three queries and a height-3 cap on
/// FRI layer 0, at least five of its eight cap nodes are reached by no query.
/// Flipping one leaves every per-query fold untouched (each still lands on its
/// own cap node), so only the cap-to-root check of the group layer's
/// `TreeCheck` rejects the proof.
#[test]
fn an_unreached_cap_node_of_a_group_layer_is_rejected_by_the_cap_to_root_check_alone() {
    let opts = options(CapPolicy::Fixed(3), FriMode::Dp, Some(&[3, 1, 3]), 3);
    let (air, honest) = prove(&opts);
    let (ok, records) =
        capture(|| Verifier::verify(&honest, &air, &mut DefaultTranscript::<F>::new(&[])));
    assert!(ok);
    let rec = FriCapture::<F>::from_any(records[0].as_ref()).expect("one record");
    let (schedule, depths) = schedule_depths(&opts);
    let (d0, d, c) = (schedule[0] as usize, depths[0], 3usize);
    // Layer 0's leaf is `iota >> d0`; its cap node is `leaf >> (d − c)`.
    let reached: Vec<usize> = rec.iotas.iter().map(|i| (i >> d0) >> (d - c)).collect();
    let unreached: Vec<usize> = (0..8).filter(|k| !reached.contains(k)).collect();
    assert!(unreached.len() >= 5, "3 queries reach at most 3 of 8 nodes");
    for k in unreached {
        let mut bad = honest.clone();
        bad.query_list[0].layers_auth_paths[0].merkle_path[d - c + k][3] ^= 1;
        assert!(!verifies(&air, &bad), "unreached cap node {k}");
        assert!(
            !verifies_archived(&air, &bad),
            "unreached cap node {k}: archived"
        );
    }
}

/// The (cap, fri) format is a verifier constant: a proof made under one fails
/// under every other cell of the matrix.
#[test]
fn a_proof_made_under_one_cap_and_fri_format_fails_under_another() {
    let cells = [
        (CapPolicy::Off, FriMode::Pair),
        (CapPolicy::Auto, FriMode::Pair),
        (CapPolicy::Off, FriMode::Dp),
        (CapPolicy::Auto, FriMode::Dp),
    ];
    for (i, &(cap, fri)) in cells.iter().enumerate() {
        let (_, proof) = prove(&options(cap, fri, None, QUERIES));
        for (k, &(cap_v, fri_v)) in cells.iter().enumerate() {
            let air = SimpleAdditionAIR::<F>::new(&options(cap_v, fri_v, None, QUERIES));
            assert_eq!(
                verifies(&air, &proof),
                i == k,
                "proved at ({cap}, {fri:?}), verified at ({cap_v}, {fri_v:?})"
            );
        }
    }
}
