//! One round-0 tree a grouped run ([`Round0Trees::One`]) under standard
//! 7-round BLAKE3 ([`Blake3Whir`]) on the host, through the stacked opening
//! the block's groups use: round trips, and the refusals — a wrong value in
//! any block of a leaf, two blocks swapped, a leaf of another width or query
//! count, a leaf hashed under another block count or with no prefix, a joint
//! proof under a per-tree verifier and the reverse — plus a retired tree
//! revived, the verifier's host commit, and what one path a query saves.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::backends::blake3_joint;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    mle::Mle,
    stacked_eval::{
        Claimed, Refusal, StackedCommitment, StackedProof, prove, round0_trees, verify_traced,
    },
    stacking::{StackedLayout, borrow},
    whir::Domain,
    whir_chain::{
        ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, GroupedShape, Round0Trees,
        RoundOpenings, WhirFolds, check_round0_trees,
    },
    whir_commit::{Commitment, LeafOrder, TreeDrop},
    whir_hash::{Blake3Whir, KeccakWhir, P1Whir, RpxWhir, WhirHash},
    whir_round::RoundProof,
};

use crate::whir_commit::joint_tag_fault::{Fault as JointTagFault, with as with_joint_tag_fault};

type H = Blake3Whir;
type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

/// Every stack here: 2^8 polynomials, round 0 folding 6 (blocks of 64 base
/// values, a 1,536-byte leaf over a run of three), uniform 4 after.
const N: usize = 8;

fn config(round0: Round0Trees, leaves: LeafOrder) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group: GroupFold::Indexed,
            leaves,
            arity4_cap: CapPolicy::Off,
            round0,
            ..ChainFormat::DEFAULT
        },
    }
}

fn one() -> ChainConfig {
    config(Round0Trees::One, LeafOrder::Coset)
}

fn each() -> ChainConfig {
    config(Round0Trees::Each, LeafOrder::Coset)
}

fn column(seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << N))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

fn tower() -> <H as WhirHash>::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"joint-tree-b3");
    t
}

struct Opened {
    layout: StackedLayout,
    roots: Vec<Commitment>,
    domain: Domain<F>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    proof: StackedProof<F, Ext>,
}

fn columns_and_claims(polys: usize, seed: u64) -> (Vec<Mle<F>>, Vec<Vec<EE>>, Vec<EE>) {
    let columns: Vec<Mle<F>> = (0..polys).map(|i| column(seed + 31 * i as u64)).collect();
    let points: Vec<Vec<EE>> = (0..polys)
        .map(|i| {
            (0..N)
                .map(|j| EE::from(7 + seed + 11 * i as u64 + 3 * j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    (columns, points, values)
}

fn open(polys: usize, cfg: &ChainConfig, seed: u64) -> Result<Opened, Error> {
    let (columns, points, values) = columns_and_claims(polys, seed);
    let layout = StackedLayout::build(&vec![N; polys], N)?;
    let stacked = StackedCommitment::<F, H>::commit(layout.clone(), &borrow(&columns), None, cfg)?;
    let roots = stacked.roots();
    let proof = prove::<F, Ext, _, H, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        cfg,
        &mut tower(),
    )?;
    Ok(Opened {
        layout,
        roots,
        domain: stacked.domain().clone(),
        points,
        values,
        proof,
    })
}

fn trace_under(
    o: &Opened,
    proof: &StackedProof<F, Ext>,
    roots: &[Commitment],
    cfg: &ChainConfig,
) -> Result<(), Refusal> {
    verify_traced::<F, Ext, _, H>(
        proof,
        &o.layout,
        roots,
        &Claimed::PerColumn(&o.points),
        &o.values,
        &o.domain,
        cfg,
        &mut tower(),
    )
}

fn trace(o: &Opened, proof: &StackedProof<F, Ext>, cfg: &ChainConfig) -> Result<(), Refusal> {
    trace_under(o, proof, &o.roots, cfg)
}

/// Chain 0 (over polynomials 0..), refused by `cause`.
fn in_chain_0(cause: Error) -> Result<(), Refusal> {
    Err(Refusal::Chain {
        chain: 0,
        poly: 0,
        cause,
    })
}

fn joint(proof: &StackedProof<F, Ext>, chain: usize) -> &RoundProof<F, Ext> {
    match &proof.polys[chain].rounds[0].openings {
        RoundOpenings::BaseJoint(p) => p,
        _ => panic!("chain {chain} is not joint"),
    }
}

fn joint_mut(proof: &mut StackedProof<F, Ext>, chain: usize) -> &mut RoundProof<F, Ext> {
    match &mut proof.polys[chain].rounds[0].openings {
        RoundOpenings::BaseJoint(p) => p,
        _ => panic!("chain {chain} is not joint"),
    }
}

fn bytes(proof: &StackedProof<F, Ext>) -> usize {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof).unwrap().len()
}

/// One tree a run is admitted under BLAKE3 and Poseidon1, whose trees have a
/// joint leaf, and refused under Keccak and RPX, whose have none, and under
/// the per-polynomial split.
#[test]
fn one_tree_a_run_is_admitted_under_blake3() {
    const {
        assert!(H::JOINT_LEAF && P1Whir::JOINT_LEAF);
        assert!(!KeccakWhir::JOINT_LEAF && !RpxWhir::JOINT_LEAF);
    }
    assert!(check_round0_trees::<H>(&one().format).is_ok());
    assert!(check_round0_trees::<H>(&each().format).is_ok());
    for refused in [
        check_round0_trees::<KeccakWhir>(&one().format),
        check_round0_trees::<RpxWhir>(&one().format),
    ] {
        assert!(matches!(refused, Err(Error::JointTreeUnsupported { .. })));
    }
    let mut per_poly = one();
    per_poly.format.group = GroupFold::PerPoly;
    assert!(matches!(
        check_round0_trees::<H>(&per_poly.format),
        Err(Error::JointTreeUnsupported { .. })
    ));
}

/// ★ A joint run round-trips under BLAKE3 at every width, beside a lone
/// polynomial, in both leaf orders; the stack carries one root a tree, and a
/// joint query opens one leaf of `P'·64` values.
#[test]
fn a_blake3_run_round_trips() {
    for leaves in [LeafOrder::Coset, LeafOrder::Natural] {
        let cfg = config(Round0Trees::One, leaves);
        for polys in [2usize, 3, 4, 5, 7] {
            let o = open(polys, &cfg, 5).unwrap();
            assert_eq!(o.roots.len(), round0_trees(polys, &cfg).len());
            trace(&o, &o.proof, &cfg).unwrap_or_else(|e| panic!("{leaves:?} P={polys}: {e:?}"));
            let width = round0_trees(polys, &cfg)[0].len();
            if width >= 2 {
                let p = joint(&o.proof, 0);
                assert!(p.current.iter().all(|q| q.values.len() == width * 64));
            }
        }
    }
}

/// ★ A queried joint leaf is the BLAKE3 joint leaf of its values: the
/// prefix with the run's shape, then every block end to end
/// (`blake3_joint::joint_leaf`), and not the per-polynomial leaf of the same
/// values.
#[test]
fn a_queried_leaf_hashes_under_the_joint_prefix() {
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    let o = open(3, &one(), 7).unwrap();
    let leaf = &joint(&o.proof, 0).current[0].values;
    assert_eq!(leaf.len(), 192);
    let digest = H::joint_leaf(leaf, 3).unwrap();
    assert_eq!(Some(digest), blake3_joint::joint_leaf(leaf, 3));
    assert_ne!(
        digest,
        <<H as WhirHash>::Backend<F> as IsMerkleTreeBackend>::hash_data(leaf)
    );
}

/// ★ A wrong value in any block (any polynomial) of a leaf is refused by its
/// path, at every query.
#[test]
fn a_tampered_value_in_any_column_is_refused() {
    let o = open(3, &one(), 17).unwrap();
    for at in [0usize, 31, 63, 64, 100, 127, 128, 160, 191] {
        for query in 0..3 {
            let mut bad = o.proof.clone();
            joint_mut(&mut bad, 0).current[query].values[at] += FE::one();
            assert_eq!(
                trace(&o, &bad, &one()),
                in_chain_0(Error::GroupedOpeningRejected { query, tree: 0 }),
                "value {at} of query {query}"
            );
        }
    }
}

/// ★ The leaf binds the order of its blocks: any two polynomials' blocks
/// swapped in a query's leaf are refused at that query.
#[test]
fn a_column_swap_is_refused() {
    let o = open(3, &one(), 47).unwrap();
    for (a, b) in [(0usize, 1usize), (1, 2), (0, 2)] {
        for query in 0..3 {
            let mut bad = o.proof.clone();
            let values = &mut joint_mut(&mut bad, 0).current[query].values;
            for t in 0..64 {
                values.swap(a * 64 + t, b * 64 + t);
            }
            assert_ne!(format!("{bad:?}"), format!("{:?}", o.proof));
            assert_eq!(
                trace(&o, &bad, &one()),
                in_chain_0(Error::GroupedOpeningRejected { query, tree: 0 }),
                "blocks {a} and {b} of query {query}"
            );
        }
    }
}

/// ★ A leaf of another width — a value short, a block more — or another
/// query count is refused by name before any path is read.
#[test]
fn a_width_or_length_mismatch_is_refused() {
    let o = open(3, &one(), 23).unwrap();
    let mut short = o.proof.clone();
    joint_mut(&mut short, 0).current[2].values.pop();
    assert_eq!(
        trace(&o, &short, &one()),
        in_chain_0(Error::GroupedShape(GroupedShape::BlockWidth {
            query: 2,
            tree: 0,
            expected: 192,
            got: 191
        }))
    );
    let mut wide = o.proof.clone();
    let extra: Vec<FE> = joint(&o.proof, 0).current[0].values[..64].to_vec();
    joint_mut(&mut wide, 0).current[0].values.extend(extra);
    assert_eq!(
        trace(&o, &wide, &one()),
        in_chain_0(Error::GroupedShape(GroupedShape::BlockWidth {
            query: 0,
            tree: 0,
            expected: 192,
            got: 256
        }))
    );
    let mut fewer = o.proof.clone();
    joint_mut(&mut fewer, 0).current.pop();
    assert_eq!(
        trace(&o, &fewer, &one()),
        in_chain_0(Error::GroupedShape(GroupedShape::Queries {
            expected: 3,
            got: 2
        }))
    );
}

/// ★ The prefix is load-bearing: a tree whose leaves are hashed with no
/// prefix (the per-polynomial leaf over the same values), or under another
/// block count, is refused at the first path.
#[test]
fn a_leaf_under_another_prefix_is_refused() {
    for fault in [JointTagFault::Plain, JointTagFault::OtherCount] {
        let forged = with_joint_tag_fault(fault, || open(3, &one(), 29).unwrap());
        assert_eq!(
            trace(&forged, &forged.proof, &one()),
            in_chain_0(Error::GroupedOpeningRejected { query: 0, tree: 0 }),
            "{fault:?}"
        );
    }
}

/// A joint proof under a per-tree verifier, and the reverse, are refused by
/// their root count.
#[test]
fn a_joint_proof_and_a_per_tree_proof_refuse_each_others_verifier() {
    let o = open(3, &one(), 11).unwrap();
    let e = open(3, &each(), 11).unwrap();
    trace(&o, &o.proof, &one()).unwrap();
    trace(&e, &e.proof, &each()).unwrap();
    assert!(matches!(
        trace(&o, &o.proof, &each()),
        Err(Refusal::Before(Error::QueryCountMismatch {
            expected: 3,
            got: 1
        }))
    ));
    assert!(matches!(
        trace(&e, &e.proof, &one()),
        Err(Refusal::Before(Error::QueryCountMismatch {
            expected: 1,
            got: 3
        }))
    ));
    assert_eq!(
        trace_under(&e, &e.proof, &e.roots[..1], &one()),
        in_chain_0(Error::GroupedShape(GroupedShape::Variant { round: 0 }))
    );
}

/// A joint BLAKE3 tree retired to its top and revived over its run's
/// codewords opens the same proof as the tree kept whole; revived over other
/// columns, the re-hashed block is refused against the kept node.
#[test]
fn a_retired_blake3_joint_tree_opens_the_same_proof() {
    let cfg = one();
    let polys = 5;
    let (columns, points, values) = columns_and_claims(polys, 31);
    let layout = StackedLayout::build(&vec![N; polys], N).unwrap();
    let kept =
        StackedCommitment::<F, H>::commit(layout.clone(), &borrow(&columns), None, &cfg).unwrap();
    let kept_proof = prove::<F, Ext, _, H, _>(
        &kept,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        &cfg,
        &mut tower(),
    )
    .unwrap();
    for drop in [1usize, 3] {
        let retired =
            StackedCommitment::<F, H>::commit(layout.clone(), &borrow(&columns), None, &cfg)
                .unwrap()
                .retire(TreeDrop::uniform(drop), &cfg)
                .unwrap();
        assert_eq!(retired.roots(), kept.roots());
        let revived = retired
            .revive::<H, _>(&borrow(&columns), None, &cfg)
            .unwrap();
        let proof = prove::<F, Ext, _, H, _>(
            &revived,
            &borrow(&columns),
            None,
            &Claimed::PerColumn(&points),
            &values,
            &cfg,
            &mut tower(),
        )
        .unwrap();
        assert_eq!(
            format!("{proof:?}"),
            format!("{kept_proof:?}"),
            "drop {drop}"
        );
    }
    let (other, _, _) = columns_and_claims(polys, 32);
    let revived = StackedCommitment::<F, H>::commit(layout, &borrow(&columns), None, &cfg)
        .unwrap()
        .retire(TreeDrop::uniform(1), &cfg)
        .unwrap()
        .revive::<H, _>(&borrow(&other), None, &cfg)
        .unwrap();
    assert!(matches!(
        prove::<F, Ext, _, H, _>(
            &revived,
            &borrow(&columns),
            None,
            &Claimed::PerColumn(&points),
            &values,
            &cfg,
            &mut tower(),
        ),
        Err(Error::RecomputedCodewordMismatch { .. })
    ));
}

/// A verifier's host commit (`commit_on_host`) has the commit's roots under
/// either round-0 format and either leaf order.
#[test]
fn the_host_commit_has_the_commits_roots_under_blake3() {
    for round0 in [Round0Trees::Each, Round0Trees::One] {
        for leaves in [LeafOrder::Coset, LeafOrder::Natural] {
            let cfg = config(round0, leaves);
            for polys in [1usize, 3, 5] {
                let (columns, _, _) = columns_and_claims(polys, 41);
                let layout = StackedLayout::build(&vec![N; polys], N).unwrap();
                let committed = StackedCommitment::<F, H>::commit(
                    layout.clone(),
                    &borrow(&columns),
                    None,
                    &cfg,
                )
                .unwrap();
                let on_host =
                    StackedCommitment::<F, H>::commit_on_host(layout, &borrow(&columns), &cfg)
                        .unwrap();
                assert_eq!(
                    on_host.roots(),
                    committed.roots(),
                    "{round0:?} {leaves:?} P={polys}"
                );
            }
        }
    }
}

/// What the format saves in bytes: a run of three opens one path a query
/// where the per-polynomial trees open three, so the joint proof is smaller
/// by at least two paths a query of round 0 (`depth` 32-byte siblings each)
/// and two roots fewer are carried beside it. The leaf values are the same
/// count either way.
#[test]
fn one_path_a_query_is_what_the_format_saves() {
    let o = open(3, &one(), 3).unwrap();
    let e = open(3, &each(), 3).unwrap();
    let depth = (o.domain.size() >> 6).trailing_zeros() as usize;
    let p = joint(&o.proof, 0);
    assert!(p.current.iter().all(|q| q.proof.merkle_path.len() == depth));
    let (one_bytes, each_bytes) = (bytes(&o.proof), bytes(&e.proof));
    let paths = 2 * one().num_queries * depth * 32;
    println!(
        "B3 JOINT BYTES one {one_bytes} each {each_bytes} saved {} (two paths a query: {paths})",
        each_bytes - one_bytes
    );
    assert!(
        each_bytes >= one_bytes + paths,
        "{each_bytes} vs {one_bytes}"
    );
    assert_eq!((o.roots.len(), e.roots.len()), (1, 3));
}
