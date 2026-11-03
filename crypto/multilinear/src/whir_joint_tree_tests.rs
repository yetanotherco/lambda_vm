//! One round-0 tree a grouped run ([`Round0Trees::One`]) on the host, through
//! the stacked opening the block's groups use: round trips, the default left
//! as it was, and the refusals — a wrong leaf value, a wrong path node, a leaf
//! of another width or tag, a per-tree proof under a joint verifier and the
//! reverse, a root count other than the verifier's own.

use crypto::fiat_shamir::is_transcript::IsTranscript;
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
        ChainConfig, ChainFormat, FOLD_WORD_ONE_TREE, FirstFold, GrindBits, GroupFold,
        GroupedShape, Round0Trees, RoundOpenings, WhirFolds,
    },
    whir_commit::{Commitment, CosetOpening, LeafOrder, TreeDrop},
    whir_hash::{KeccakWhir, P1Whir, WhirHash},
    whir_round::{GroupedRoundProof, RoundProof},
};

use crate::whir_commit::joint_tag_fault::{Fault as JointTagFault, with as with_joint_tag_fault};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

/// Every stack here: 2^8 polynomials, round 0 folding 6 (blocks of 64, a
/// 192-felt leaf over a run of three), uniform 4 after.
const N: usize = 8;

fn config(round0: Round0Trees, leaves: LeafOrder, cap: CapPolicy) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group: GroupFold::Indexed,
            leaves,
            arity4_cap: cap,
            round0,
            ..ChainFormat::DEFAULT
        },
    }
}

fn one() -> ChainConfig {
    config(Round0Trees::One, LeafOrder::Coset, CapPolicy::Off)
}

fn each() -> ChainConfig {
    config(Round0Trees::Each, LeafOrder::Coset, CapPolicy::Off)
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

fn tower() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"joint-tree");
    t
}

/// A stack of `polys` polynomials committed and opened under `cfg`, each
/// column at a point of its own.
struct Opened {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    roots: Vec<Commitment>,
    domain: Domain<F>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    proof: StackedProof<F, Ext>,
}

fn columns_and_claims(polys: usize, seed: u64) -> (Vec<Mle<F>>, Vec<Vec<EE>>, Vec<EE>) {
    let columns: Vec<Mle<F>> = (0..polys)
        .map(|i| column(N, seed + 31 * i as u64))
        .collect();
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
    let stacked =
        StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrow(&columns), None, cfg)?;
    let roots = stacked.roots();
    let proof = prove::<F, Ext, _, P1Whir, _>(
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
        columns,
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
    verify_traced::<F, Ext, _, P1Whir>(
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

/// The stack's trees: one a run of at least two under `One`, the rest one a
/// polynomial; one a polynomial under `Each`.
#[test]
fn the_trees_are_the_runs_of_at_least_two() {
    assert_eq!(round0_trees(3, &one()), vec![0..3]);
    assert_eq!(round0_trees(5, &one()), vec![0..4, 4..5]);
    assert_eq!(round0_trees(6, &one()), vec![0..4, 4..6]);
    assert_eq!(round0_trees(1, &one()), vec![0..1]);
    assert_eq!(round0_trees(3, &each()), vec![0..1, 1..2, 2..3]);
    // Under the per-polynomial split there is no run to join.
    let mut per_poly = each();
    per_poly.format.group = GroupFold::PerPoly;
    assert_eq!(round0_trees(2, &per_poly), vec![0..1, 1..2]);
}

/// The default is today's format: `Each` everywhere a format is built
/// without the lever, and its fold word is the word without the lever's bit.
#[test]
fn the_default_is_one_tree_a_polynomial() {
    assert_eq!(ChainFormat::DEFAULT.round0, Round0Trees::Each);
    assert_eq!(Round0Trees::default(), Round0Trees::Each);
    assert!(ChainFormat::DEFAULT.is_default());
    let mut lever = ChainFormat::DEFAULT;
    lever.round0 = Round0Trees::One;
    assert!(!lever.is_default());
    assert_eq!(each().fold_word() & FOLD_WORD_ONE_TREE, 0);
    assert_eq!(one().fold_word(), each().fold_word() | FOLD_WORD_ONE_TREE);
}

/// A joint run round-trips at every width, beside a lone polynomial, in both
/// leaf orders and under a 4-ary cap; the stack carries one root a tree.
#[test]
fn one_tree_a_run_round_trips() {
    for leaves in [LeafOrder::Coset, LeafOrder::Natural] {
        for cap in [CapPolicy::Off, CapPolicy::Fixed(1)] {
            let cfg = config(Round0Trees::One, leaves, cap);
            for polys in [2usize, 3, 4, 5, 7] {
                let o = open(polys, &cfg, 5).unwrap();
                assert_eq!(o.roots.len(), round0_trees(polys, &cfg).len());
                trace(&o, &o.proof, &cfg)
                    .unwrap_or_else(|e| panic!("{leaves:?} {cap} P={polys}: {e:?}"));
                // Every query of a joint run opens one leaf of P'·64 values.
                let width = round0_trees(polys, &cfg)[0].len();
                if width >= 2 {
                    let p = joint(&o.proof, 0);
                    assert!(p.current.iter().all(|q| q.values.len() == width * 64));
                }
            }
        }
    }
}

/// A run of three: one root where `Each` has three, the leaf 192 felts.
#[test]
fn a_run_of_three_is_one_root_and_a_192_felt_leaf() {
    let o = open(3, &one(), 9).unwrap();
    let e = open(3, &each(), 9).unwrap();
    assert_eq!((o.roots.len(), e.roots.len()), (1, 3));
    assert_eq!(joint(&o.proof, 0).current[0].values.len(), 192);
    // The leaf's blocks are each polynomial's block at the one position: the
    // per-tree proof opens the same values, block by block, at the same
    // positions only when the transcripts agree, which they do not (another
    // fold word, other roots) — so compare the shapes, not the values.
    assert_eq!(e.proof.polys.len(), 1);
}

/// A joint proof under a per-tree verifier, and the reverse, are refused.
#[test]
fn a_joint_proof_and_a_per_tree_proof_refuse_each_others_verifier() {
    let o = open(3, &one(), 11).unwrap();
    let e = open(3, &each(), 11).unwrap();
    trace(&o, &o.proof, &one()).unwrap();
    trace(&e, &e.proof, &each()).unwrap();
    // The joint proof under `Each`: one root where the verifier derives three.
    assert!(matches!(
        trace(&o, &o.proof, &each()),
        Err(Refusal::Before(Error::QueryCountMismatch {
            expected: 3,
            got: 1
        }))
    ));
    // And with three roots supplied, the variant is refused.
    let three = vec![o.roots[0]; 3];
    assert_eq!(
        trace_under(&o, &o.proof, &three, &each()),
        in_chain_0(Error::GroupedShape(GroupedShape::Variant { round: 0 }))
    );
    // The per-tree proof under `One`: three roots where the verifier derives one.
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

/// Each joint variant is refused where the other is expected, whatever its
/// content: a joint opening rewritten as `BaseGrouped` under `One`, and a
/// grouped one rewritten as `BaseJoint` under `Each`.
#[test]
fn the_joint_variant_is_refused_wherever_it_does_not_belong() {
    let o = open(3, &one(), 13).unwrap();
    let mut as_grouped = o.proof.clone();
    let p = joint(&o.proof, 0).clone();
    as_grouped.polys[0].rounds[0].openings = RoundOpenings::BaseGrouped(GroupedRoundProof {
        current: p.current.iter().map(|q| vec![q.clone()]).collect(),
        next: p.next.clone(),
    });
    assert_eq!(
        trace(&o, &as_grouped, &one()),
        in_chain_0(Error::GroupedShape(GroupedShape::Variant { round: 0 }))
    );
    let e = open(3, &each(), 13).unwrap();
    let mut as_joint = e.proof.clone();
    let g = match &e.proof.polys[0].rounds[0].openings {
        RoundOpenings::BaseGrouped(g) => g.clone(),
        _ => panic!("chain 0 is grouped"),
    };
    as_joint.polys[0].rounds[0].openings = RoundOpenings::BaseJoint(RoundProof {
        current: g.current.iter().map(|q| q[0].clone()).collect(),
        next: g.next.clone(),
    });
    assert_eq!(
        trace(&e, &as_joint, &each()),
        in_chain_0(Error::GroupedShape(GroupedShape::Variant { round: 0 }))
    );
    // A joint opening at a later round is refused too.
    let mut late = o.proof.clone();
    late.polys[0].rounds[1].openings = RoundOpenings::BaseJoint(p.clone());
    assert_eq!(
        trace(&o, &late, &one()),
        in_chain_0(Error::GroupedShape(GroupedShape::Variant { round: 1 }))
    );
}

/// A wrong value in any block of a leaf is refused by the path.
#[test]
fn a_wrong_leaf_value_is_refused() {
    let o = open(3, &one(), 17).unwrap();
    for at in [0usize, 63, 64, 130, 191] {
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

/// A wrong node on a path is refused.
#[test]
fn a_wrong_path_node_is_refused() {
    let o = open(3, &one(), 19).unwrap();
    let path = joint(&o.proof, 0).current[1].proof.merkle_path.len();
    assert!(path > 0);
    for node in 0..path {
        let mut bad = o.proof.clone();
        joint_mut(&mut bad, 0).current[1].proof.merkle_path[node][0] ^= 1;
        assert_eq!(
            trace(&o, &bad, &one()),
            in_chain_0(Error::GroupedOpeningRejected { query: 1, tree: 0 }),
            "node {node}"
        );
    }
}

/// A leaf of another width — a value short, a value more, a block more — is
/// refused by name before any path is read.
#[test]
fn a_leaf_of_another_width_is_refused() {
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

/// The joint tag is load-bearing: a tree whose leaves are hashed under the
/// plain tag (no block count) is refused at the first path, and so is one
/// hashed under another block count.
#[test]
fn a_leaf_under_another_tag_is_refused() {
    for fault in [JointTagFault::Plain, JointTagFault::OtherCount] {
        let forged = with_joint_tag_fault(fault, || open(3, &one(), 29).unwrap());
        assert_eq!(
            trace(&forged, &forged.proof, &one()),
            in_chain_0(Error::GroupedOpeningRejected { query: 0, tree: 0 }),
            "{fault:?}"
        );
    }
}

/// A joint tree retired to its top and revived over its run's codewords
/// opens the same proof as the tree kept whole.
#[test]
fn a_retired_joint_tree_opens_the_same_proof() {
    let cfg = one();
    let polys = 5;
    let (columns, points, values) = columns_and_claims(polys, 31);
    let layout = StackedLayout::build(&vec![N; polys], N).unwrap();
    let kept =
        StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrow(&columns), None, &cfg)
            .unwrap();
    let kept_proof = prove::<F, Ext, _, P1Whir, _>(
        &kept,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        &cfg,
        &mut tower(),
    )
    .unwrap();
    for drop in [1usize, 2] {
        let retired =
            StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrow(&columns), None, &cfg)
                .unwrap()
                .retire(TreeDrop::uniform(drop), &cfg)
                .unwrap();
        assert_eq!(retired.roots(), kept.roots());
        let revived = retired
            .revive::<P1Whir, _>(&borrow(&columns), None, &cfg)
            .unwrap();
        let proof = prove::<F, Ext, _, P1Whir, _>(
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
    // Revived over other columns, the re-hashed block is refused against
    // the kept node.
    let (other, _, _) = columns_and_claims(polys, 32);
    let revived = StackedCommitment::<F, P1Whir>::commit(layout, &borrow(&columns), None, &cfg)
        .unwrap()
        .retire(TreeDrop::uniform(1), &cfg)
        .unwrap()
        .revive::<P1Whir, _>(&borrow(&other), None, &cfg)
        .unwrap();
    assert!(matches!(
        prove::<F, Ext, _, P1Whir, _>(
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

/// One tree a run needs grouped chains and a hash with a joint leaf: either
/// missing is refused before a tree is built or a proof is read.
#[test]
fn one_tree_a_run_needs_grouped_chains_and_poseidon1() {
    let mut per_poly = one();
    per_poly.format.group = GroupFold::PerPoly;
    assert!(matches!(
        crate::whir_chain::check_round0_trees::<P1Whir>(&per_poly.format),
        Err(Error::JointTreeUnsupported { .. })
    ));
    let natural = config(Round0Trees::One, LeafOrder::Natural, CapPolicy::Off);
    assert!(matches!(
        crate::whir_chain::check_round0_trees::<KeccakWhir>(&natural.format),
        Err(Error::JointTreeUnsupported { .. })
    ));
    assert!(crate::whir_chain::check_round0_trees::<P1Whir>(&one().format).is_ok());
    assert!(crate::whir_chain::check_round0_trees::<KeccakWhir>(&each().format).is_ok());
}

/// The opened values are each polynomial's coset block, end to end: block `p`
/// of a query's leaf is what polynomial `p`'s own tree opens there.
#[test]
fn a_leaf_is_the_runs_blocks_end_to_end() {
    let o = open(3, &one(), 37).unwrap();
    let leaf: &CosetOpening<F> = &joint(&o.proof, 0).current[0];
    // Re-derive block `p` from polynomial `p`'s codeword at the queried block.
    let cfg = one();
    let domain = &o.domain;
    let size = domain.size();
    let block = 64;
    let num_leaves = size / block;
    let codewords: Vec<Vec<FE>> = o
        .columns
        .iter()
        .map(|c| crate::whir::encode::<F, F>(&crate::whir::lift_coefficients(c), domain).unwrap())
        .collect();
    // Find the block index whose concatenation is the leaf.
    let found = (0..num_leaves).find(|&j| {
        (0..3).all(|p| {
            (0..block).all(|t| codewords[p][j + t * num_leaves] == leaf.values[p * block + t])
        })
    });
    assert!(
        found.is_some(),
        "the leaf is three blocks at one position ({cfg:?})"
    );
}
