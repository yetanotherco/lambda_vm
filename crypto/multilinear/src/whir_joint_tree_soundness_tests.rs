//! The steps of the soundness argument for one round-0 tree a run, each
//! stated as a test of its own.
//!
//! - the joint leaf is the spelled-out rate-12 chain with the block count in
//!   the first permutation's input (and nothing else differs from the plain
//!   leaf);
//! - the fold word is injective over every format the chain takes, bit 45 set
//!   exactly under `One`;
//! - the verifier's transcript schedule inside a stack is `Each`'s, call for
//!   call (only the roots, absorbed by the caller, differ);
//! - at the verifier's own positions, block `p` of a joint leaf is polynomial
//!   `p`'s coset block — what an `Each` verifier reads from tree `p`;
//! - a swapped pair of blocks is refused (the leaf binds the order);
//! - `BaseJoint` at a lone polynomial is refused;
//! - the chain-level entry points refuse a mode their config does not have.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::poseidon1_stark::{
    LEAF_DOMAIN, joint_leaf_capacity, joint_linear_hash, leaf_capacity, linear_hash,
};
use crypto::hash::poseidon1_w16::permute;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    mle::Mle,
    stacked_eval::{
        Claimed, Refusal, StackedCommitment, StackedProof, index_chains, prove, verify_traced,
    },
    stacking::{StackedLayout, borrow},
    whir::Domain,
    whir_chain::{
        ChainConfig, ChainFormat, FOLD_WORD_ONE_TREE, FirstFold, GrindBits, GroupFold,
        GroupedShape, Round0Trees, RoundOpenings, WhirFolds,
    },
    whir_commit::{Commitment, LeafOrder},
    whir_hash::{P1Whir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

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

fn column(seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << N))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

/// Records every call's kind (and an index draw's bound and value), passing
/// it on to the real sponge.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Op {
    Bytes(usize),
    Felt,
    Draw,
    Index(u64),
}

struct Recording<T> {
    inner: T,
    ops: Vec<Op>,
    indices: Vec<u64>,
}

impl<T: IsTranscript<Ext>> IsTranscript<Ext> for Recording<T> {
    fn append_field_element(&mut self, element: &EE) {
        self.ops.push(Op::Felt);
        self.inner.append_field_element(element)
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.ops.push(Op::Bytes(new_bytes.len()));
        self.inner.append_bytes(new_bytes)
    }
    fn mark_statement_end(&mut self) {
        self.inner.mark_statement_end()
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> EE {
        self.ops.push(Op::Draw);
        self.inner.sample_field_element()
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        let v = self.inner.sample_u64(upper_bound);
        self.ops.push(Op::Index(upper_bound));
        self.indices.push(v);
        v
    }
}

fn tower() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"joint-tree-soundness");
    t
}

fn recording() -> Recording<<P1Whir as WhirHash>::Sponge> {
    Recording {
        inner: tower(),
        ops: Vec::new(),
        indices: Vec::new(),
    }
}

struct Opened {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    roots: Vec<Commitment>,
    domain: Domain<F>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    proof: StackedProof<F, Ext>,
}

fn open(polys: usize, cfg: &ChainConfig, seed: u64) -> Opened {
    let columns: Vec<Mle<F>> = (0..polys).map(|i| column(seed + 31 * i as u64)).collect();
    let points: Vec<Vec<EE>> = (0..polys)
        .map(|i| {
            (0..N)
                .map(|j| EE::from(7 + seed + 11 * i as u64 + 3 * j as u64))
                .collect()
        })
        .collect();
    let values: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    let layout = StackedLayout::build(&vec![N; polys], N).unwrap();
    let stacked =
        StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrow(&columns), None, cfg)
            .unwrap();
    let roots = stacked.roots();
    let proof = prove::<F, Ext, _, P1Whir, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        cfg,
        &mut tower(),
    )
    .unwrap();
    Opened {
        layout,
        columns,
        roots,
        domain: stacked.domain().clone(),
        points,
        values,
        proof,
    }
}

fn verify_with<T: IsTranscript<Ext>>(
    o: &Opened,
    proof: &StackedProof<F, Ext>,
    cfg: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Refusal> {
    verify_traced::<F, Ext, _, P1Whir>(
        proof,
        &o.layout,
        &o.roots,
        &Claimed::PerColumn(&o.points),
        &o.values,
        &o.domain,
        cfg,
        transcript,
    )
}

/// The joint leaf, spelled out: the first permutation's input is the first
/// twelve felts and `[len, LEAF_DOMAIN, P', 0]`; every later one the next
/// twelve (zero-filled) and the previous output's lanes 0..4; the digest is
/// lanes 0..4 of the last output. With lane 14 zero it is the plain leaf.
/// The block count is therefore in the hashed input, and nothing else
/// differs from the plain leaf of the same felts.
#[test]
fn the_joint_leaf_is_the_spelled_out_chain() {
    let by_hand = |felts: &[FE], cap: [FE; 4]| {
        let mut state = [FE::zero(); 16];
        state[12..].copy_from_slice(&cap);
        for (k, chunk) in felts.chunks(12).enumerate() {
            if k > 0 {
                let carry = [state[0], state[1], state[2], state[3]];
                state[12..].copy_from_slice(&carry);
            }
            for (lane, slot) in state[..12].iter_mut().enumerate() {
                *slot = chunk.get(lane).copied().unwrap_or(FE::zero());
            }
            state = permute(state);
        }
        [state[0], state[1], state[2], state[3]]
    };
    for blocks in 2..=4usize {
        for block in [2usize, 4, 16, 64] {
            let len = blocks * block;
            let felts: Vec<FE> = (0..len as u64)
                .map(|i| FE::from(i.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x5A))
                .collect();
            let tag = [
                FE::from(len as u64),
                FE::from(LEAF_DOMAIN),
                FE::from(blocks as u64),
                FE::zero(),
            ];
            assert_eq!(joint_leaf_capacity(len, blocks), tag);
            assert_eq!(
                joint_linear_hash(&felts, blocks),
                by_hand(&felts, tag),
                "P' {blocks}, block {block}"
            );
            // The tree's leaf is that digest's bytes.
            assert_eq!(
                P1Whir::joint_leaf(&felts, blocks).unwrap(),
                crypto::hash::rpx::digest_to_commitment(&by_hand(&felts, tag)),
            );
            let plain = [tag[0], tag[1], FE::zero(), FE::zero()];
            assert_eq!(leaf_capacity(len), plain);
            assert_eq!(linear_hash(&felts), by_hand(&felts, plain));
            assert_ne!(joint_linear_hash(&felts, blocks), linear_hash(&felts));
            for other in (1..=8usize).filter(|&b| b != blocks) {
                assert_ne!(
                    joint_linear_hash(&felts, blocks),
                    joint_linear_hash(&felts, other)
                );
            }
        }
    }
    // No first capacity of a joint leaf is a plain leaf's, at any width.
    for len in 1..=1024usize {
        for blocks in 2..=4usize {
            for width in [len, len / 3, 3 * len] {
                assert_ne!(joint_leaf_capacity(len, blocks), leaf_capacity(width));
            }
        }
    }
}

/// The fold word is injective over the chain's formats: folds (uniform or a
/// first fold), `log_folding`, group, leaf order and round-0 trees, and bit 45
/// is set exactly under `One`.
#[test]
fn fold_words_are_distinct_across_formats() {
    let mut seen: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    for log_folding in 1..=6usize {
        for first in [
            None,
            Some(1usize),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
        ] {
            for group in [GroupFold::PerPoly, GroupFold::Indexed] {
                for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
                    for round0 in [Round0Trees::Each, Round0Trees::One] {
                        let cfg = ChainConfig {
                            log_blowup: 1,
                            log_folding,
                            num_queries: 3,
                            grind: GrindBits::default(),
                            format: ChainFormat {
                                folds: first.map_or(WhirFolds::Uniform, |k| {
                                    WhirFolds::First(FirstFold::new(k).unwrap())
                                }),
                                group,
                                leaves,
                                round0,
                                ..ChainFormat::DEFAULT
                            },
                        };
                        let word = cfg.fold_word();
                        assert_eq!(
                            word & FOLD_WORD_ONE_TREE != 0,
                            round0 == Round0Trees::One,
                            "{cfg:?}"
                        );
                        let key =
                            format!("{log_folding} {first:?} {group:?} {leaves:?} {round0:?}");
                        if let Some(prev) = seen.insert(word, key.clone()) {
                            panic!("{word:#x}: {prev} and {key}");
                        }
                    }
                }
            }
        }
    }
}

/// Inside a stack, a joint verifier's transcript calls are a per-tree
/// verifier's, call for call (kind, byte length, draw bound): the roots, which
/// the caller absorbs, are the only transcript difference between the modes.
/// Over runs of two, three and four, a [4, 1] and a [4, 2] split, both leaf
/// orders and a 4-ary cap.
#[test]
fn a_joint_stacks_transcript_schedule_is_the_per_tree_ones() {
    for polys in [2usize, 3, 4, 5, 6] {
        for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
            for cap in [CapPolicy::Off, CapPolicy::Fixed(2)] {
                let one = config(Round0Trees::One, leaves, cap);
                let each = config(Round0Trees::Each, leaves, cap);
                let o = open(polys, &one, 41);
                let e = open(polys, &each, 41);
                let (mut ro, mut re) = (recording(), recording());
                verify_with(&o, &o.proof, &one, &mut ro).unwrap();
                verify_with(&e, &e.proof, &each, &mut re).unwrap();
                assert!(!ro.ops.is_empty());
                assert_eq!(ro.ops, re.ops, "{polys} polys, {leaves:?}, {cap:?}");
            }
        }
    }
}

/// ★ At the verifier's own positions — the index draws its transcript makes
/// in round 0 of each chain — block `p` of the joint leaf is polynomial `p`'s
/// coset block: `codeword_p[q + t·L]`, `t < 2^k0`, which is what tree `p`
/// opens to an `Each` verifier at `q`. Runs of two, three and four, the
/// [4, 2] split, both orders.
#[test]
fn a_joint_leaf_is_each_polynomials_block_at_the_verifiers_position() {
    for polys in [2usize, 3, 4, 6] {
        for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
            let cfg = config(Round0Trees::One, leaves, CapPolicy::Off);
            let o = open(polys, &cfg, 43);
            let mut rec = recording();
            verify_with(&o, &o.proof, &cfg, &mut rec).unwrap();
            let block = 1usize << cfg.schedule(N)[0];
            let size = o.domain.size();
            let num_leaves = size / block;
            let rounds = cfg.schedule(N).len();
            let q_count = cfg.num_queries;
            let codewords: Vec<Vec<FE>> = o
                .columns
                .iter()
                .map(|c| {
                    crate::whir::encode::<F, F>(&crate::whir::lift_coefficients(c), &o.domain)
                        .unwrap()
                })
                .collect();
            let runs = index_chains(polys, GroupFold::Indexed);
            let mut checked = 0;
            for (chain, run) in runs.iter().enumerate() {
                if run.len() < 2 {
                    continue;
                }
                // Chain `c`'s round 0 draws its `Q` positions first among its
                // `Q · rounds` index draws.
                let at = chain * q_count * rounds;
                let positions = &rec.indices[at..at + q_count];
                let opened = match &o.proof.polys[chain].rounds[0].openings {
                    RoundOpenings::BaseJoint(p) => p,
                    _ => panic!("chain {chain} is joint"),
                };
                for (i, &q) in positions.iter().enumerate() {
                    assert!((q as usize) < num_leaves);
                    let leaf = &opened.current[i].values;
                    assert_eq!(leaf.len(), run.len() * block);
                    for (p, poly) in run.clone().enumerate() {
                        for t in 0..block {
                            assert_eq!(
                                leaf[p * block + t],
                                codewords[poly][q as usize + t * num_leaves],
                                "{polys} polys {leaves:?}: chain {chain} query {i} slot {p} value {t}"
                            );
                        }
                    }
                    checked += 1;
                }
            }
            assert!(checked > 0);
        }
    }
}

/// The leaf binds the ORDER of its blocks, not only their set: blocks 0 and 1
/// swapped in every query's leaf are refused at the first path.
#[test]
fn swapped_blocks_are_refused() {
    let cfg = config(Round0Trees::One, LeafOrder::Coset, CapPolicy::Off);
    let o = open(3, &cfg, 47);
    let mut bad = o.proof.clone();
    match &mut bad.polys[0].rounds[0].openings {
        RoundOpenings::BaseJoint(p) => {
            for opening in &mut p.current {
                let (a, b) = opening.values.split_at_mut(64);
                a.swap_with_slice(&mut b[..64]);
            }
        }
        _ => panic!("chain 0 is joint"),
    }
    assert_ne!(format!("{bad:?}"), format!("{:?}", o.proof));
    assert_eq!(
        verify_with(&o, &bad, &cfg, &mut tower()),
        Err(Refusal::Chain {
            chain: 0,
            poly: 0,
            cause: Error::GroupedOpeningRejected { query: 0, tree: 0 },
        })
    );
}

/// `BaseJoint` where a lone polynomial stands (the 1 of a [4, 1] split) is
/// refused by name, whatever it holds.
#[test]
fn a_lone_polynomial_takes_no_joint_variant() {
    let cfg = config(Round0Trees::One, LeafOrder::Coset, CapPolicy::Off);
    let o = open(5, &cfg, 53);
    verify_with(&o, &o.proof, &cfg, &mut tower()).unwrap();
    let mut bad = o.proof.clone();
    let lone = match &o.proof.polys[1].rounds[0].openings {
        RoundOpenings::Base(p) => p.clone(),
        _ => panic!("chain 1 is a lone polynomial's"),
    };
    bad.polys[1].rounds[0].openings = RoundOpenings::BaseJoint(lone);
    assert_eq!(
        verify_with(&o, &bad, &cfg, &mut tower()),
        Err(Refusal::Chain {
            chain: 1,
            poly: 4,
            cause: Error::GroupedShape(GroupedShape::Variant { round: 0 }),
        })
    );
}

/// The chain-level entry points take the mode from their config alone:
/// `verify_joint` under `Each`, or over one polynomial, is refused before
/// anything is read; `verify_grouped` handed a run's `P'` roots under `One`
/// is refused by its root count.
#[test]
fn the_chain_entry_points_refuse_a_mode_their_config_lacks() {
    let one = config(Round0Trees::One, LeafOrder::Coset, CapPolicy::Off);
    let each = config(Round0Trees::Each, LeafOrder::Coset, CapPolicy::Off);
    let o = open(3, &one, 59);
    let chain = &o.proof.polys[0];
    let weight = |_: &[EE]| -> Result<EE, Error> { Ok(EE::one()) };
    assert!(matches!(
        crate::whir_chain::verify_joint::<F, Ext, _, _, P1Whir>(
            chain,
            &o.roots[0],
            3,
            weight,
            EE::zero(),
            N,
            &o.domain,
            &each,
            &mut tower(),
        ),
        Err(Error::JointTreeUnsupported { .. })
    ));
    assert!(matches!(
        crate::whir_chain::verify_joint::<F, Ext, _, _, P1Whir>(
            chain,
            &o.roots[0],
            1,
            weight,
            EE::zero(),
            N,
            &o.domain,
            &one,
            &mut tower(),
        ),
        Err(Error::JointTreeUnsupported { .. })
    ));
    let three = vec![o.roots[0]; 3];
    assert!(matches!(
        crate::whir_chain::verify_grouped::<F, Ext, _, _, P1Whir>(
            chain,
            &three,
            weight,
            EE::zero(),
            N,
            &o.domain,
            &one,
            &mut tower(),
        ),
        Err(Error::GroupedShape(GroupedShape::RunWidth { polys: 3 }))
    ));
}
