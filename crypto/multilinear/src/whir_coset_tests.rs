//! The coset leaf order ([`LeafOrder::Coset`]) on the host: a round-0 tree
//! places block `j` at position `bitrev(j)`, the prover's paths (whole trees
//! and kept tops) and the verifier read it there, and nothing else moves. The
//! order is Poseidon1's alone: under any other hash it is refused by name.
//!
//! The refusals across orders are the mutations that show each side's map is
//! load-bearing: a verifier that reads positions in the natural order (the
//! map dropped from the verifier) refuses a coset proof, and a prover whose
//! trees are natural (the map dropped from the prover) is refused by a coset
//! verifier.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    mle::Mle,
    stacked_eval::{Claimed, Refusal, StackedCommitment, StackedProof, prove, verify_traced},
    stacking::{StackedLayout, borrow},
    whir_chain::{
        self, ChainConfig, ChainFormat, FOLD_WORD_COSET, FOLD_WORD_INDEXED, FOLD_WORD_TAG,
        FirstFold, GrindBits, GroupFold, LeafOrder, UnencodedSuccessor, WhirFolds,
    },
    whir_commit::{CodewordCommitment, Commitment, TreeDrop, verify_opening},
    whir_hash::{KeccakWhir, P1Whir, RpxWhir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn config(
    first: Option<usize>,
    log_folding: usize,
    group: GroupFold,
    leaves: LeafOrder,
    cap: CapPolicy,
) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding,
        num_queries: 9,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: first.map_or(WhirFolds::Uniform, |k| {
                WhirFolds::First(FirstFold::new(k).unwrap())
            }),
            group,
            leaves,
            cap,
            arity4_cap: cap,
            ..ChainFormat::DEFAULT
        },
    }
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

/// A stack, its claims, its roots and its proof.
struct Opened {
    layout: StackedLayout,
    roots: Vec<Commitment>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    domain: crate::whir::Domain<F>,
    proof: StackedProof<F, Ext>,
}

fn sponge<H: WhirHash>() -> H::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"coset");
    t
}

/// Commits the stack under `cfg`, retires its trees down to `drop` levels and
/// revives them when `drop` is given, and opens it.
fn open<H: WhirHash>(
    heights: &[usize],
    n_stack: usize,
    cfg: &ChainConfig,
    drop: Option<usize>,
) -> Opened {
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 5 + 31 * i as u64))
        .collect();
    let points: Vec<Vec<EE>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| EE::from(7 + 11 * i as u64 + 3 * j as u64))
                .collect()
        })
        .collect();
    let values: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    let layout = StackedLayout::build(heights, n_stack).unwrap();
    let committed =
        StackedCommitment::<F, H>::commit(layout.clone(), &borrow(&columns), None, cfg).unwrap();
    let roots = committed.roots();
    let domain = committed.domain().clone();
    let stacked = match drop {
        None => committed,
        Some(drop) => committed
            .retire(TreeDrop::uniform(drop), cfg)
            .unwrap()
            .revive::<H, _>(&borrow(&columns), None, cfg)
            .unwrap(),
    };
    let proof = prove::<F, Ext, _, H, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        cfg,
        &mut sponge::<H>(),
    )
    .unwrap();
    Opened {
        layout,
        roots,
        points,
        values,
        domain,
        proof,
    }
}

fn check<H: WhirHash>(
    o: &Opened,
    proof: &StackedProof<F, Ext>,
    cfg: &ChainConfig,
) -> Result<(), Refusal> {
    verify_traced::<F, Ext, _, H>(
        proof,
        &o.layout,
        &o.roots,
        &Claimed::PerColumn(&o.points),
        &o.values,
        &o.domain,
        cfg,
        &mut sponge::<H>(),
    )
}

/// A refusal in a chain by a round-0 opening, per polynomial or grouped.
fn refused_by_an_opening(r: &Result<(), Refusal>) -> bool {
    matches!(
        r,
        Err(Refusal::Chain {
            cause: Error::OpeningRejected { .. } | Error::GroupedOpeningRejected { .. },
            ..
        })
    )
}

/// The shapes every test below runs: a first fold of 6 over an 11-variable
/// stack (round 0 opens 64-wide blocks of a 2^7-leaf tree), a uniform chain,
/// and a chain whose round 0 is its last (its openings checked by the final
/// round's verifier).
const SHAPES: [(Option<usize>, usize, usize); 3] =
    [(Some(6), 4, 11), (None, 3, 10), (Some(6), 4, 6)];

fn heights(n_stack: usize) -> Vec<usize> {
    vec![n_stack, n_stack - 1, n_stack - 2, n_stack - 2, n_stack - 1]
}

#[test]
fn the_coset_position_is_an_involution_that_keeps_an_index_past_the_tree() {
    for depth in 0..=10usize {
        let positions: Vec<usize> = (0..1usize << depth)
            .map(|j| LeafOrder::Coset.position(j, depth))
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..1usize << depth).collect::<Vec<_>>(),
            "a permutation"
        );
        for (j, &p) in positions.iter().enumerate() {
            assert_eq!(LeafOrder::Coset.position(p, depth), j, "its own inverse");
            assert_eq!(LeafOrder::Natural.position(j, depth), j);
        }
        let past = (1usize << depth) + 3;
        assert_eq!(
            LeafOrder::Coset.position(past, depth),
            past,
            "left for the range check"
        );
    }
    assert_eq!(LeafOrder::Coset.position(1, 3), 4);
    assert_eq!(LeafOrder::Coset.position(3, 3), 6);
    assert_eq!(LeafOrder::Coset.position(6, 5), 12);
}

/// The coset tree holds block `j` at `bitrev(j)` and nowhere else: its opening
/// of `j` authenticates at that position, not at `j`; the natural tree's at
/// `j`.
#[test]
fn a_coset_tree_holds_block_j_at_its_reversed_position() {
    type H = P1Whir;
    let log_folding = 2;
    let codeword: Vec<FE> = (0..1u64 << 9).map(|i| FE::from(i * i + 17)).collect();
    let natural = CodewordCommitment::<F, H>::from_codeword_on_host_in(
        codeword.clone(),
        log_folding,
        LeafOrder::Natural,
    )
    .unwrap();
    let coset = CodewordCommitment::<F, H>::from_codeword_on_host_in(
        codeword,
        log_folding,
        LeafOrder::Coset,
    )
    .unwrap();
    assert_ne!(natural.root(), coset.root(), "the order moves the root");
    assert_eq!(coset.leaves(), LeafOrder::Coset);
    let depth = coset.depth();
    for j in [1usize, 2, 5, 6, 77, 100] {
        let at = LeafOrder::Coset.position(j, depth);
        assert_ne!(at, j, "a block whose position moves");
        let opened = coset.open(j).unwrap();
        assert_eq!(
            opened.values,
            natural.open(j).unwrap().values,
            "the same block"
        );
        assert!(
            verify_opening::<F, H>(&coset.root(), depth, at, &opened),
            "block {j} at {at}"
        );
        assert!(
            !verify_opening::<F, H>(&coset.root(), depth, j, &opened),
            "block {j} not at {j}"
        );
        assert!(verify_opening::<F, H>(
            &natural.root(),
            depth,
            j,
            &natural.open(j).unwrap()
        ));
    }
}

/// A coset stack verifies under the coset config, whole or retired to any
/// kept top (the paths re-hashed from the kept blocks), and the retired
/// proofs are the whole tree's, byte for byte. Per polynomial and grouped,
/// uncapped and capped.
#[test]
fn a_coset_stack_verifies_whole_or_retired() {
    type H = P1Whir;
    for (first, log_folding, n_stack) in SHAPES {
        for group in [GroupFold::PerPoly, GroupFold::Indexed] {
            for cap in [CapPolicy::Off, CapPolicy::Auto] {
                let cfg = config(first, log_folding, group, LeafOrder::Coset, cap);
                let whole = open::<H>(&heights(n_stack), n_stack, &cfg, None);
                let label = format!("{first:?}/{log_folding}/{n_stack} {group:?} {cap:?}");
                assert_eq!(check::<H>(&whole, &whole.proof, &cfg), Ok(()), "{label}");
                let want = format!("{:?}", whole.proof);
                for drop in [1usize, 2, 3] {
                    let retired = open::<H>(&heights(n_stack), n_stack, &cfg, Some(drop));
                    assert_eq!(retired.roots, whole.roots, "{label} drop {drop}");
                    assert_eq!(format!("{:?}", retired.proof), want, "{label} drop {drop}");
                }
            }
        }
    }
}

/// ★ Each side's map is load-bearing. A verifier reading the natural order
/// (the map dropped from the verifier) refuses a coset proof at a round-0
/// opening; a coset verifier refuses a proof whose trees are natural (the map
/// dropped from the prover). The roots differ too, so the natural verifier is
/// handed the coset roots: only the position can refuse.
///
/// On the two shapes whose round-0 trees have 2^7 and 2^9 leaves: a query
/// whose index is its own bit reversal opens the same leaf under either order
/// (as it should), and a chain whose nine queries all are has probability
/// 8^-9 there; on the 4-leaf tree of the one-round shape it is 2^-9 a chain.
#[test]
fn each_side_of_the_order_is_load_bearing() {
    type H = P1Whir;
    for (first, log_folding, n_stack) in SHAPES.into_iter().take(2) {
        for group in [GroupFold::PerPoly, GroupFold::Indexed] {
            let coset = config(first, log_folding, group, LeafOrder::Coset, CapPolicy::Off);
            let natural = config(
                first,
                log_folding,
                group,
                LeafOrder::Natural,
                CapPolicy::Off,
            );
            let label = format!("{first:?}/{log_folding}/{n_stack} {group:?}");
            let made_coset = open::<H>(&heights(n_stack), n_stack, &coset, None);
            let r = check::<H>(&made_coset, &made_coset.proof, &natural);
            assert!(
                refused_by_an_opening(&r),
                "{label}: a natural verifier: {r:?}"
            );
            let made_natural = open::<H>(&heights(n_stack), n_stack, &natural, None);
            assert_eq!(
                check::<H>(&made_natural, &made_natural.proof, &natural),
                Ok(()),
                "{label}"
            );
            let r = check::<H>(&made_natural, &made_natural.proof, &coset);
            assert!(
                refused_by_an_opening(&r),
                "{label}: a coset verifier: {r:?}"
            );
        }
    }
}

/// One sibling byte of a round-0 path moved: refused, by that opening.
#[test]
fn a_tampered_round0_sibling_is_refused_under_the_coset_order() {
    for (first, log_folding, n_stack) in SHAPES {
        let cfg = config(
            first,
            log_folding,
            GroupFold::PerPoly,
            LeafOrder::Coset,
            CapPolicy::Off,
        );
        let honest = open::<P1Whir>(&heights(n_stack), n_stack, &cfg, None);
        let mut proof = honest.proof.clone();
        let opening = match &mut proof.polys[0].rounds[0].openings {
            crate::whir_chain::RoundOpenings::Base(round) => &mut round.current[1],
            other => panic!("a per-polynomial round 0 opens base blocks: {other:?}"),
        };
        opening.proof.merkle_path[0][7] ^= 1;
        let r = check::<P1Whir>(&honest, &proof, &cfg);
        assert!(refused_by_an_opening(&r), "{first:?}/{n_stack}: {r:?}");
    }
}

/// The order is part of the statement's fold word, on top of every other
/// word, and never part of the default's.
#[test]
fn the_coset_order_has_its_own_fold_word_and_is_not_the_default() {
    assert!(ChainFormat::DEFAULT.is_default());
    assert_eq!(ChainFormat::DEFAULT.leaves, LeafOrder::Natural);
    let mut words = Vec::new();
    for first in [None, Some(6)] {
        for group in [GroupFold::PerPoly, GroupFold::Indexed] {
            let natural = config(first, 4, group, LeafOrder::Natural, CapPolicy::Off);
            let coset = config(first, 4, group, LeafOrder::Coset, CapPolicy::Off);
            assert!(!coset.format.is_default());
            assert_eq!(
                coset.fold_word(),
                natural.fold_word() | FOLD_WORD_TAG | FOLD_WORD_COSET,
                "{first:?} {group:?}"
            );
            assert_eq!(natural.fold_word() & FOLD_WORD_COSET, 0);
            words.push(natural.fold_word());
            words.push(coset.fold_word());
        }
    }
    assert_eq!(
        config(
            None,
            4,
            GroupFold::PerPoly,
            LeafOrder::Natural,
            CapPolicy::Off
        )
        .fold_word(),
        4
    );
    let mut distinct = words.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), words.len(), "eight formats, eight words");
    assert_ne!(FOLD_WORD_COSET, FOLD_WORD_INDEXED);
}

/// ★ The coset order is Poseidon1's alone: under Keccak and RPX trees the
/// config is refused by name before a tree is built (a stack's commit, a
/// chain's commit) and before a proof is read (a stack's verifier, a chain's).
fn the_coset_order_is_refused_by_name<H: WhirHash>() {
    let cfg = config(
        Some(6),
        4,
        GroupFold::PerPoly,
        LeafOrder::Coset,
        CapPolicy::Off,
    );
    let refused = || crate::Error::LeafOrderUnsupported {
        order: LeafOrder::Coset,
        hash: H::NAME,
    };
    let columns: Vec<Mle<F>> = (0..3).map(|i| column(10 - i, 9 + i as u64)).collect();
    let layout = StackedLayout::build(&[10, 9, 8], 11).unwrap();
    assert_eq!(
        StackedCommitment::<F, H>::commit(layout, &borrow(&columns), None, &cfg).err(),
        Some(refused()),
        "{}: a stack's commit",
        H::NAME
    );
    assert_eq!(
        whir_chain::commit::<F, H>(&columns[0], &cfg, true).err(),
        Some(refused()),
        "{}: a chain's commit",
        H::NAME
    );
    // A proof made where the order is defined, read under `H`.
    let made = open::<P1Whir>(&heights(11), 11, &cfg, None);
    let r = check::<H>(&made, &made.proof, &cfg);
    assert!(
        matches!(&r, Err(Refusal::Before(e) | Refusal::Chain { cause: e, .. }) if *e == refused()),
        "{}: a stack's verifier: {r:?}",
        H::NAME
    );
    let chain = &made.proof.polys[0];
    let z: Vec<EE> = (0..11).map(|i| EE::from(3 + i as u64)).collect();
    assert_eq!(
        whir_chain::verify::<F, Ext, _, H>(
            chain,
            &made.roots[0],
            &z,
            EE::from(1),
            &made.domain,
            &cfg,
            &mut sponge::<H>(),
        )
        .err(),
        Some(refused()),
        "{}: a chain's verifier",
        H::NAME
    );
    // The natural order stays every hash's.
    let natural = config(
        Some(6),
        4,
        GroupFold::PerPoly,
        LeafOrder::Natural,
        CapPolicy::Off,
    );
    assert!(whir_chain::commit::<F, H>(&columns[0], &natural, true).is_ok());
}

#[test]
fn the_coset_order_is_refused_by_name_under_keccak() {
    the_coset_order_is_refused_by_name::<KeccakWhir>();
}

#[test]
fn the_coset_order_is_refused_by_name_under_rpx() {
    the_coset_order_is_refused_by_name::<RpxWhir>();
}

/// A coset round 0 whose successor was not encoded is refused by cause: the
/// knob off whatever the factors, then factors on the host, then the card
/// declining. (The paths themselves need a card: `tests/whir_coset_device.rs`.)
#[test]
fn an_unencoded_successor_is_named_by_its_cause() {
    assert_eq!(
        UnencodedSuccessor::of(false, true),
        UnencodedSuccessor::KnobOff
    );
    assert_eq!(
        UnencodedSuccessor::of(false, false),
        UnencodedSuccessor::KnobOff
    );
    assert_eq!(
        UnencodedSuccessor::of(true, false),
        UnencodedSuccessor::HostFactors
    );
    assert_eq!(
        UnencodedSuccessor::of(true, true),
        UnencodedSuccessor::CardDeclined
    );
}

/// A round-0 commitment placed in the natural order is not proved under a
/// coset config (nor the reverse): the prover refuses by name rather than
/// emit paths the verifier would read at other positions.
#[test]
fn a_commitment_in_another_order_is_not_proved() {
    let coset = config(
        Some(6),
        4,
        GroupFold::PerPoly,
        LeafOrder::Coset,
        CapPolicy::Off,
    );
    let natural = config(
        Some(6),
        4,
        GroupFold::PerPoly,
        LeafOrder::Natural,
        CapPolicy::Off,
    );
    let f = column(11, 4);
    let z: Vec<EE> = (0..11).map(|i| EE::from(5 + 2 * i as u64)).collect();
    for (committed_under, proved_under) in [(&natural, &coset), (&coset, &natural)] {
        let (commitment, domain) =
            whir_chain::commit::<F, P1Whir>(&f, committed_under, true).unwrap();
        let r = whir_chain::prove::<F, Ext, _, P1Whir>(
            &f,
            &z,
            &commitment,
            &domain,
            proved_under,
            &mut sponge::<P1Whir>(),
        );
        assert_eq!(
            r.err(),
            Some(crate::Error::LeafOrderMismatch {
                expected: proved_under.format.leaves,
                got: committed_under.format.leaves,
            })
        );
    }
}
