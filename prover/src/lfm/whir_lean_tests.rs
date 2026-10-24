//! Gates for the leaf's cheaper query check ([`LeafQuery::Lean`]) on Poseidon1
//! chains: the tail rounds checked at every leaf once
//! ([`ChainShape::tail_round`]), against the host's own proofs.
//!
//! The fixtures are `whir_p1_leaf_tests`' small chains with enough queries that
//! their last trees, lying wholly in their caps, take [`TAIL_QUERIES_A_LEAF`]
//! queries a leaf.

use multilinear::whir_chain::{CapPolicy, ChainConfig, ChainProof, LeafOrder, RoundOpenings};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::edsl::WrapHash;
use super::executor::execute;
use super::hash::HasherKind;
use super::whir_chain::{
    ChainShape, LeafQuery, TAIL_QUERIES_A_LEAF, chain_perms, round_words, tail_faults,
    try_push_round_words,
};
use super::whir_p1_leaf_tests::{
    Fixture, fixture, hash16_rows, host_accepts, p1_chain_program, p1_config, p1_ordered_config,
};
use super::whir_transcript::SpongeEntry;
use super::word::{LfmWord, ext_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

const SOCKET: HasherKind = HasherKind::Poseidon1W16;

/// `shape` under [`LeafQuery::Lean`].
fn lean(shape: &ChainShape) -> ChainShape {
    ChainShape {
        query: LeafQuery::Lean,
        ..shape.clone()
    }
}

/// The chain's arena laid out at `shape` (the fixture's own or its lean
/// twin), as `whir_p1_leaf_tests::p1_chain_arena` lays it at the fixture's.
fn arena_at(
    f: &Fixture,
    proof: &ChainProof<F, E>,
    shape: &ChainShape,
) -> Result<Vec<LfmWord>, String> {
    let mut words: Vec<LfmWord> = f.z.iter().map(ext_word).collect();
    words.push(ext_word(&f.y));
    words.push(commitment_to_digest(&f.root_bytes));
    words.push(ext_word(&proof.final_value));
    try_push_round_words(&mut words, shape, proof)?;
    Ok(words)
}

/// Chains with tail rounds: uniform folds of 4 at the C2 cap (rounds 1 and 2,
/// 8 and 4 leaves, at 192 queries; round 2 alone at 96), and the production's
/// first fold of 6 under the production cap (its last round).
fn tail_cases() -> Vec<(usize, ChainConfig)> {
    vec![
        (9, p1_config(192, 0, CapPolicy::Fixed(2))),
        (9, p1_config(96, 6, CapPolicy::Fixed(2))),
        (
            10,
            p1_ordered_config(96, 0, CapPolicy::Auto, Some(6), LeafOrder::Coset),
        ),
        (
            11,
            p1_ordered_config(100, 6, CapPolicy::Auto, Some(6), LeafOrder::Coset),
        ),
    ]
}

/// The tail rounds of `shape`, which must have some.
fn tails(shape: &ChainShape) -> Vec<usize> {
    let tails: Vec<usize> = (0..shape.rounds())
        .filter(|&r| shape.tail_round(r))
        .collect();
    assert!(
        !tails.is_empty(),
        "depths {:?} caps {:?} at {} queries: no tail round",
        (0..shape.rounds())
            .map(|r| shape.current_depth(r))
            .collect::<Vec<_>>(),
        shape.caps,
        shape.num_queries
    );
    tails
}

/// The arena offset of tail round `r`'s first block: the fixture's prefix
/// (`z`, `y`, the root, the final value), the rounds before, then round `r`'s
/// sumcheck words and nonces.
fn tail_block_at(shape: &ChainShape, r: usize) -> usize {
    let prefix = shape.num_vars + 3;
    let before: u32 = (0..r).map(|t| round_words(shape, t)).sum();
    let nonces = shape.carries(r).iter().filter(|&&c| c).count();
    prefix + before as usize + 2 * shape.messages(r) + nonces
}

/// Fewer than [`TAIL_QUERIES_A_LEAF`] queries a leaf of any tree: a lean
/// chain with no tail round.
fn untailed() -> (usize, ChainConfig) {
    (9, p1_config(12, 0, CapPolicy::Fixed(2)))
}

/// ★ Under `Lean` the tail rule picks a suffix of rounds ≥ 1, every one of
/// them wholly in its cap with [`TAIL_QUERIES_A_LEAF`] queries a leaf, and
/// none under `V1`.
#[test]
fn the_tail_rounds_are_a_capped_suffix_and_only_under_lean() {
    for (num_vars, cfg) in tail_cases() {
        let shape = ChainShape::new_at(&cfg, num_vars, 4);
        assert!(
            (0..shape.rounds()).all(|r| !shape.tail_round(r)),
            "V1 has no tail round"
        );
        let lean = lean(&shape);
        let tails = tails(&lean);
        assert!(tails[0] >= 1, "round 0 is never a tail round");
        assert_eq!(
            *tails.last().unwrap(),
            lean.rounds() - 1,
            "a suffix of the chain"
        );
        assert!(tails.windows(2).all(|w| w[1] == w[0] + 1), "contiguous");
        for &r in &tails {
            assert!(TAIL_QUERIES_A_LEAF * lean.current_leaves(r) <= lean.num_queries);
            assert_eq!(lean.current_path(r), 0, "no walked level");
        }
    }
    // One query short of round 1's bound (8 leaves): round 2 alone is a tail.
    let (num_vars, cfg) = tail_cases().remove(0);
    let short = lean(&ChainShape::new_at(
        &ChainConfig {
            num_queries: cfg.num_queries - 1,
            ..cfg
        },
        num_vars,
        4,
    ));
    assert_eq!(tails(&short), vec![2], "8 leaves need 192 queries");
    let (num_vars, cfg) = untailed();
    let few = lean(&ChainShape::new_at(&cfg, num_vars, 4));
    assert!((0..few.rounds()).all(|r| !few.tail_round(r)));
}

/// ★ The lean chain executes on every proof the host accepts, and spends its
/// closed form of socket rows, fewer than the V1 chain's.
#[test]
fn a_lean_p1_chain_executes_on_a_proof_the_host_accepts() {
    let entry = SpongeEntry::fresh_for(WrapHash::Poseidon1);
    for (num_vars, cfg) in tail_cases() {
        let f = fixture(num_vars, cfg);
        let shape = lean(&f.shape);
        let tails = tails(&shape);
        let program = p1_chain_program(&shape);
        let arena = arena_at(&f, &f.proof, &shape).expect("an honest proof lays out every leaf");
        execute(&program, &[arena], &SOCKET).unwrap_or_else(|e| {
            panic!("S={num_vars} tails {tails:?}: the machine refused an accepted proof: {e:?}")
        });
        let v1 = p1_chain_program(&f.shape);
        assert_eq!(
            hash16_rows(&program),
            chain_perms(&shape, entry),
            "S={num_vars}: Hash16 rows against the lean closed form"
        );
        assert!(
            hash16_rows(&program) < hash16_rows(&v1),
            "S={num_vars}: lean {} socket rows against V1 {}",
            hash16_rows(&program),
            hash16_rows(&v1)
        );
    }
}

/// ★ A tail block that is not its leaf (one word off in the arena) misses its
/// cap node and has no execution; the final value one off has none either.
#[test]
fn a_lean_chain_refuses_a_tampered_tail_block() {
    for (num_vars, cfg) in tail_cases() {
        let f = fixture(num_vars, cfg);
        let shape = lean(&f.shape);
        let program = p1_chain_program(&shape);
        let honest = arena_at(&f, &f.proof, &shape).expect("laid out");
        assert!(execute(&program, &[honest.clone()], &SOCKET).is_ok());
        for r in tails(&shape) {
            let at = tail_block_at(&shape, r);
            for leaf in [0, shape.current_leaves(r) - 1] {
                let mut forged = honest.clone();
                let word = at + leaf * (1usize << shape.schedule[r]);
                forged[word][1] += FE::from(1u64);
                assert!(
                    execute(&program, &[forged], &SOCKET).is_err(),
                    "S={num_vars}: tail round {r}'s leaf {leaf} one lane off must be refused"
                );
            }
        }
        let mut forged = f.proof.clone();
        forged.final_value += FEE::one();
        assert!(!host_accepts(&f, &forged));
        let arena = arena_at(&f, &forged, &shape).expect("laid out");
        assert!(
            execute(&program, &[arena], &SOCKET).is_err(),
            "S={num_vars}: the final value one off must be refused"
        );
    }
}

/// ★ The tail compare is live and reads the slot its position names: comparing
/// with the neighbouring slot of the successor block, an honest proof has no
/// execution.
#[test]
fn a_tail_compare_at_the_wrong_slot_refuses_an_honest_proof() {
    let mut compared = 0;
    for (num_vars, cfg) in tail_cases() {
        let f = fixture(num_vars, cfg);
        let shape = lean(&f.shape);
        // Only a tail round with a successor compares a slot.
        if !tails(&shape).iter().any(|&r| shape.next_depth(r).is_some()) {
            continue;
        }
        let arena = arena_at(&f, &f.proof, &shape).expect("laid out");
        let faulted = tail_faults::with_slot_off(|| p1_chain_program(&shape));
        assert!(
            execute(&faulted, &[arena.clone()], &SOCKET).is_err(),
            "S={num_vars}: the neighbouring slot must be refused"
        );
        assert!(execute(&p1_chain_program(&shape), &[arena], &SOCKET).is_ok());
        compared += 1;
    }
    assert!(compared > 0, "some case has a tail round with a successor");
}

/// Every opening of tail round `r`'s leaf `leaf` in `proof` (the round
/// before's successor openings and the round's own), each value one off: no
/// opening hashes to that leaf's cap node any more.
fn spoil_leaf(proof: &mut ChainProof<F, E>, shape: &ChainShape, r: usize, digest: [u8; 32]) {
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    type Leaf = <crate::hash_pin::BlockWhirHash as multilinear::whir_hash::WhirHash>::Backend<E>;
    let _ = shape;
    let spoil = |openings: &mut Vec<multilinear::whir_commit::CosetOpening<E>>| {
        for o in openings.iter_mut() {
            if Leaf::hash_data(&o.values) == digest {
                o.values[0] += FEE::one();
            }
        }
    };
    match &mut proof.rounds[r - 1].openings {
        RoundOpenings::Base(p) => spoil(&mut p.next),
        RoundOpenings::Extension(p) => spoil(&mut p.next),
        RoundOpenings::BaseGrouped(p) => spoil(&mut p.next),
    }
    match &mut proof.rounds[r].openings {
        RoundOpenings::Extension(p) => spoil(&mut p.current),
        _ => panic!("a tail round's openings are extension-valued"),
    }
}

/// ★ A proof that opens no block of some tail leaf (every such opening spoilt)
/// makes the arena writer return an error naming the leaf, not a panic, and
/// lays out no arena; under V1 the same proof still lays out (its paths, not
/// the tail, are what refuse it).
#[test]
fn a_lean_arena_refuses_a_proof_missing_a_tail_leaf() {
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    type Leaf = <crate::hash_pin::BlockWhirHash as multilinear::whir_hash::WhirHash>::Backend<E>;
    let (num_vars, cfg) = tail_cases().remove(0);
    let f = fixture(num_vars, cfg);
    let shape = lean(&f.shape);
    let r = *tails(&shape).last().expect("a tail round");
    // Leaf 0's digest: the hash of any opened block placed first.
    let honest = arena_at(&f, &f.proof, &shape).expect("laid out");
    let at = tail_block_at(&shape, r);
    let block = 1usize << shape.schedule[r];
    let values: Vec<math::field::element::FieldElement<E>> = honest[at..at + block]
        .iter()
        .map(|w| FEE::new([w[0], w[1], w[2]]))
        .collect();
    let digest = Leaf::hash_data(&values);
    let mut spoilt = f.proof.clone();
    spoil_leaf(&mut spoilt, &shape, r, digest);
    let err = arena_at(&f, &spoilt, &shape).expect_err("a missing leaf is refused");
    assert!(
        err.contains(&format!("tail round {r}: no opening of leaf 0")),
        "{err}"
    );
    assert!(
        arena_at(&f, &spoilt, &f.shape).is_ok(),
        "V1 lays the paths out"
    );
}

/// ★ Equal tail blocks (a constant polynomial folds to a constant codeword)
/// share one cap digest; each leaf takes a block that hashes to it, and the
/// lean chain executes.
#[test]
fn equal_tail_blocks_are_each_placed() {
    let cfg = p1_config(96, 0, CapPolicy::Fixed(2));
    let f = super::whir_p1_leaf_tests::constant_fixture(9, cfg);
    let shape = lean(&f.shape);
    let program = p1_chain_program(&shape);
    let arena = arena_at(&f, &f.proof, &shape).expect("every leaf placed");
    execute(&program, &[arena], &SOCKET).expect("the lean chain executes");
}
