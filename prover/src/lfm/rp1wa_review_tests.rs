//! REV-P1W-A (break reviewer) probes of the Poseidon1 WHIR leaf: the width-8
//! grind check against the host's accept set, bit for bit, and the
//! child-order 4-ary walk + cap mux against the host's trees at every index.

use crate::tables::types::{FE, GoldilocksField};
use crypto::merkle_tree::merkle::MerkleTree;
use stark::config::Commitment;

use super::algebraic_commit::{commitment_to_digest, digest_to_commitment};
use super::builder::LfmBuilder;
use super::compiler::compile;
use super::edsl::{WrapDigest, WrapHash};
use super::executor::execute_serial;
use super::hash::HasherKind;
use super::merkle_cap::CapCells;
use super::p1_commit::P1GrindDigest;
use super::word::{LfmWord, base_word};

type B = super::p1_commit::P1BatchBackend<GoldilocksField>;

fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn lanes(seed: u64) -> [FE; 4] {
    core::array::from_fn(|k| FE::from(mix(seed, k as u64)))
}

/// The emitted Poseidon1 grind check over `(seed, nonce, factor)`: Ok iff it
/// executes.
fn guest_grind(seed: [FE; 4], nonce: u64, factor: u8) -> Result<(), String> {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
    let a_seed = b.declare_arena(1);
    let a_nonce = b.declare_arena(1);
    let s = WrapDigest::from_cell(b.hint_word(a_seed, 0));
    let n = b.hint_felt(a_nonce, 0);
    super::epoch::emit_grinding_check(&mut b, s, n, factor);
    let program = compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("{e:?}"))?;
    execute_serial(
        &program,
        &[vec![seed], vec![base_word(FE::from(nonce))]],
        &HasherKind::Poseidon1W16,
    )
    .map(|_| ())
    .map_err(|e| format!("{e:?}"))
}

/// ★ The sparse width-8 grind check accepts EXACTLY the nonces the host's
/// `is_valid_nonce::<P1GrindDigest>` accepts: over many nonces at several
/// factors and seeds (lanes near `p` included), and, sharper than one good
/// and one bad nonce, at the boundary — a nonce valid at `g − 1` bits and not
/// at `g` is refused at `g` and accepted at `g − 1` (an off-by-one in the bit
/// count or a low-bit check would pass the existing pair). Nonces past `p`
/// reduce on both sides alike.
#[test]
fn rp1wa_the_emitted_grind_accepts_exactly_the_hosts_nonces() {
    let p = 0xFFFF_FFFF_0000_0001u64;
    let mut seeds: Vec<[FE; 4]> = (0..3).map(|s| lanes(900 + s)).collect();
    seeds.push([
        FE::from(p - 1),
        FE::from(p - 2),
        FE::zero(),
        FE::from(1u64 << 63),
    ]);
    let mut boundary_hits = 0;
    for seed in &seeds {
        let bytes = digest_to_commitment(seed);
        for factor in [1u8, 2, 3, 5, 7] {
            let host =
                |n: u64| crypto::grinding::is_valid_nonce::<P1GrindDigest>(&bytes, n, factor);
            let mut nonces: Vec<u64> = (0..48).collect();
            nonces.extend([p, p + 1, p + 5, u64::MAX, u64::MAX - 3]);
            for n in nonces {
                assert_eq!(
                    guest_grind(*seed, n, factor).is_ok(),
                    host(n),
                    "factor {factor} nonce {n}: guest and host disagree"
                );
            }
            if factor >= 2 {
                let edge = (0u64..1 << 16).find(|&n| {
                    crypto::grinding::is_valid_nonce::<P1GrindDigest>(&bytes, n, factor - 1)
                        && !host(n)
                });
                if let Some(n) = edge {
                    boundary_hits += 1;
                    assert!(
                        guest_grind(*seed, n, factor).is_err(),
                        "nonce {n} has exactly {} leading zero bits and passed at {factor}",
                        factor - 1
                    );
                    assert!(
                        guest_grind(*seed, n, factor - 1).is_ok(),
                        "nonce {n} at {}",
                        factor - 1
                    );
                }
            }
        }
    }
    assert!(
        boundary_hits >= 12,
        "the boundary was probed: {boundary_hits}"
    );
}

/// One opening of a host Poseidon1 tree over `2^depth` leaves checked in-guest
/// the way the WHIR leaf checks it: the owner path as the proof carries it
/// (`embed_cap_arity`), laid out in CHILD order by `child_path_to_cap`, the cap
/// authenticated and the opening checked through `verify_path_child` (or, at
/// `c = 0`, `walk4_child` to the root). `bits_of` is the index whose bits the
/// guest walks (the transcript's), `index` the leaf whose path is supplied.
/// `tamper` moves one arena word.
fn child_opening(
    depth: usize,
    c: usize,
    index: usize,
    bits_of: usize,
    tamper: Option<(usize, usize)>,
) -> Result<(), String> {
    let n = 1usize << depth;
    let leaves: Vec<Commitment> = (0..n)
        .map(|i| digest_to_commitment(&lanes(7_000 + depth as u64 * 131 + i as u64)))
        .collect();
    let tree = MerkleTree::<B>::build_from_hashed_leaves(leaves.clone()).expect("a tree");
    let mut path = tree
        .get_proof_by_pos(index)
        .expect("a leaf")
        .merkle_path
        .clone();
    let cap = if c == 0 {
        Vec::new()
    } else {
        tree.cap(c).expect("the cap")
    };
    if c > 0 {
        crypto::merkle_tree::cap::embed_cap_arity(&mut [&mut path], depth, &cap, 4)
            .expect("the owner path");
    }
    let (hints, split) =
        super::merkle_cap::child_path_to_cap(&path, depth, c, 4, true).expect("laid out");
    assert_eq!(split.unwrap_or_default(), cap);
    assert_eq!(hints.len(), super::merkle_cap::path_hints(depth, c, 4));

    let word = |c: &Commitment| -> LfmWord { commitment_to_digest(c) };
    let mut arenas: Vec<Vec<LfmWord>> = vec![
        vec![word(&leaves[index])],
        vec![base_word(FE::from(bits_of as u64))],
        hints.iter().map(word).collect(),
        cap.iter().map(word).collect(),
        vec![word(&tree.root)],
    ];
    if let Some((a, w)) = tamper {
        arenas[a][w][1] += FE::one();
    }
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
    let lens: Vec<u32> = arenas.iter().map(|a| a.len() as u32).collect();
    let ids: Vec<_> = lens.iter().map(|&l| b.declare_arena(l)).collect();
    let leaf = WrapDigest::from_cell(b.hint_word(ids[0], 0));
    let idx = b.hint_felt(ids[1], 0);
    let bits = b.bit_dec(idx, depth);
    let h: Vec<WrapDigest> = (0..lens[2])
        .map(|i| WrapDigest::from_cell(b.hint_word(ids[2], i)))
        .collect();
    let root = b.hint_word(ids[4], 0);
    let root_lanes = [b.unpack(root)];
    if c == 0 {
        let walked = super::p1w16_emit::walk4_child(&mut b, leaf, &bits, &h);
        super::edsl::assert_digest_eq_lanes(&mut b, walked, &root_lanes);
    } else {
        let nodes: Vec<WrapDigest> = (0..lens[3])
            .map(|i| WrapDigest::from_cell(b.hint_word(ids[3], i)))
            .collect();
        let capped = CapCells::authenticate_at(&mut b, &nodes, c, &root_lanes);
        capped.verify_path_child(&mut b, leaf, &bits, &h);
    }
    let program = compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("{e:?}"))?;
    execute_serial(&program, &arenas, &HasherKind::Poseidon1W16)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

/// ★ The production walk (child order, two placing `Select`s a level) and the
/// 4-ary cap mux against the host's trees, EXHAUSTIVELY at small depths: at
/// every depth 1–7 (both parities), every cap height up to the tree's levels,
/// and EVERY leaf index, the honest opening executes; the same path walked
/// with another index's bits does not (the index places the node); and every
/// arena word the check reads — leaf, each sibling, each cap node (queried or
/// not), the root — moved refuses it.
#[test]
fn rp1wa_the_child_order_walk_and_cap_bind_every_index() {
    for depth in 1usize..=7 {
        let n = 1usize << depth;
        let levels = crypto::merkle_tree::cap::tree_levels(depth, 4);
        for c in 0..=levels {
            let hints = super::merkle_cap::path_hints(depth, c, 4);
            let nodes = if c == 0 {
                0
            } else {
                super::merkle_cap::cap_nodes(depth, c, 4)
            };
            for index in 0..n {
                child_opening(depth, c, index, index, None)
                    .unwrap_or_else(|e| panic!("depth {depth} cap {c} index {index}: {e}"));
                let other = (index + 1 + index % 3) % n;
                if other != index {
                    assert!(
                        child_opening(depth, c, index, other, None).is_err(),
                        "depth {depth} cap {c}: leaf {index}'s path walked at index {other}"
                    );
                }
                // Every word, at a spread of indices (all of them at depth <= 5).
                if depth <= 5 || index % 9 == 0 {
                    let mut words = vec![(0usize, 0usize), (4, 0)];
                    words.extend((0..hints).map(|w| (2, w)));
                    words.extend((0..nodes).map(|w| (3, w)));
                    for (a, w) in words {
                        assert!(
                            child_opening(depth, c, index, index, Some((a, w))).is_err(),
                            "depth {depth} cap {c} index {index}: arena {a} word {w} moved"
                        );
                    }
                }
            }
        }
    }
}
