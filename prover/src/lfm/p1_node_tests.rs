//! A node verifies a Poseidon1 LFM child proof in-guest (D-P3B P2): the
//! trivial program proved under `hash_pin::P1Family`, harvested into the
//! node's arenas, and one leg (`per_table_aggregator::emit_leg`) emitted under
//! `WrapHash::Poseidon1` — ZisK's transcript replay with the family's
//! statement tag, the width-8 grind check, the tagged leaf hash and the 4-ary
//! hint-order walks (`p1w16_emit::walk4`) to each tree's cap
//! (`P1_STARK_CAP`). The honest arenas execute; every tamper below is refused
//! by the machine, never by the host harvest. The node-level checks that one
//! constraint alone holds — the grind at the node's factor, the cap-to-root
//! of every cap node no query reaches — each have a test of their own, and the
//! legs' opening (hint-order walk, cap mux and root) is checked exhaustively
//! against the host's trees at small depths.

use crypto::merkle_tree::merkle::MerkleTree;
use stark::config::Commitment;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::hash_pin::{P1Family, ProofFamily};
use crate::tables::types::{FE, GoldilocksField};

use super::algebraic_commit::{commitment_to_digest, digest_to_commitment};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl::{WrapDigest, WrapHash};
use super::executor::{LfmExecError, execute, execute_serial};
use super::harvest::{HarvestedChild, harvest_child_under, try_child_arena_words};
use super::hash::HasherKind;
use super::per_table_aggregator::{DerivedChild, declare_leg_arenas, emit_leg};
use super::programs::trivial_program;
use super::proof::{LfmProof, lfm_prove_under, verify_against_artifacts_under};
use super::registry::{LfmArtifacts, build_artifacts_under};
use super::word::{LfmWord, base_word};

/// The child's options: the production format's `fri = dp` at blowup 4, with
/// few queries so the node stays small, and a grind so the leg checks one.
/// Its binary cap policy is OFF: an arity-4 configuration caps its trees at
/// its own height (`P1_STARK_CAP`) whatever the format says, so a shape that
/// took the format's policy instead would lay the trees out uncapped and
/// refuse the child (at six queries `auto` could coincide with it and hide
/// that).
fn options() -> ProofOptions {
    options_at(4)
}

/// [`options`] at `grinding` bits.
fn options_at(grinding: u8) -> ProofOptions {
    let mut o = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
    o.fri_number_of_queries = 6;
    o.grinding_factor = grinding;
    o.fri_final_poly_log_degree = 2;
    let mut o = crate::zf_format::ZfFormat::DEFAULT.options(o);
    o.format.merkle_cap = stark::proof::options::CapPolicy::Off;
    o
}

fn arenas() -> Vec<Vec<LfmWord>> {
    arenas_salted(0)
}

/// The trivial program's arena, its words moved by `salt` (another
/// execution, another transcript).
fn arenas_salted(salt: u64) -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64 + 17 * salt)))
            .collect(),
    ]
}

struct P1Child {
    artifacts: LfmArtifacts,
    proof: LfmProof,
    opts: ProofOptions,
}

fn p1_child() -> P1Child {
    let opts = options();
    let program = trivial_program();
    let hasher = super::registry::build_artifacts(&program, &opts).hasher;
    let artifacts =
        build_artifacts_under::<<P1Family as ProofFamily>::Stark>(&program, &opts, hasher);
    let proof = lfm_prove_under::<P1Family>(&program, &artifacts, &arenas(), &opts).expect("prove");
    assert!(verify_against_artifacts_under::<P1Family>(
        &artifacts,
        &proof.proof,
        &proof.public_words,
        &opts
    ));
    P1Child {
        artifacts,
        proof,
        opts,
    }
}

fn harvest(c: &P1Child) -> HarvestedChild {
    harvest_child_under::<P1Family>(c.artifacts.clone(), c.opts.clone(), &c.proof)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// One leg over the child, under `wrap`, with the child's statement tag
/// `tag` (the family's, unless a mutation swaps it).
fn node(c: &P1Child, tag: &'static [u8]) -> LfmProgram {
    node_at(&c.artifacts, &c.opts, c.proof.public_words.len(), tag)
}

/// One leg over a child of `artifacts` whose shapes — the grind factor
/// among them — the node derives at `opts`.
fn node_at(
    artifacts: &LfmArtifacts,
    opts: &ProofOptions,
    words: usize,
    tag: &'static [u8],
) -> LfmProgram {
    let derived = DerivedChild::from_artifacts_under::<P1Family>(artifacts, opts, words)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut shape = derived.shape();
    shape.statement_tag = tag;
    let mut b = LfmBuilder::new().with_wrap_hash(P1Family::WRAP);
    let a = declare_leg_arenas(&mut b, &shape);
    let _ = emit_leg(&mut b, &shape, &a);
    compile(b.finish())
}

fn runs(program: &LfmProgram, child: &HarvestedChild) -> bool {
    let arenas = try_child_arena_words(child).unwrap_or_else(|e| panic!("{e}"));
    execute(program, &arenas, &HasherKind::Poseidon1W16).is_ok()
}

/// Whether the node refuses the child at one of its checks: the execution
/// fails, and not at the arenas' layout or the socket (a refusal of those
/// kinds would say nothing about the check a test isolates).
fn refused_by_a_check(program: &LfmProgram, child: &HarvestedChild) -> bool {
    let arenas = try_child_arena_words(child).unwrap_or_else(|e| panic!("{e}"));
    match execute(program, &arenas, &HasherKind::Poseidon1W16) {
        Ok(_) => false,
        Err(
            e @ (LfmExecError::ArenaCountMismatch { .. }
            | LfmExecError::ArenaLenMismatch { .. }
            | LfmExecError::ArenaOutOfBounds { .. }
            | LfmExecError::HasherRejected(_)
            | LfmExecError::Internal(_)),
        ) => panic!("refused before any check: {e:?}"),
        Err(_) => true,
    }
}

fn flip(c: &mut [u8; 32]) {
    c[31] ^= 1;
}

#[test]
fn a_p1_node_verifies_a_p1_child_and_refuses_its_tampers() {
    let c = p1_child();
    let honest = harvest(&c);
    let program = node(&c, P1Family::STATEMENT_TAG);
    assert!(
        program.hasher(crate::hash_pin::BLOCK_HASHER) == HasherKind::Poseidon1W16,
        "a P1 node's hash rows are the width-16 socket's"
    );
    assert!(runs(&program, &honest), "the honest child executes");
    // The child's arenas are laid at the family's digest width, the node's.
    assert_eq!(honest.wrap, P1Family::WRAP);
    assert!(honest.legs.iter().all(|l| l.wrap == P1Family::WRAP));
    // The P1 child's trees are capped (`P1_STARK_CAP`): a cap is hinted per
    // tree.
    assert!(
        honest.legs.iter().any(|l| !l.caps.is_empty()),
        "some table carries caps"
    );

    let refused = |what: &str, edit: &dyn Fn(&mut HarvestedChild)| {
        let mut bad = harvest(&c);
        edit(&mut bad);
        assert!(!runs(&program, &bad), "{what}: the node must refuse");
    };
    // A table whose trees are capped and whose FRI commits a layer.
    let t = honest
        .legs
        .iter()
        .position(|l| !l.caps.is_empty() && !l.fri_openings[0].is_empty())
        .expect("a capped table with a committed FRI layer");
    refused("main root", &|h| flip(&mut h.tables[t].main_root));
    refused("composition root", &|h| {
        flip(&mut h.tables[t].composition_root)
    });
    refused("trace sibling", &|h| {
        flip(&mut h.legs[t].openings[1][0].1[0])
    });
    refused("cap node", &|h| {
        let last = h.legs[t].caps.len() - 1;
        flip(&mut h.legs[t].caps[last])
    });
    refused("FRI sibling", &|h| {
        flip(&mut h.legs[t].fri_openings[2][0].1[0])
    });
    refused("FRI root", &|h| flip(&mut h.tables[t].fri_roots[0]));
    refused("opened value", &|h| {
        h.legs[t].openings[0][0].0[0][0] += FE::from(1u64)
    });
    refused("OOD value (the transcript)", &|h| {
        h.tables[t].ood_current[0] += crate::tables::types::FEE::from(1u64)
    });
    refused("nonce", &|h| {
        h.tables[t].nonce = h.tables[t].nonce.map(|n| n ^ 1)
    });
    // ★ The query index is the transcript's: swapping two queries' openings
    // (each internally consistent) moves every leaf to another index.
    refused("swapped queries", &|h| {
        h.legs[t].openings.swap(0, 1);
        h.legs[t].fri_openings.swap(0, 1);
    });
    refused("a claimed word", &|h| {
        h.public_words[0].1[0] += FE::from(1u64)
    });
}

/// The family's statement tag is load-bearing in the guest: a node that
/// replays the child's statement under today's tag (the legacy family's)
/// refuses the honest child.
#[test]
fn a_node_without_the_family_tag_refuses_the_p1_child() {
    let c = p1_child();
    let honest = harvest(&c);
    assert!(runs(&node(&c, P1Family::STATEMENT_TAG), &honest));
    assert!(!runs(
        &node(&c, crate::hash_pin::LegacyFamily::STATEMENT_TAG),
        &honest
    ));
}

/// ★ The leg checks each table's grind at the factor the NODE derives from
/// the child's options, not at one the child chose (REV-P3B14-A A1): a child
/// proved at `g − 1` bits whose nonces do not all meet `g` is accepted by the
/// node derived at `g − 1` and refused by the one derived at `g`, as the host
/// verifier at `g` refuses it. `rp1wa_the_emitted_grind_accepts_exactly_the_
/// hosts_nonces` drives the grind gadget directly, and the tamper test's nonce
/// flip re-rolls every query, so neither sees the leg's call site
/// (`epoch::emit_table_challenges`) drop the check or skip it on the P1 wrap
/// hash; this does. (A weaker factor at the call site is no stealthy mutant:
/// the factor is part of the grind's hashed preimage, so the honest node
/// tests refuse too.) From the reviewers' short-grind tests (REV-P3B14).
#[test]
fn a_p1_node_refuses_a_child_ground_one_bit_short() {
    let (short, full) = (options_at(3), options_at(4));
    let program = trivial_program();
    let hasher = super::registry::build_artifacts(&program, &short).hasher;
    let artifacts =
        build_artifacts_under::<<P1Family as ProofFamily>::Stark>(&program, &short, hasher);
    let full_artifacts =
        build_artifacts_under::<<P1Family as ProofFamily>::Stark>(&program, &full, hasher);
    assert_eq!(
        artifacts.program_id, full_artifacts.program_id,
        "the grind is no part of the program"
    );
    // A child at 3 bits that some table's nonce does not carry to 4: the
    // factor is part of the hashed preimage, so a 3-bit nonce passes a 4-bit
    // check about 1/16 of the time per table, and the first salt nearly
    // always gives one.
    let (proof, salt) = (0..16u64)
        .find_map(|salt| {
            let proof =
                lfm_prove_under::<P1Family>(&program, &artifacts, &arenas_salted(salt), &short)
                    .expect("prove");
            let verifies = |opts: &ProofOptions| {
                verify_against_artifacts_under::<P1Family>(
                    &artifacts,
                    &proof.proof,
                    &proof.public_words,
                    opts,
                )
            };
            assert!(verifies(&short), "salt {salt}: the honest child at 3 bits");
            (!verifies(&full)).then_some((proof, salt))
        })
        .expect("a child whose nonces do not all meet 4 bits");
    println!("P1 NODE GRIND: salt {salt}: the host at 4 bits refuses the 3-bit child");
    let child = harvest_child_under::<P1Family>(artifacts.clone(), short.clone(), &proof)
        .unwrap_or_else(|e| panic!("{e}"));
    let words = proof.public_words.len();
    assert!(
        runs(
            &node_at(&artifacts, &short, words, P1Family::STATEMENT_TAG),
            &child
        ),
        "the node at 3 bits accepts the 3-bit child"
    );
    assert!(
        refused_by_a_check(
            &node_at(&artifacts, &full, words, P1Family::STATEMENT_TAG),
            &child
        ),
        "the node at 4 bits must refuse a child ground at 3"
    );
}

/// ★ Every cap node of a P1 child that NO query reaches is bound in the node:
/// such a node is read only by `CapCells::authenticate_at`'s cap-to-root (the
/// mux passes it into a branch no opening keeps), so moving it is refused by
/// that check alone. Over every table, every capped trace group and FRI layer:
/// the cap nodes the transcript's indices reach are computed from the
/// harvest's `iotas` (trace leaf `iota`, layer `j` leaf `iota >> (G_j + d_j)`,
/// cap node `leaf >> (D − log2(cap nodes))`), and each unreached one is moved.
/// Which nodes a run reaches follows its grind nonces, so the set is computed,
/// never assumed (the tamper test above flips one node whose reach it does not
/// control). From REV-P3B14-B's
/// `rp3b14b_every_unqueried_cap_node_of_a_p1_child_is_bound_in_the_node`.
#[test]
#[ignore = "proves a P1 child and runs its node once per unreached cap node; box tier"]
fn every_unqueried_cap_node_of_a_p1_child_is_bound_in_the_node() {
    let c = p1_child();
    let honest = harvest(&c);
    let program = node(&c, P1Family::STATEMENT_TAG);
    assert!(runs(&program, &honest), "the honest child executes");
    let mut unreached: Vec<(usize, usize)> = Vec::new();
    let mut reached_one: Option<(usize, usize)> = None;
    for (t, leg) in honest.legs.iter().enumerate() {
        let v = &leg.verify;
        let iotas = &honest.tables[t].iotas;
        // (depth, cap height, the leaf index of each query) per capped tree,
        // in the caps arena's order: the groups, then the capped layers.
        let mut capped: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        if v.sub.trace_cap > 0 {
            for _ in 0..v.sub.groups().len() {
                capped.push((v.sub.merkle_depth, v.sub.trace_cap, iotas.clone()));
            }
        }
        for j in 0..v.fri.num_committed() {
            if v.fri.layer_cap(j) > 0 {
                let shift = v.fri.layer_bit_offset(j) + v.fri.layer_fold(j) as usize;
                let leaves = iotas.iter().map(|i| i >> shift).collect();
                capped.push((v.fri.layer_depth(j), v.fri.layer_cap(j), leaves));
            }
        }
        let mut at = 0usize;
        for (depth, cap, leaves) in capped {
            let nodes = super::merkle_cap::cap_nodes(depth, cap, 4);
            let walked = depth - super::merkle_cap::mux_bits(depth, cap, 4);
            let hit: std::collections::BTreeSet<usize> =
                leaves.iter().map(|l| l >> walked).collect();
            for k in 0..nodes {
                if hit.contains(&k) {
                    reached_one.get_or_insert((t, at + k));
                } else {
                    unreached.push((t, at + k));
                }
            }
            at += nodes;
        }
        assert_eq!(
            at,
            leg.caps.len(),
            "table {t}: the caps arena is every capped tree's cap"
        );
    }
    assert!(
        !unreached.is_empty(),
        "some cap node is reached by no query"
    );
    for &(t, k) in &unreached {
        let mut bad = harvest(&c);
        flip(&mut bad.legs[t].caps[k]);
        assert!(
            refused_by_a_check(&program, &bad),
            "table {t} cap word {k}: an unreached cap node moved"
        );
    }
    let (t, k) = reached_one.expect("some cap node is reached");
    let mut bad = harvest(&c);
    flip(&mut bad.legs[t].caps[k]);
    assert!(
        refused_by_a_check(&program, &bad),
        "a reached cap node moved"
    );
    println!(
        "P1 NODE CAPS: {} unreached cap nodes moved and refused (of {} cap words)",
        unreached.len(),
        honest.legs.iter().map(|l| l.caps.len()).sum::<usize>()
    );
}

// ============ the legs' opening against the host's own trees ============

type P1Backend = super::p1_commit::P1BatchBackend<GoldilocksField>;

fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn lanes(seed: u64) -> [FE; 4] {
    core::array::from_fn(|k| FE::from(mix(seed, k as u64)))
}

/// One opening of a host Poseidon1 tree over `2^depth` leaves, checked the
/// way an LFM leg checks a trace group or a FRI layer: the path as the proof
/// carries it (`embed_cap_arity` appends the cap to the owner's and cuts every
/// other), laid out by the harvest's `path_to_cap` at `layout_at` (the index
/// the harvest believes), the cap hinted and authenticated against the root by
/// `hint_and_authenticate`, then `authenticate_leaf` at the bits of `bits_of`
/// (the transcript's index, decomposed by `BitDec`). `index` is the leaf whose
/// path is supplied; `owner` picks query 0's path or another query's; `tamper`
/// moves one arena word (arenas: 0 leaf, 1 index, 2 hints, 3 cap, 4 root).
fn leg_opening(
    depth: usize,
    c: usize,
    index: usize,
    layout_at: usize,
    bits_of: usize,
    owner: bool,
    tamper: Option<(usize, usize)>,
) -> Result<(), String> {
    let n = 1usize << depth;
    let leaves: Vec<Commitment> = (0..n)
        .map(|i| digest_to_commitment(&lanes(31_000 + depth as u64 * 257 + i as u64)))
        .collect();
    let tree = MerkleTree::<P1Backend>::build_from_hashed_leaves(leaves.clone()).expect("a tree");
    let full = tree
        .get_proof_by_pos(index)
        .expect("a leaf")
        .merkle_path
        .clone();
    let cap = if c == 0 {
        Vec::new()
    } else {
        tree.cap(c).expect("the cap")
    };
    let (mut owner_path, mut other_path) = (full.clone(), full);
    if c > 0 {
        crypto::merkle_tree::cap::embed_cap_arity(
            &mut [&mut owner_path, &mut other_path],
            depth,
            &cap,
            4,
        )
        .expect("the proof's paths");
    }
    let path = if owner { &owner_path } else { &other_path };
    let (hints, split) = super::merkle_cap::path_to_cap(path, depth, c, 4, owner, Some(layout_at))
        .map_err(|e| format!("layout: {e}"))?;
    if owner {
        assert_eq!(
            split.unwrap_or_default(),
            cap,
            "the cap rides on the owner path"
        );
    } else {
        assert!(split.is_none(), "only the owner carries the cap");
    }
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
        arenas[a][w][1] += FE::from(1u64);
    }
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
    let lens: Vec<u32> = arenas.iter().map(|a| a.len().max(1) as u32).collect();
    let ids: Vec<_> = lens.iter().map(|&l| b.declare_arena(l)).collect();
    let leaf = WrapDigest::from_cell(b.hint_word(ids[0], 0));
    let idx = b.hint_felt(ids[1], 0);
    let bits = b.bit_dec(idx, depth);
    let h: Vec<WrapDigest> = (0..arenas[2].len() as u32)
        .map(|i| WrapDigest::from_cell(b.hint_word(ids[2], i)))
        .collect();
    let root = b.hint_word(ids[4], 0);
    let root_lanes = [b.unpack(root)];
    let capped = (c > 0).then(|| {
        let (cells, next) =
            super::merkle_cap::hint_and_authenticate(&mut b, ids[3], 0, c, depth, &root_lanes);
        assert_eq!(next as usize, cap.len(), "the caps arena is read exactly");
        cells
    });
    super::sub_proof::authenticate_leaf(&mut b, leaf, &root_lanes, capped.as_ref(), &bits, &h);
    // An arena the check reads nothing from (no hints under a leaf-level cap,
    // no cap uncapped) is declared one word long.
    for a in arenas.iter_mut().filter(|a| a.is_empty()) {
        a.push(base_word(FE::from(0u64)));
    }
    let program = compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("{e:?}"))?;
    execute_serial(&program, &arenas, &HasherKind::Poseidon1W16)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

/// ★ The LFM legs' production opening (hint-order `walk4` to the cap, the
/// cap's mux and its root) against the host's own Poseidon1 trees,
/// EXHAUSTIVELY at small depths: at every depth 1–7 (both parities), every cap
/// height up to the tree's levels and EVERY leaf index, the harvest's layout
/// of query 0's path (cap split off) and of any other query's path executes;
/// a leaf's path walked at another index's bits is refused, whether the hints
/// are laid out for the leaf's index or for the other one (the prover's choice
/// of hint order cannot move the leaf: every placement is a `Select` on a
/// transcript bit); and every word the check reads — the leaf, each hint, each
/// cap node (queried or not), the root — moved refuses it. Then the production
/// cap (C2) at depths 8–12 over a spread of indices. From REV-P3B14-B's
/// `rp3b14b_the_hint_order_walk_and_cap_bind_every_index`.
#[test]
fn the_hint_order_walk_and_cap_bind_every_index() {
    let mut refused_moves = 0usize;
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
                for owner in [true, false] {
                    leg_opening(depth, c, index, index, index, owner, None).unwrap_or_else(|e| {
                        panic!("depth {depth} cap {c} index {index} owner {owner}: {e}")
                    });
                }
                let others: Vec<usize> = if depth <= 4 {
                    (0..n).filter(|&o| o != index).collect()
                } else {
                    [
                        index ^ 1,
                        index ^ 2,
                        index ^ 3,
                        index ^ (n / 2),
                        n - 1 - index,
                    ]
                    .into_iter()
                    .filter(|&o| o != index)
                    .collect()
                };
                for other in others {
                    assert!(
                        leg_opening(depth, c, index, index, other, true, None).is_err(),
                        "depth {depth} cap {c}: leaf {index}'s hints walked at {other}"
                    );
                    assert!(
                        leg_opening(depth, c, index, other, other, false, None).is_err(),
                        "depth {depth} cap {c}: leaf {index}'s path laid out and walked at {other}"
                    );
                }
                if depth <= 5 || index % 9 == 0 {
                    let mut words = vec![(0usize, 0usize), (4, 0)];
                    words.extend((0..hints).map(|w| (2, w)));
                    words.extend((0..nodes).map(|w| (3, w)));
                    for (a, w) in words {
                        assert!(
                            leg_opening(depth, c, index, index, index, true, Some((a, w))).is_err(),
                            "depth {depth} cap {c} index {index}: arena {a} word {w} moved"
                        );
                        refused_moves += 1;
                    }
                }
            }
        }
    }
    // The production cap height on taller trees.
    for depth in 8usize..=12 {
        let n = 1usize << depth;
        for index in [0, 1, 2, 3, n / 3, n / 2 + 5, n - 4, n - 1] {
            for owner in [true, false] {
                leg_opening(depth, 2, index, index, index, owner, None)
                    .unwrap_or_else(|e| panic!("depth {depth} C2 index {index}: {e}"));
            }
            for other in [index ^ 1, index ^ 4, index ^ (n / 2)] {
                assert!(leg_opening(depth, 2, index, index, other, false, None).is_err());
                assert!(leg_opening(depth, 2, index, other, other, false, None).is_err());
            }
        }
        let nodes = super::merkle_cap::cap_nodes(depth, 2, 4);
        for w in 0..nodes {
            assert!(
                leg_opening(depth, 2, 5, 5, 5, true, Some((3, w))).is_err(),
                "depth {depth} C2: cap node {w} moved"
            );
        }
    }
    assert!(
        refused_moves > 1000,
        "the moves were probed: {refused_moves}"
    );
}
