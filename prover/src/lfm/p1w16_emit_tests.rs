//! The emitted Poseidon1 constructions against the host's: each program is
//! executed (the width-16 socket on the executor's host permutation) and its
//! published words compared with `poseidon1_stark` / `poseidon1_w8`.

use crate::tables::types::FE;
use crypto::hash::poseidon1_stark::{self as zisk, Merkle4};
use crypto::hash::{poseidon1_w8, poseidon1_w16};

use super::builder::{Felt, LfmBuilder};
use super::compiler::compile;
use super::edsl::WrapDigest;
use super::executor::execute_serial;
use super::hash::HasherKind;
use super::p1w16_emit::*;
use super::word::{LfmWord, base_word};

fn felt(seed: u64, i: usize) -> FE {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i as u64 + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    FE::from(z ^ (z >> 31))
}

/// Run `b`'s program over `arenas`; the published words, flattened.
fn run(b: LfmBuilder, arenas: &[Vec<LfmWord>]) -> Vec<FE> {
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    let exec = execute_serial(&program, arenas, &HasherKind::Poseidon1W16).expect("executes");
    exec.public_words.iter().flat_map(|(_, w)| *w).collect()
}

fn hinted_felts(b: &mut LfmBuilder, n: usize) -> Vec<Felt> {
    let a = b.declare_arena(n as u32);
    (0..n as u32).map(|i| b.hint_felt(a, i)).collect()
}

#[test]
fn the_leaf_hash_is_zisks_linear_hash() {
    for n in [0usize, 1, 4, 11, 12, 13, 24, 25, 36, 41] {
        let values: Vec<FE> = (0..n).map(|i| felt(n as u64, i)).collect();
        let mut b = LfmBuilder::new();
        let felts = hinted_felts(&mut b, n);
        let d = leaf_hash(&mut b, &felts);
        b.public(d.cells()[0]);
        let got = run(b, &[values.iter().copied().map(base_word).collect()]);
        assert_eq!(got, zisk::linear_hash(&values).to_vec(), "{n} felts");
    }
}

fn digest(seed: u64, i: usize) -> [FE; 4] {
    core::array::from_fn(|k| felt(seed, 4 * i + k))
}

fn sibling_triples(path: &[[FE; zisk::SIBLING_FELTS]]) -> Vec<[[FE; 4]; 3]> {
    path.iter()
        .map(|s| core::array::from_fn(|j| core::array::from_fn(|k| s[4 * j + k])))
        .collect()
}

/// Every leaf of trees of even and odd depth (and the one-leaf tree): the walk
/// over the hinted path reaches `Merkle4`'s root, and `tree_root4` builds it.
#[test]
fn the_walk_and_the_tree_reach_merkle4s_root() {
    for depth in 0usize..=5 {
        let n = 1usize << depth;
        let leaves: Vec<[FE; 4]> = (0..n).map(|i| digest(depth as u64 + 7, i)).collect();
        let tree = Merkle4::new(&leaves).expect("non-empty");
        let root = tree.root();
        for index in [0, n / 2, n - 1] {
            let hints = hint_order(
                index,
                depth,
                &sibling_triples(&tree.path(index).expect("leaf")),
            );
            assert_eq!(hints.len(), path_hints(depth));
            let mut b = LfmBuilder::new();
            let a_leaf = b.declare_arena(1);
            let a_idx = b.declare_arena(1);
            let a_hint = b.declare_arena(hints.len() as u32);
            let leaf = WrapDigest::from_cell(b.hint_word(a_leaf, 0));
            let idx = b.hint_felt(a_idx, 0);
            let bits = b.bit_dec(idx, depth);
            let h: Vec<WrapDigest> = (0..hints.len() as u32)
                .map(|i| WrapDigest::from_cell(b.hint_word(a_hint, i)))
                .collect();
            let walked = walk4(&mut b, leaf, &bits, &h);
            b.public(walked.cells()[0]);
            let got = run(
                b,
                &[
                    vec![leaves[index]],
                    vec![base_word(FE::from(index as u64))],
                    hints.clone(),
                ],
            );
            assert_eq!(got, root.to_vec(), "depth {depth} index {index}");
        }

        let mut b = LfmBuilder::new();
        let a = b.declare_arena(n as u32);
        let ds: Vec<WrapDigest> = (0..n as u32)
            .map(|i| WrapDigest::from_cell(b.hint_word(a, i)))
            .collect();
        let r = tree_root4(&mut b, &ds);
        b.public(r.cells()[0]);
        assert_eq!(
            run(b, std::slice::from_ref(&leaves)),
            root.to_vec(),
            "tree, depth {depth}"
        );
    }
}

/// A wrong sibling moves the walked root: the hints are bound, including the
/// partner at an odd top.
#[test]
fn a_tampered_hint_moves_the_walked_root() {
    for depth in [3usize, 4] {
        let n = 1usize << depth;
        let leaves: Vec<[FE; 4]> = (0..n).map(|i| digest(99, i)).collect();
        let tree = Merkle4::new(&leaves).expect("non-empty");
        let index = n - 2;
        let hints = hint_order(
            index,
            depth,
            &sibling_triples(&tree.path(index).expect("leaf")),
        );
        for t in 0..hints.len() {
            let mut bad = hints.clone();
            bad[t][1] += FE::one();
            let mut b = LfmBuilder::new();
            let a_leaf = b.declare_arena(1);
            let a_idx = b.declare_arena(1);
            let a_hint = b.declare_arena(bad.len() as u32);
            let leaf = WrapDigest::from_cell(b.hint_word(a_leaf, 0));
            let idx = b.hint_felt(a_idx, 0);
            let bits = b.bit_dec(idx, depth);
            let h: Vec<WrapDigest> = (0..bad.len() as u32)
                .map(|i| WrapDigest::from_cell(b.hint_word(a_hint, i)))
                .collect();
            let walked = walk4(&mut b, leaf, &bits, &h);
            b.public(walked.cells()[0]);
            let got = run(
                b,
                &[
                    vec![leaves[index]],
                    vec![base_word(FE::from(index as u64))],
                    bad,
                ],
            );
            assert_ne!(got, tree.root().to_vec(), "depth {depth}, hint {t}");
        }
    }
}

/// One step of a transcript script, run on both sides.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// Absorb a machine felt.
    Felt(u64),
    /// Absorb a program constant.
    Const(u64),
    /// Squeeze one felt (published).
    Squeeze,
    /// The state digest, without advancing (published).
    State,
}

fn run_script(ops: &[Op]) {
    // The host.
    let mut host = zisk::Transcript::new();
    let mut want: Vec<FE> = Vec::new();
    for op in ops {
        match *op {
            Op::Felt(v) | Op::Const(v) => host.put1(FE::from(v)),
            Op::Squeeze => want.push(host.squeeze1()),
            Op::State => want.extend_from_slice(&host.clone().state()[..4]),
        }
    }
    // The machine.
    let mut b = LfmBuilder::new();
    let vars: Vec<u64> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Felt(v) => Some(*v),
            _ => None,
        })
        .collect();
    let a = b.declare_arena(vars.len() as u32);
    let mut s = P1SpongeVar::new(&mut b);
    let mut next = 0u32;
    let zero = b.felt_const(FE::zero());
    for op in ops {
        match *op {
            Op::Felt(_) => {
                let f = b.hint_felt(a, next);
                next += 1;
                s.put_felt(&mut b, f);
            }
            Op::Const(v) => s.put_const(&mut b, FE::from(v)),
            Op::Squeeze => {
                let f = s.squeeze(&mut b);
                let w = b.pack_word([f, zero, zero, zero]);
                b.public(w);
            }
            Op::State => {
                let c = s.digest(&mut b);
                b.public(c);
            }
        }
    }
    let got = run(b, &[vars.iter().map(|v| base_word(FE::from(*v))).collect()]);
    // Squeezes publish one lane and three zeros; states publish four lanes.
    let mut flat = Vec::new();
    let mut it = got.chunks(4);
    for op in ops {
        match op {
            Op::Squeeze => flat.push(it.next().expect("a word")[0]),
            Op::State => flat.extend_from_slice(it.next().expect("a word")),
            _ => {}
        }
    }
    assert_eq!(flat, want, "{ops:?}");
}

#[test]
fn the_sponge_is_zisks_transcript() {
    use Op::*;
    // A squeeze before any absorb; then squeezes across the sixteen-lane
    // boundary; a state with an empty and a non-empty buffer.
    run_script(&[Squeeze, Squeeze, State]);
    run_script(&(0..18).map(|_| Squeeze).collect::<Vec<_>>());
    run_script(&[
        Felt(1),
        State,
        Squeeze,
        Felt(2),
        Const(3),
        Squeeze,
        State,
        Squeeze,
    ]);
    // Exactly twelve absorbs flush the buffer and the squeeze reads that output.
    let mut ops: Vec<Op> = (0..12).map(|i| Felt(100 + i)).collect();
    ops.extend([Squeeze, Squeeze]);
    run_script(&ops);
    // Thirteen, then a mix of constants and machine felts over two buffers.
    let mut ops: Vec<Op> = (0..13)
        .map(|i| if i % 3 == 0 { Const(i) } else { Felt(i) })
        .collect();
    ops.extend([State, Squeeze, Squeeze, Squeeze]);
    ops.extend((0..25).map(|i| Felt(1_000 + i)));
    ops.extend([Squeeze, State]);
    run_script(&ops);
}

#[test]
fn the_emulated_width8_permutation_is_poseidon1_w8() {
    for seed in [0u64, 1, 0xFFFF] {
        let input: [FE; 8] = core::array::from_fn(|i| felt(seed, i));
        let mut b = LfmBuilder::new();
        let s: [Felt; 8] = {
            let v = hinted_felts(&mut b, 8);
            core::array::from_fn(|i| v[i])
        };
        let out = w8_permute(&mut b, s);
        for q in out.chunks(4) {
            let w = b.pack_word([q[0], q[1], q[2], q[3]]);
            b.public(w);
        }
        let got = run(b, &[input.iter().copied().map(base_word).collect()]);
        assert_eq!(got, poseidon1_w8::permute(input).to_vec(), "seed {seed}");
    }
    // The socket's width-16 permutation is the same instance's, for the record.
    let x: [FE; 16] = core::array::from_fn(|i| FE::from(i as u64));
    assert_eq!(
        poseidon1_w16::permute(x)[0],
        FE::from(9_350_316_517_402_464_675u64)
    );
}

/// The transcript replay's Poseidon1 arm against `P1Transcript`: every append
/// kind a sub-proof replays (constant bytes, a root, a felt, an extension
/// element, machine bytes) and every draw (an extension element, index bits,
/// a felt, the state).
#[test]
fn the_replay_arm_is_p1_transcript() {
    use super::edsl::WrapHash;
    use super::p1_commit::P1Transcript;
    use super::transcript_replay::TranscriptReplay;
    use crate::tables::types::FEE;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let seed = b"p3 replay seed";
    let root: [FE; 4] = digest(5, 0);
    let felt_v = felt(6, 0);
    let ext_v: [FE; 3] = core::array::from_fn(|i| felt(7, i));
    let bytes: [u8; 8] = 0x0123_4567_89ab_cdefu64.to_le_bytes();
    let nbits = 13usize;

    // The host.
    let mut host = P1Transcript::with_seed(seed);
    host.append_bytes(&super::algebraic_commit::digest_to_commitment(&root));
    host.append_bytes(&felt_v.canonical().to_be_bytes());
    host.append_field_element(&FEE::new(ext_v));
    host.append_bytes(&bytes);
    let e = host.sample_field_element();
    let idx = host.sample_u64(1 << nbits);
    host.append_bytes(&[1, 2, 3]);
    let state = host.state();
    let f = host.sample_field_element();

    // The machine.
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
    let a_root = b.declare_arena(1);
    let a_felts = b.declare_arena(6);
    let r = b.hint_word(a_root, 0);
    let fv = b.hint_felt(a_felts, 0);
    let ev: [Felt; 3] = core::array::from_fn(|i| b.hint_felt(a_felts, 1 + i as u32));
    let halves = [b.hint_felt(a_felts, 4), b.hint_felt(a_felts, 5)];
    let mut t = TranscriptReplay::new(seed);
    t.append_root_cells(&mut b, &[r]);
    t.append_felt(&mut b, fv);
    t.append_ext(&mut b, ev);
    t.append_halves(&halves);
    let e_m = t.sample_ext(&mut b);
    b.public(e_m.as_cell());
    let bits = t.sample_u64_pow2(&mut b, nbits);
    let idx_m = super::edsl::bits_to_felt(&mut b, &bits);
    let zero = b.felt_const(FE::zero());
    let w = b.pack_word([idx_m, zero, zero, zero]);
    b.public(w);
    t.append_const_bytes(&[1, 2, 3]);
    let s = t.state(&mut b);
    b.public(s.cells()[0]);
    let f_m = t.sample_ext(&mut b);
    b.public(f_m.as_cell());

    let half = |k: usize| {
        FE::from(u64::from(u32::from_le_bytes(
            bytes[4 * k..4 * k + 4].try_into().unwrap(),
        )))
    };
    let got = run(
        b,
        &[
            vec![root],
            [felt_v, ext_v[0], ext_v[1], ext_v[2], half(0), half(1)]
                .map(base_word)
                .to_vec(),
        ],
    );
    let ext3 = |x: FEE| -> Vec<FE> {
        let v = x.value();
        vec![v[0], v[1], v[2], FE::zero()]
    };
    let mut want = ext3(e);
    want.extend([FE::from(idx), FE::zero(), FE::zero(), FE::zero()]);
    want.extend(super::algebraic_commit::commitment_to_digest(&state));
    want.extend(ext3(f));
    assert_eq!(got, want);
}

/// The grinding check accepts the host's nonce and refuses one that misses.
#[test]
fn the_grinding_check_is_p1s() {
    use super::edsl::WrapHash;
    use super::p1_commit::P1GrindDigest;

    let lanes: [FE; 4] = digest(11, 0);
    let seed = super::algebraic_commit::digest_to_commitment(&lanes);
    let factor = 6u8;
    let valid = |n: u64| crypto::grinding::is_valid_nonce::<P1GrindDigest>(&seed, n, factor);
    let good = (0u64..).find(|&n| valid(n)).expect("a nonce");
    let bad = (0u64..).find(|&n| !valid(n)).expect("a miss");

    let check = |nonce: u64| {
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
        let a_seed = b.declare_arena(1);
        let a_nonce = b.declare_arena(1);
        let s = WrapDigest::from_cell(b.hint_word(a_seed, 0));
        let n = b.hint_felt(a_nonce, 0);
        super::epoch::emit_grinding_check(&mut b, s, n, factor);
        let program = compile(b.finish());
        execute_serial(
            &program,
            &[vec![lanes], vec![base_word(FE::from(nonce))]],
            &HasherKind::Poseidon1W16,
        )
        .map(|_| ())
    };
    assert!(check(good).is_ok(), "nonce {good}");
    assert!(check(bad).is_err(), "nonce {bad}");
}

/// Run `b`'s program over `arenas`, `Err` when it does not execute.
fn try_run(b: LfmBuilder, arenas: &[Vec<LfmWord>]) -> Result<(), String> {
    let program = compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("{e:?}"))?;
    execute_serial(&program, arenas, &HasherKind::Poseidon1W16)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

/// One opening of a host Poseidon1 tree checked in-guest at cap height `c`:
/// the owner path as the proof carries it (the path to the cap, then the cap,
/// `embed_cap_arity`) laid out by the harvest (`path_to_cap`), the cap
/// authenticated against the root and the opening against the cap — or, at
/// `c = 0`, walked to the root. `tamper` moves one arena word first.
fn cap_opening(
    depth: usize,
    c: usize,
    index: usize,
    tamper: Option<(usize, usize)>,
) -> Result<(), String> {
    use super::algebraic_commit::{commitment_to_digest, digest_to_commitment};
    use super::edsl::WrapHash;
    use super::merkle_cap::CapCells;
    use crypto::merkle_tree::merkle::MerkleTree;
    use stark::config::Commitment;
    type B = super::p1_commit::P1BatchBackend<crate::tables::types::GoldilocksField>;

    let n = 1usize << depth;
    let leaves: Vec<Commitment> = (0..n)
        .map(|i| digest_to_commitment(&digest(depth as u64 + 31, i)))
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
        super::harvest::path_to_cap(&path, depth, c, 4, true, Some(index)).expect("laid out");
    assert_eq!(split.unwrap_or_default(), cap, "the cap rides on the owner path");
    assert_eq!(
        hints.len(),
        super::merkle_cap::path_hints(depth, c, 4),
        "depth {depth} cap {c}"
    );

    let word = |c: &Commitment| -> LfmWord { commitment_to_digest(c) };
    let mut arenas: Vec<Vec<LfmWord>> = vec![
        vec![word(&leaves[index])],
        vec![base_word(FE::from(index as u64))],
        hints.iter().map(word).collect(),
        cap.iter().map(word).collect(),
        vec![word(&tree.root)],
    ];
    if let Some((a, w)) = tamper {
        arenas[a][w][2] += FE::one();
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
        let walked = super::edsl::wrap_merkle_walk(&mut b, leaf, &bits, &h);
        super::edsl::assert_digest_eq_lanes(&mut b, walked, &root_lanes);
    } else {
        let nodes: Vec<WrapDigest> = (0..lens[3])
            .map(|i| WrapDigest::from_cell(b.hint_word(ids[3], i)))
            .collect();
        let capped = CapCells::authenticate_at(&mut b, &nodes, c, &root_lanes);
        assert_eq!(capped.height(), c);
        capped.verify_path(&mut b, leaf, &bits, &h);
    }
    try_run(b, &arenas)
}

/// ★ Arity-4 caps against the host's own trees: at every depth (both
/// parities) and every cap height up to the tree's levels, an opening checked
/// against the host's cap (`MerkleTree::cap`, `4^c` or `2·4^(c−1)` nodes) over
/// the path the proof carries executes; uncapped, the walk reaches the root
/// (the odd top's padding supplied by the walk).
#[test]
fn an_arity4_cap_checks_the_hosts_openings() {
    for depth in 1usize..=7 {
        let n = 1usize << depth;
        let levels = crypto::merkle_tree::cap::tree_levels(depth, 4);
        for c in 0..=levels {
            for index in [0, n / 2, n - 1, (5 * n) / 7] {
                cap_opening(depth, c, index, None).unwrap_or_else(|e| {
                    panic!("depth {depth} cap {c} index {index}: {e}");
                });
            }
        }
    }
}

/// Every word the check reads is bound: a moved hint, cap node or root makes
/// the program unexecutable, at an even and an odd depth, capped and not.
#[test]
fn an_arity4_cap_refuses_a_moved_word() {
    for (depth, c) in [(6usize, 2usize), (7, 1), (7, 0), (5, 3)] {
        let index = (1usize << depth) - 3;
        let hints = super::merkle_cap::path_hints(depth, c, 4);
        let nodes = super::merkle_cap::cap_nodes(depth, c, 4);
        let mut words = vec![(0usize, 0usize), (4, 0)];
        words.extend((0..hints).map(|w| (2, w)));
        if c > 0 {
            words.extend((0..nodes).map(|w| (3, w)));
        }
        for (a, w) in words {
            assert!(
                cap_opening(depth, c, index, Some((a, w))).is_err(),
                "depth {depth} cap {c}: arena {a} word {w} moved and the opening executed"
            );
        }
    }
}
