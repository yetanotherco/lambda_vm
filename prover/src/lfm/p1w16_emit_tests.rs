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
            run(b, &[leaves.clone()]),
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
