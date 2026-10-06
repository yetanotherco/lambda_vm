//! ZisK's Poseidon1 constructions as emitted programs — what a level-0 program
//! runs to verify a Poseidon1 base proof (P3a).
//!
//! Every construction here mirrors a host function the base prover and its
//! verifier already run, and each is pinned to it by an execution test
//! (`p1w16_emit_tests`):
//!
//! | here | host |
//! |---|---|
//! | [`leaf_hash`] | `poseidon1_stark::linear_hash` |
//! | [`node4`], [`walk4`], [`tree_root4`] | `poseidon1_w16::compress4`, `Merkle4` (zero-digest padding) |
//! | [`P1SpongeVar`] | `poseidon1_stark::Transcript` under `p1_commit::P1Transcript`'s encodings |
//! | [`w8_permute`] | `poseidon1_w8::permute` (the grinding permutation) |
//!
//! The width-16 permutation runs on the socket (`Instr::Hash16`); the width-8
//! one, needed only by the grinding check, is emulated with ALU instructions
//! (two per check, ≈ 2,300 instructions each).
//!
//! # A 4-ary level's hints
//!
//! ZisK's path lists a level's three siblings in child order. The walk takes them
//! in a different order, which the host arranges when it fills the arena
//! ([`hint_order`]): the current node's PARTNER (the other child of its pair),
//! then the OTHER PAIR in child order. Then the bit `b0` orders the current pair
//! with one `Select` per digest cell and `b1` orders the two pairs with two, all
//! outputs used. Hint order is the prover's to choose; the root equality is what
//! binds every sibling.
//!
//! An odd-depth tree's top level holds two real children and two zero digests
//! (`Merkle4`'s padding). There the index's phantom bit is zero, the walk takes
//! ONE hint (the partner), and the two padding children are constant zero cells,
//! not hints, so they cannot be anything else.

use crate::tables::types::FE;
use crypto::hash::poseidon1_w8 as w8;

use super::builder::{Bit, Cell, Felt, LfmBuilder};
use super::edsl::WrapDigest;

/// Felts of rate in one width-16 block.
pub const RATE_FELTS: usize = 12;

fn zero_cell(b: &mut LfmBuilder) -> Cell {
    b.digest_const([FE::zero(); 4]).as_cell()
}

/// A lane the sponge or a leaf packs: a program constant or a machine felt.
#[derive(Clone, Copy, Debug)]
pub enum Lane {
    Const(FE),
    Var(Felt),
}

/// Four lanes as one word: an interned constant when every lane is one, else a
/// `Pack` with the constant lanes interned.
fn pack(b: &mut LfmBuilder, lanes: [Lane; 4]) -> Cell {
    if let [
        Lane::Const(a),
        Lane::Const(c),
        Lane::Const(d),
        Lane::Const(e),
    ] = lanes
    {
        return b.digest_const([a, c, d, e]).as_cell();
    }
    let felts = lanes.map(|l| match l {
        Lane::Const(v) => b.felt_const(v),
        Lane::Var(f) => f,
    });
    b.pack_word(felts)
}

/// Up to twelve lanes as the three rate words, zero-filled.
fn rate_cells(b: &mut LfmBuilder, lanes: &[Lane]) -> [Cell; 3] {
    debug_assert!(lanes.len() <= RATE_FELTS);
    core::array::from_fn(|k| {
        let word: [Lane; 4] = core::array::from_fn(|j| {
            lanes
                .get(4 * k + j)
                .copied()
                .unwrap_or(Lane::Const(FE::zero()))
        });
        pack(b, word)
    })
}

/// ZisK's leaf hash over `felts`: blocks of twelve (the last zero-filled), the
/// capacity the zero cell for the first block and the previous output's cell 0
/// for every later one; the digest is the last output's cell 0. No felts hash
/// to the zero digest, as on the host.
pub fn leaf_hash(b: &mut LfmBuilder, felts: &[Felt]) -> WrapDigest {
    let zero = zero_cell(b);
    let mut carry = zero;
    for block in felts.chunks(RATE_FELTS) {
        let lanes: Vec<Lane> = block.iter().map(|f| Lane::Var(*f)).collect();
        let [c0, c1, c2] = rate_cells(b, &lanes);
        carry = b.hash16([c0, c1, c2, carry])[0];
    }
    WrapDigest::from_cell(carry)
}

/// One 4-ary node: the four children, permuted, truncated to cell 0.
pub fn node4(b: &mut LfmBuilder, children: [Cell; 4]) -> Cell {
    b.hash16(children)[0]
}

/// Hints a path of `nbits` index bits takes: three per full 4-ary level, one at
/// an odd-depth top level.
pub const fn path_hints(nbits: usize) -> usize {
    3 * (nbits / 2) + nbits % 2
}

/// Walk one 4-ary Merkle path: `bits` the leaf index low first, `hints` in
/// [`hint_order`] (partner, then the other pair, per level; one at an odd top).
pub fn walk4(
    b: &mut LfmBuilder,
    leaf: WrapDigest,
    bits: &[Bit],
    hints: &[WrapDigest],
) -> WrapDigest {
    assert_eq!(
        hints.len(),
        path_hints(bits.len()),
        "three hints per 4-ary level, one at an odd top"
    );
    let mut cur = leaf.cells()[0];
    let mut h = hints.iter().map(|d| d.cells()[0]);
    for pair in bits.chunks(2) {
        let partner = h.next().expect("counted above");
        let (l, r) = b.select(pair[0], cur, partner);
        cur = match pair {
            [_, b1] => {
                let u0 = h.next().expect("counted above");
                let u1 = h.next().expect("counted above");
                let (c0, c2) = b.select(*b1, l, u0);
                let (c1, c3) = b.select(*b1, r, u1);
                node4(b, [c0, c1, c2, c3])
            }
            _ => {
                // The odd top: two real children and two zero digests.
                let z = zero_cell(b);
                node4(b, [l, r, z, z])
            }
        };
    }
    WrapDigest::from_cell(cur)
}

/// A 4-ary tree's root over `leaves`, each level padded with zero digests to a
/// multiple of four (`Merkle4`). One leaf is its own root.
pub fn tree_root4(b: &mut LfmBuilder, leaves: &[WrapDigest]) -> WrapDigest {
    assert!(!leaves.is_empty(), "a tree has at least one leaf");
    let mut level: Vec<Cell> = leaves.iter().map(|d| d.cells()[0]).collect();
    while level.len() > 1 {
        let z = zero_cell(b);
        level.resize(level.len().div_ceil(4) * 4, z);
        level = level
            .chunks_exact(4)
            .map(|q| node4(b, [q[0], q[1], q[2], q[3]]))
            .collect();
    }
    WrapDigest::from_cell(level[0])
}

/// Host side: ZisK's path for leaf `index` (per level, its three siblings in
/// child order) as the walk's hints, leaf level first. At an odd-depth top
/// (`depth_bits` odd) only the partner is kept: the other two are the zero
/// padding the walk supplies itself.
pub fn hint_order<D: Copy>(index: usize, depth_bits: usize, path: &[[D; 3]]) -> Vec<D> {
    assert_eq!(
        path.len(),
        depth_bits.div_ceil(2),
        "one sibling triple per 4-ary level"
    );
    let mut out = Vec::with_capacity(path_hints(depth_bits));
    let mut i = index;
    for (level, sib) in path.iter().enumerate() {
        let p = i % 4;
        // The children other than `p`, in child order, as ZisK lists them.
        let child = |c: usize| -> D {
            debug_assert_ne!(c, p);
            sib[if c < p { c } else { c - 1 }]
        };
        out.push(child(p ^ 1));
        let odd_top = depth_bits % 2 == 1 && level + 1 == path.len();
        if !odd_top {
            let other = 2 * (1 - p / 2);
            out.push(child(other));
            out.push(child(other + 1));
        }
        i /= 4;
    }
    out
}

/// ZisK's transcript (`poseidon1_stark::Transcript`) as an emitted sponge.
///
/// The state is the last permutation's four output cells (zero before the
/// first). Absorbed lanes collect in a buffer of up to twelve; a full buffer,
/// or a squeeze after an absorb, permutes `buffer (zero-filled) ‖ state[0]`. A
/// squeeze reads the output's lanes 0, 1, …, 15 in order and permutes again
/// once all sixteen are read. As on the host, every absorb sets the unread
/// count to zero BEFORE a full buffer flushes, so a squeeze right after exactly
/// twelve absorbs reads the fresh output.
#[derive(Clone, Debug)]
pub struct P1SpongeVar {
    state: [Cell; 4],
    lanes: [Option<[Felt; 4]>; 4],
    pending: Vec<Lane>,
    unread: usize,
}

impl P1SpongeVar {
    /// The empty transcript.
    pub fn new(b: &mut LfmBuilder) -> Self {
        let z = zero_cell(b);
        Self {
            state: [z; 4],
            lanes: [None; 4],
            pending: Vec::with_capacity(RATE_FELTS),
            unread: 0,
        }
    }

    /// Absorb one lane.
    pub fn put(&mut self, b: &mut LfmBuilder, x: Lane) {
        self.pending.push(x);
        self.unread = 0;
        if self.pending.len() == RATE_FELTS {
            self.update(b);
        }
    }

    /// Absorb a machine felt.
    pub fn put_felt(&mut self, b: &mut LfmBuilder, f: Felt) {
        self.put(b, Lane::Var(f));
    }

    /// Absorb a program constant.
    pub fn put_const(&mut self, b: &mut LfmBuilder, v: FE) {
        self.put(b, Lane::Const(v));
    }

    fn permute_pending(&self, b: &mut LfmBuilder) -> [Cell; 4] {
        let [c0, c1, c2] = rate_cells(b, &self.pending);
        b.hash16([c0, c1, c2, self.state[0]])
    }

    fn update(&mut self, b: &mut LfmBuilder) {
        self.state = self.permute_pending(b);
        self.lanes = [None; 4];
        self.pending.clear();
        self.unread = 16;
    }

    /// Squeeze one felt.
    pub fn squeeze(&mut self, b: &mut LfmBuilder) -> Felt {
        if self.unread == 0 {
            self.update(b);
        }
        let idx = 16 - self.unread;
        self.unread -= 1;
        let word = idx / 4;
        let state = self.state[word];
        let lanes = *self.lanes[word].get_or_insert_with(|| b.unpack(state));
        lanes[idx % 4]
    }

    /// The transcript's state digest without advancing it — `P1Transcript::state`,
    /// which flushes a COPY: a non-empty buffer costs one permutation whose
    /// output is not adopted.
    pub fn digest(&self, b: &mut LfmBuilder) -> Cell {
        if self.pending.is_empty() {
            self.state[0]
        } else {
            self.permute_pending(b)[0]
        }
    }
}

/// `x⁷` in four multiplications.
fn sbox7(b: &mut LfmBuilder, x: Felt) -> Felt {
    let x2 = b.mul(x, x);
    let x3 = b.mul(x2, x);
    let x6 = b.mul(x3, x3);
    b.mul(x6, x)
}

/// The width-8 permutation (`poseidon1_w8::permute`) in ALU instructions:
/// each round's MDS row is a `MulAdd` chain seeded with the NEXT round's
/// constant, so the round-constant additions after round 0 cost nothing. 8 +
/// 8·(32 + 64) + 22·(4 + 64) = 2,272 instructions.
pub fn w8_permute(b: &mut LfmBuilder, state: [Felt; 8]) -> [Felt; 8] {
    use w8::constants::{MDS_CIRC_ROW, ROUND_CONSTANTS};
    const T: usize = w8::STATE_FELTS;
    let half = w8::HALF_FULL_ROUNDS;
    let partial = half..half + w8::PARTIAL_ROUNDS;
    let m = |o: usize, i: usize| FE::from(MDS_CIRC_ROW[(i + T - o) % T]);

    let mut a: [Felt; T] = core::array::from_fn(|i| {
        let c = b.felt_const(FE::from(ROUND_CONSTANTS[0][i]));
        b.add(state[i], c)
    });
    for r in 0..w8::NUM_ROUNDS {
        let mut f = a;
        let lanes = if partial.contains(&r) { 1 } else { T };
        for x in f.iter_mut().take(lanes) {
            *x = sbox7(b, *x);
        }
        a = core::array::from_fn(|o| {
            let mut acc = if r + 1 < w8::NUM_ROUNDS {
                b.felt_const(FE::from(ROUND_CONSTANTS[r + 1][o]))
            } else {
                let k = b.felt_const(m(o, 0));
                b.mul(k, f[0])
            };
            let first = usize::from(r + 1 == w8::NUM_ROUNDS);
            for (i, fi) in f.iter().enumerate().skip(first) {
                let k = b.felt_const(m(o, i));
                acc = b.mul_add(k, *fi, acc);
            }
            acc
        });
    }
    a
}
