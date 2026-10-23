//! ZisK's Poseidon1 constructions as emitted programs — what a level-0 program
//! runs to verify a Poseidon1 base proof (P3a).
//!
//! Every construction here mirrors a host function the base prover and its
//! verifier already run, and each is pinned to it by an execution test
//! (`p1w16_emit_tests`):
//!
//! | here | host |
//! |---|---|
//! | [`leaf_hash`] | `poseidon1_stark::linear_hash` (ZisK's chain with the width tag) |
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
//!
//! [`walk4_child`] takes the siblings in ZisK's child order instead, as the
//! proof carries them, and spends two more `Select`s a full level placing
//! them: its arena needs no leaf index (the WHIR leaf's, whose arena is filled
//! from the proof alone).

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

/// The P1 leaf hash over `felts` (`poseidon1_stark::linear_hash`): blocks of
/// twelve (the last zero-filled), the capacity the width tag
/// `[len, LEAF_DOMAIN, 0, 0]` (a program constant: the leaf's width is program
/// shape) for the first block and the previous output's cell 0 for every later
/// one; the digest is the last output's cell 0. No felts hash to the zero
/// digest, as on the host.
pub fn leaf_hash(b: &mut LfmBuilder, felts: &[Felt]) -> WrapDigest {
    if felts.is_empty() {
        return WrapDigest::from_cell(zero_cell(b));
    }
    let tag = crypto::hash::poseidon1_stark::leaf_capacity(felts.len());
    let mut carry = b.digest_const(tag).as_cell();
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

/// Walk one 4-ary Merkle path whose hints are in CHILD order: per full level
/// the three siblings as ZisK's path lists them (the children other than the
/// current one, in child order), at an odd-depth top only the partner (the
/// first sibling; the two padding children are constant zeros, never hints).
///
/// With the current node at child `p = b0 + 2·b1`, the partner is sibling 0
/// when `b1 = 0` and sibling 2 when `b1 = 1`, and the other pair is siblings
/// `(1, 2)` or `(0, 1)`: hint order is the siblings rotated by one place when
/// `b1 = 1`, two `Select`s on `b1` using both of their outputs — `(partner, y)`
/// from `(s0, s2)`, then the other pair from `(s1, y)` — and [`walk4`]'s level
/// follows. Every placement is a `Select` on an index bit, so the node hashed
/// is the one the index names, whatever the hints.
pub fn walk4_child(
    b: &mut LfmBuilder,
    leaf: WrapDigest,
    bits: &[Bit],
    siblings: &[WrapDigest],
) -> WrapDigest {
    assert_eq!(
        siblings.len(),
        path_hints(bits.len()),
        "three siblings per 4-ary level, one at an odd top"
    );
    let mut cur = leaf.cells()[0];
    let mut s = siblings.iter().map(|d| d.cells()[0]);
    for pair in bits.chunks(2) {
        cur = match pair {
            [b0, b1] => {
                let s0 = s.next().expect("counted above");
                let s1 = s.next().expect("counted above");
                let s2 = s.next().expect("counted above");
                // b1 = 0: (s0, s2), then (s1, s2); b1 = 1: (s2, s0), then (s0, s1).
                let (partner, y) = b.select(*b1, s0, s2);
                let (u0, u1) = b.select(*b1, s1, y);
                let (l, r) = b.select(*b0, cur, partner);
                let (c0, c2) = b.select(*b1, l, u0);
                let (c1, c3) = b.select(*b1, r, u1);
                node4(b, [c0, c1, c2, c3])
            }
            _ => {
                // The odd top: the partner, then two zero digests.
                let partner = s.next().expect("counted above");
                let (l, r) = b.select(pair[0], cur, partner);
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

// ============================ the sparse width-8 ============================

/// The width-8 permutation's partial rounds in sparse form, precomputed from
/// the textbook constants ([`sparse_w8`]); [`w8_sparse_host`] is its host
/// form, held to `poseidon1_w8::permute` by test.
///
/// Two equivalences, both exact over the field:
///
/// - **Constants.** A partial round adds `c` before an S-box on lane 0 only,
///   so `c`'s lanes 1..8 commute past the S-box and through the round's MDS
///   into the next round's constant. Pushed forward, every partial round adds
///   one scalar to lane 0, and the last one's residue joins the first
///   terminal full round's constant.
/// - **Matrices.** Any `A = [[a, vᵀ], [w, Â]]` with `Â` invertible is
///   `Sp · D` with `D = diag(1, Â)` and `Sp = [[a, (Â⁻ᵀv)ᵀ], [w, I]]`. `D`
///   touches neither lane 0 nor the S-box, so it moves to the round before,
///   whose matrix becomes `D · M`; from the last partial round back, every
///   partial round keeps a sparse `Sp` (15 multiply-adds against the MDS's
///   64) and the first-half's last full round takes `D₀ · M`, still dense.
///   `Â` is invertible at every step: `M̂` is (every square submatrix of an
///   MDS matrix is), and each later `Â` is a product of invertible ones.
pub struct SparseW8 {
    /// Full rounds' constants, `[full round][lane]`: rounds 0–3, then 26–29
    /// (the first of those carrying the partial block's residue).
    full_rc: [[FE; W8]; 8],
    /// The first half's last full round's matrix, `D₀ · M`.
    first_matrix: [[FE; W8]; W8],
    /// Per partial round: its lane-0 constant, and its `Sp = [[a, vᵀ], [w, I]]`.
    partial_rc: [FE; w8::PARTIAL_ROUNDS],
    sp_a: [FE; w8::PARTIAL_ROUNDS],
    sp_v: [[FE; W8 - 1]; w8::PARTIAL_ROUNDS],
    sp_w: [[FE; W8 - 1]; w8::PARTIAL_ROUNDS],
}

const W8: usize = w8::STATE_FELTS;

fn w8_mds() -> [[FE; W8]; W8] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| FE::from(w8::constants::MDS_CIRC_ROW[(j + W8 - i) % W8]))
    })
}

fn mat_mul(a: &[[FE; W8]; W8], b: &[[FE; W8]; W8]) -> [[FE; W8]; W8] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..W8).fold(FE::zero(), |acc, k| acc + a[i][k] * b[k][j]))
    })
}

fn mat_vec(a: &[[FE; W8]; W8], x: &[FE; W8]) -> [FE; W8] {
    core::array::from_fn(|i| (0..W8).fold(FE::zero(), |acc, k| acc + a[i][k] * x[k]))
}

/// The inverse of a 7×7 matrix by Gauss–Jordan; `None` if singular.
fn inverse7(m: &[[FE; W8 - 1]; W8 - 1]) -> Option<[[FE; W8 - 1]; W8 - 1]> {
    const N: usize = W8 - 1;
    let mut a = *m;
    let mut inv: [[FE; N]; N] = core::array::from_fn(|i| {
        core::array::from_fn(|j| if i == j { FE::one() } else { FE::zero() })
    });
    for col in 0..N {
        let pivot = (col..N).find(|&r| a[r][col] != FE::zero())?;
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let p = a[col][col].inv().ok()?;
        for j in 0..N {
            a[col][j] *= p;
            inv[col][j] *= p;
        }
        for r in 0..N {
            if r != col && a[r][col] != FE::zero() {
                let f = a[r][col];
                for j in 0..N {
                    let (ac, ic) = (a[col][j], inv[col][j]);
                    a[r][j] = a[r][j] - f * ac;
                    inv[r][j] = inv[r][j] - f * ic;
                }
            }
        }
    }
    Some(inv)
}

/// [`SparseW8`], computed once from `poseidon1_w8`'s constants.
pub fn sparse_w8() -> &'static SparseW8 {
    static TABLE: std::sync::OnceLock<SparseW8> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let rc = |r: usize| -> [FE; W8] {
            core::array::from_fn(|i| FE::from(w8::constants::ROUND_CONSTANTS[r][i]))
        };
        let m = w8_mds();
        let half = w8::HALF_FULL_ROUNDS;
        let partial = w8::PARTIAL_ROUNDS;
        // Constants: push each partial round's lanes 1..8 forward.
        let mut partial_rc = [FE::zero(); w8::PARTIAL_ROUNDS];
        let mut carry = rc(half);
        for (k, out) in partial_rc.iter_mut().enumerate() {
            *out = carry[0];
            let mut rest = carry;
            rest[0] = FE::zero();
            let pushed = mat_vec(&m, &rest);
            carry = if k + 1 < partial {
                let next = rc(half + k + 1);
                core::array::from_fn(|i| next[i] + pushed[i])
            } else {
                pushed
            };
        }
        let mut full_rc: [[FE; W8]; 8] = core::array::from_fn(|f| {
            rc(if f < half {
                f
            } else {
                half + partial + f - half
            })
        });
        for (i, v) in full_rc[half].iter_mut().enumerate() {
            *v += carry[i];
        }
        // Matrices: from the last partial round back.
        let mut sp_a = [FE::zero(); w8::PARTIAL_ROUNDS];
        let mut sp_v = [[FE::zero(); W8 - 1]; w8::PARTIAL_ROUNDS];
        let mut sp_w = [[FE::zero(); W8 - 1]; w8::PARTIAL_ROUNDS];
        let mut a = m;
        for k in (0..partial).rev() {
            let hat: [[FE; W8 - 1]; W8 - 1] =
                core::array::from_fn(|i| core::array::from_fn(|j| a[i + 1][j + 1]));
            let hat_inv =
                inverse7(&hat).expect("an MDS matrix's square submatrices are invertible");
            sp_a[k] = a[0][0];
            // ṽᵀ = vᵀ · Â⁻¹.
            sp_v[k] = core::array::from_fn(|j| {
                (0..W8 - 1).fold(FE::zero(), |acc, i| acc + a[0][i + 1] * hat_inv[i][j])
            });
            sp_w[k] = core::array::from_fn(|i| a[i + 1][0]);
            let d: [[FE; W8]; W8] = core::array::from_fn(|i| {
                core::array::from_fn(|j| match (i, j) {
                    (0, 0) => FE::one(),
                    (0, _) | (_, 0) => FE::zero(),
                    _ => hat[i - 1][j - 1],
                })
            });
            a = mat_mul(&d, &m);
        }
        SparseW8 {
            full_rc,
            first_matrix: a,
            partial_rc,
            sp_a,
            sp_v,
            sp_w,
        }
    })
}

/// The sparse form on the host: what [`w8_permute_lanes`] emits, round for
/// round. Equal to `poseidon1_w8::permute` (`p1w16_emit_tests`).
pub fn w8_sparse_host(input: [FE; W8]) -> [FE; W8] {
    let t = sparse_w8();
    let m = w8_mds();
    let half = w8::HALF_FULL_ROUNDS;
    let sbox = |x: FE| x * x * x * x * x * x * x;
    let mut s = input;
    for r in 0..half {
        for (x, c) in s.iter_mut().zip(&t.full_rc[r]) {
            *x = sbox(*x + *c);
        }
        s = mat_vec(if r + 1 == half { &t.first_matrix } else { &m }, &s);
    }
    for k in 0..w8::PARTIAL_ROUNDS {
        s[0] = sbox(s[0] + t.partial_rc[k]);
        let s0 = s[0];
        let y0 = s[1..]
            .iter()
            .zip(&t.sp_v[k])
            .fold(t.sp_a[k] * s0, |acc, (x, v)| acc + *v * *x);
        for (x, w) in s[1..].iter_mut().zip(&t.sp_w[k]) {
            *x += *w * s0;
        }
        s[0] = y0;
    }
    for f in half..2 * half {
        for (x, c) in s.iter_mut().zip(&t.full_rc[f]) {
            *x = sbox(*x + *c);
        }
        s = mat_vec(&m, &s);
    }
    s
}

/// `seed + Σ coeff·lane` in as few rows as the terms allow: constant terms
/// fold into the seed; a variable term of coefficient one starts the chain
/// when the seed is zero; every other variable term is one `Mul`/`MulAdd`.
fn lin(b: &mut LfmBuilder, seed: FE, terms: &[(FE, Lane)]) -> Lane {
    let mut seed = seed;
    let mut vars: Vec<(FE, Felt)> = Vec::with_capacity(terms.len());
    for &(k, lane) in terms {
        match lane {
            Lane::Const(v) => seed += k * v,
            Lane::Var(f) if k != FE::zero() => vars.push((k, f)),
            Lane::Var(_) => {}
        }
    }
    if vars.is_empty() {
        return Lane::Const(seed);
    }
    let mut acc: Option<Felt> = None;
    if seed == FE::zero()
        && let Some(at) = vars.iter().position(|(k, _)| *k == FE::one())
    {
        acc = Some(vars.remove(at).1);
    }
    for (k, f) in vars {
        let kc = b.felt_const(k);
        acc = Some(match acc {
            Some(a) => b.mul_add(kc, f, a),
            None if seed == FE::zero() => b.mul(kc, f),
            None => {
                let c = b.felt_const(seed);
                b.mul_add(kc, f, c)
            }
        });
    }
    Lane::Var(acc.expect("at least one variable term"))
}

fn sbox_lane(b: &mut LfmBuilder, x: Lane) -> Lane {
    match x {
        Lane::Const(v) => Lane::Const(v * v * v * v * v * v * v),
        Lane::Var(f) => Lane::Var(sbox7(b, f)),
    }
}

/// ★ The width-8 permutation in its sparse form ([`SparseW8`]), with the
/// constant input lanes folded at emit time and only the first `keep` output
/// lanes computed. Each round's constant seeds the chains of the matrix before
/// it, as [`w8_permute`] does, so after round 0 the constants cost nothing.
///
/// Rows: a partial round is its S-box (four `Mul`s) and `Sp` (eight for the
/// new lane 0, seven `MulAdd`s for the rest) — 19 against the dense 68 — and
/// a constant input lane costs no S-box and no matrix term in round 0.
pub fn w8_permute_lanes(b: &mut LfmBuilder, input: [Lane; W8], keep: usize) -> Vec<Felt> {
    assert!((1..=W8).contains(&keep), "keep 1..=8 output lanes");
    let t = sparse_w8();
    let m = w8_mds();
    let half = w8::HALF_FULL_ROUNDS;
    let partial = w8::PARTIAL_ROUNDS;
    // Round 0's constant: added (a variable lane costs one `Add`).
    let mut s: [Lane; W8] = core::array::from_fn(|i| match input[i] {
        Lane::Const(v) => Lane::Const(v + t.full_rc[0][i]),
        Lane::Var(f) if t.full_rc[0][i] == FE::zero() => Lane::Var(f),
        Lane::Var(f) => {
            let c = b.felt_const(t.full_rc[0][i]);
            Lane::Var(b.add(f, c))
        }
    });
    // The first half: full rounds, the last one's matrix `D₀ · M`, each seeded
    // with the next round's constant (the first partial round's is lane 0's).
    for r in 0..half {
        let f: [Lane; W8] = core::array::from_fn(|i| sbox_lane(b, s[i]));
        let matrix = if r + 1 == half { &t.first_matrix } else { &m };
        s = core::array::from_fn(|o| {
            let seed = if r + 1 < half {
                t.full_rc[r + 1][o]
            } else if o == 0 {
                t.partial_rc[0]
            } else {
                FE::zero()
            };
            let terms: Vec<(FE, Lane)> = (0..W8).map(|i| (matrix[o][i], f[i])).collect();
            lin(b, seed, &terms)
        });
    }
    // The partial rounds: S-box on lane 0, then `Sp`, seeded with the next
    // round's constant (after the last, the terminal full round's).
    for k in 0..partial {
        let s0 = sbox_lane(b, s[0]);
        let next: [FE; W8] = if k + 1 < partial {
            core::array::from_fn(|i| {
                if i == 0 {
                    t.partial_rc[k + 1]
                } else {
                    FE::zero()
                }
            })
        } else {
            t.full_rc[half]
        };
        let mut terms0: Vec<(FE, Lane)> = vec![(t.sp_a[k], s0)];
        terms0.extend((1..W8).map(|j| (t.sp_v[k][j - 1], s[j])));
        let y0 = lin(b, next[0], &terms0);
        for j in 1..W8 {
            s[j] = lin(b, next[j], &[(FE::one(), s[j]), (t.sp_w[k][j - 1], s0)]);
        }
        s[0] = y0;
    }
    // The terminal full rounds, the last computing only `keep` lanes.
    for f in half..2 * half {
        let x: [Lane; W8] = core::array::from_fn(|i| sbox_lane(b, s[i]));
        let last = f + 1 == 2 * half;
        let outputs = if last { keep } else { W8 };
        let mut out = s;
        for (o, slot) in out.iter_mut().enumerate().take(outputs) {
            let seed = if last {
                FE::zero()
            } else {
                t.full_rc[f + 1][o]
            };
            let terms: Vec<(FE, Lane)> = (0..W8).map(|i| (m[o][i], x[i])).collect();
            *slot = lin(b, seed, &terms);
        }
        s = out;
    }
    s[..keep]
        .iter()
        .map(|lane| match *lane {
            Lane::Const(v) => b.felt_const(v),
            Lane::Var(f) => f,
        })
        .collect()
}
