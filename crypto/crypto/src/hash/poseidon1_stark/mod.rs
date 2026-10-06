//! The hashing of a Poseidon1 base STARK, constructed exactly as ZisK
//! (pil2-proofman v1.3.0-alpha, `Poseidon1` hash family) constructs it.
//!
//! ⚠ **Not a shipped hash**: an exploration-branch instance for a like-for-like
//! comparison with ZisK. Nothing commits with it by default.
//!
//! # The instance (every item checked against ZisK's own code, `zisk_kat.rs`)
//!
//! | primitive | construction |
//! |---|---|
//! | permutation | [`super::poseidon1_w16::permute`]: Goldilocks, `x^7`, `R_F = 8`, `R_P = 22`, Grain constants, Plonky3's small circulant MDS |
//! | leaf | [`linear_hash`]: rate 12; the first block's capacity is the width tag [`leaf_capacity`] `[len, LEAF_DOMAIN, 0, 0]`, every later block sees the previous output's lanes `0..4` there. ZisK's own leaf has a zero first capacity ([`zisk_linear_hash`], the KATs' reference) |
//! | node | [`super::poseidon1_w16::compress4`]: the four child digests in child order, permuted, truncated to lanes `0..4` |
//! | tree | arity 4; a level whose length is not a multiple of 4 is padded with zero digests ([`Merkle4`]) |
//! | transcript | [`Transcript`]: an overwrite sponge, rate 12, the previous state's lanes `0..4` as capacity, every lane squeezed |
//! | grinding | [`grinding_lane0`]: the WIDTH-8 permutation of `[c0, c1, c2, nonce, 0, 0, 0, 0]`, lane 0 below `2^(64 − bits)` |
//!
//! The leaf is the one place this instance departs from ZisK's: its first block's
//! capacity carries the leaf's width and a leaf domain (the review's width tag,
//! REV-P1-JUDGE §3.4), as RPX's leaf capacity does. So a leaf binds its own
//! width: a leaf of `n` felts and the same leaf zero-padded hash apart, and a
//! 12-felt leaf is not a node whose fourth child is zero. It costs no
//! permutation (the first block's capacity lanes are otherwise zero). The node
//! and the transcript carry no tag: the verifier fixes every tree's depth and
//! every absorb schedule.

#[cfg(test)]
mod tests;
#[cfg(test)]
mod zisk_kat;

use alloc::vec::Vec;

use super::poseidon1_w8;
use super::poseidon1_w16::{self, ARITY, DIGEST_FELTS, Digest, Fp, compress4, permute};

const WIDTH: usize = poseidon1_w16::STATE_FELTS;
const RATE: usize = poseidon1_w16::RATE_FELTS;

/// Felts of one Merkle path level: the three siblings, in child order.
pub const SIBLING_FELTS: usize = (ARITY - 1) * DIGEST_FELTS;

/// The leaf domain a leaf's first-block capacity carries: `"P1WL"`
/// ([`poseidon1_w16::DOMAIN_LEAF`]), distinct from RPX's.
pub const LEAF_DOMAIN: u64 = poseidon1_w16::DOMAIN_LEAF;

/// ★ The width tag: the first block's capacity of a leaf of `num_felts` felts,
/// `[num_felts, LEAF_DOMAIN, 0, 0]`. It mirrors RPX's `leaf_capacity` with the
/// whole length rather than its residue, so two leaf widths never share a
/// first block.
pub fn leaf_capacity(num_felts: usize) -> Digest {
    [
        Fp::from(num_felts as u64),
        Fp::from(LEAF_DOMAIN),
        Fp::zero(),
        Fp::zero(),
    ]
}

/// The leaf hash: ZisK's (`linear_hash_seq` at width 16) with the width tag.
///
/// Each block of up to 12 felts overwrites the rate (zero-filled past its end);
/// the capacity is [`leaf_capacity`] for the first block and the previous
/// output's digest lanes `0..4` for every later one. The digest is lanes `0..4`
/// of the last output. An empty input never permutes and hashes to the zero
/// digest.
pub fn linear_hash(felts: &[Fp]) -> Digest {
    linear_hash_from(felts, leaf_capacity(felts.len()))
}

/// ZisK's own leaf hash: [`linear_hash`] with a zero first capacity. The
/// reference ZisK's known-answer vectors pin; no commitment uses it.
pub fn zisk_linear_hash(felts: &[Fp]) -> Digest {
    linear_hash_from(felts, [Fp::zero(); DIGEST_FELTS])
}

/// The rate-12 chain with `first` as the first block's capacity.
fn linear_hash_from(felts: &[Fp], first: Digest) -> Digest {
    let mut state = [Fp::zero(); WIDTH];
    state[RATE..].copy_from_slice(&first);
    for (k, block) in felts.chunks(RATE).enumerate() {
        if k > 0 {
            let carry = [state[0], state[1], state[2], state[3]];
            state[RATE..].copy_from_slice(&carry);
        }
        state[..RATE].fill(Fp::zero());
        state[..block.len()].copy_from_slice(block);
        state = permute(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// A 4-ary Merkle tree over leaf digests, every level kept.
pub struct Merkle4 {
    /// Bottom first; every level but the root padded with zero digests to a
    /// multiple of 4.
    levels: Vec<Vec<Digest>>,
    /// Leaves before padding.
    leaves: usize,
}

impl Merkle4 {
    /// Builds the tree. An empty leaf set has no tree.
    pub fn new(leaves: &[Digest]) -> Option<Self> {
        if leaves.is_empty() {
            return None;
        }
        let mut levels = vec![leaves.to_vec()];
        loop {
            let level = levels.last_mut().expect("levels starts non-empty");
            if level.len() == 1 {
                break;
            }
            level.resize(
                level.len().div_ceil(ARITY) * ARITY,
                [Fp::zero(); DIGEST_FELTS],
            );
            let parents = level
                .chunks_exact(ARITY)
                .map(|c| compress4(&[c[0], c[1], c[2], c[3]]))
                .collect();
            levels.push(parents);
        }
        Some(Self {
            levels,
            leaves: leaves.len(),
        })
    }

    /// The root.
    pub fn root(&self) -> Digest {
        self.levels.last().expect("a tree has a root")[0]
    }

    /// Levels from leaf to root, i.e. the length of every path.
    pub fn depth(&self) -> usize {
        self.levels.len() - 1
    }

    /// The path of leaf `index`: per level, its three siblings in child order.
    /// A padding position is not a leaf.
    pub fn path(&self, index: usize) -> Option<Vec<[Fp; SIBLING_FELTS]>> {
        if index >= self.leaves {
            return None;
        }
        let mut i = index;
        let path = self.levels[..self.depth()]
            .iter()
            .map(|level| {
                let first = i / ARITY * ARITY;
                let mut siblings = [Fp::zero(); SIBLING_FELTS];
                let mut slots = siblings.chunks_exact_mut(DIGEST_FELTS);
                for c in (first..first + ARITY).filter(|&c| c != i) {
                    slots
                        .next()
                        .expect("three siblings")
                        .copy_from_slice(&level[c]);
                }
                i /= ARITY;
                siblings
            })
            .collect();
        Some(path)
    }
}

/// Recomputes the root from leaf `index`'s digest and path, as ZisK's
/// `calculate_root_from_proof` does, and compares it with `root`.
///
/// Unlike ZisK's `verify_mt` it also rejects an index past the tree, so one leaf
/// has one path.
pub fn verify_path4(
    root: &Digest,
    leaf: &Digest,
    index: usize,
    path: &[[Fp; SIBLING_FELTS]],
) -> bool {
    let mut node = *leaf;
    let mut i = index;
    for siblings in path {
        let at = i % ARITY;
        let mut children = [[Fp::zero(); DIGEST_FELTS]; ARITY];
        let mut sib = siblings.chunks_exact(DIGEST_FELTS);
        for (c, child) in children.iter_mut().enumerate() {
            if c == at {
                *child = node;
            } else {
                child.copy_from_slice(sib.next().expect("three siblings"));
            }
        }
        node = compress4(&children);
        i /= ARITY;
    }
    i == 0 && node == *root
}

/// ZisK's Fiat–Shamir transcript over the width-16 permutation
/// (`Transcript<Goldilocks, Poseidon1_16>`).
///
/// Absorbed felts collect in a 12-felt buffer; a full buffer, or a squeeze after
/// an absorb, permutes `buffer (zero-filled) ‖ state[0..4]`. A squeeze reads the
/// output's lanes `0, 1, …, 15` in order and permutes again once all sixteen are
/// read.
#[derive(Clone)]
pub struct Transcript {
    state: [Fp; WIDTH],
    pending: [Fp; RATE],
    pending_len: usize,
    /// Output lanes not yet squeezed; 0 forces a permutation first.
    unread: usize,
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Transcript {
    /// An empty transcript.
    pub fn new() -> Self {
        Self {
            state: [Fp::zero(); WIDTH],
            pending: [Fp::zero(); RATE],
            pending_len: 0,
            unread: 0,
        }
    }

    fn update(&mut self) {
        let mut input = [Fp::zero(); WIDTH];
        input[..self.pending_len].copy_from_slice(&self.pending[..self.pending_len]);
        input[RATE..].copy_from_slice(&self.state[..WIDTH - RATE]);
        self.state = permute(input);
        self.pending_len = 0;
        self.unread = WIDTH;
    }

    /// Absorbs one felt.
    pub fn put1(&mut self, x: Fp) {
        self.pending[self.pending_len] = x;
        self.pending_len += 1;
        self.unread = 0;
        if self.pending_len == RATE {
            self.update();
        }
    }

    /// Absorbs felts in order.
    pub fn put(&mut self, xs: &[Fp]) {
        for x in xs {
            self.put1(*x);
        }
    }

    /// Squeezes one felt.
    pub fn squeeze1(&mut self) -> Fp {
        if self.unread == 0 {
            self.update();
        }
        let x = self.state[WIDTH - self.unread];
        self.unread -= 1;
        x
    }

    /// Squeezes an extension-field challenge: three consecutive felts.
    pub fn field(&mut self) -> [Fp; 3] {
        [self.squeeze1(), self.squeeze1(), self.squeeze1()]
    }

    /// ZisK's `get_state`: flushes a non-empty buffer, then returns the state.
    /// It does not reset the squeeze position.
    pub fn state(&mut self) -> [Fp; WIDTH] {
        if self.pending_len > 0 {
            self.update();
        }
        self.state
    }

    /// ZisK's `get_permutations`: `n` integers of `n_bits` bits each, read
    /// least-significant bit first from the low 63 bits of consecutive squeezed
    /// felts. `n_bits` must be at most 63.
    pub fn indices(&mut self, n: usize, n_bits: usize) -> Vec<u64> {
        assert!(
            n > 0 && (1..=63).contains(&n_bits),
            "indices: n {n}, n_bits {n_bits}"
        );
        let felts: Vec<u64> = (0..(n * n_bits - 1) / 63 + 1)
            .map(|_| self.squeeze1().canonical())
            .collect();
        let mut bit = 0usize;
        (0..n)
            .map(|_| {
                let mut v = 0u64;
                for j in 0..n_bits {
                    v |= ((felts[bit / 63] >> (bit % 63)) & 1) << j;
                    bit += 1;
                }
                v
            })
            .collect()
    }
}

/// Lane 0 (canonical) of the width-8 permutation of `[c0, c1, c2, nonce, 0, 0,
/// 0, 0]`: ZisK's proof-of-work hash.
pub fn grinding_lane0(challenge: &[Fp; 3], nonce: &Fp) -> u64 {
    let mut s = [Fp::zero(); poseidon1_w8::STATE_FELTS];
    s[..3].copy_from_slice(challenge);
    s[3] = *nonce;
    let out = poseidon1_w8::permute(s);
    out[0].canonical()
}

/// Does `nonce` meet `bits` of proof of work for `challenge`?
pub fn grinding_ok(challenge: &[Fp; 3], nonce: &Fp, bits: u32) -> bool {
    match bits {
        0 => true,
        1..=63 => grinding_lane0(challenge, nonce) < 1u64 << (64 - bits),
        _ => false,
    }
}
