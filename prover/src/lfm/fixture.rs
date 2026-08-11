//! The Milestone-C inner-proof fixture: a host-side FRI commitment-opening
//! prover the machine verifies.
//!
//! Structurally real, deliberately small: coset LDE domains (offset 3, the
//! production pin), row-pair Merkle leaves, per-layer commitments, the
//! **unnormalized fold** convention (`(lo+hi) + inv_x·ζ·(lo−hi)`), a compress-chain
//! transcript over the machine's own hash, query indices sampled at a
//! power-of-two bound, and a terminal polynomial checked at the queried
//! points. What it is NOT: the production 25-AIR proof format — that lands
//! when the ecosystem hash decision unblocks the real machine-facing
//! pipeline (`crypto/stark` hardcodes keccak at its Merkle layer; the
//! measured 26-site migration seam is deliberately not touched here).
//!
//! Everything here mirrors `edsl.rs` bit-exactly; the emitted verifier
//! program (`programs::fri_toy_program`) consumes exactly the arena layout
//! `fixture_prove` produces.

use math::field::traits::{IsFFTField, IsPrimeField};

use crate::tables::types::{FE, FEE, GoldilocksField};

use super::edsl::SQUEEZE_MARK;
use super::hash::{HasherKind, LfmHasher, TestPermutation};
use super::word::{LfmWord, base_word, ext_word};

/// The fixed shape — compile-time constants of the emitted program.
pub mod shape {
    /// log2 of the LDE domain size.
    pub const LOG_LDE: usize = 5; // 32 points
    pub const LDE_SIZE: usize = 1 << LOG_LDE;
    /// Trace length 8 = LDE/blowup (blowup 4).
    pub const TRACE_LEN: usize = 8;
    /// Committed base columns (one machine word per row).
    pub const NUM_COLS: usize = 4;
    /// Two folds: 32 → 16 → 8, terminal degree < 2.
    pub const NUM_LAYERS: usize = 2;
    pub const TERMINAL_LEN: usize = 2;
    pub const NUM_QUERIES: usize = 4;
    /// Query index bits (indices sampled in [0, LDE/2)).
    pub const QUERY_BITS: usize = LOG_LDE - 1;
    /// The production coset offset.
    pub const COSET_OFFSET: u64 = 3;
    /// Words per query in the openings arena.
    pub const WORDS_PER_QUERY: usize = 17;
}

/// Host mirror of [`super::edsl::SpongeVar`] — the compress chain, state 1
/// cell.
///
/// Bit-exact by construction, not by coincidence: every operation here is the
/// same sequence of [`LfmHasher::transcript`] calls the emitted program makes
/// of `LFM_HASH`, in the same order, on the same operands. The two are rewritten
/// together for exactly that reason; a divergence would show up as a fixture
/// proof the machine rejects, which is a slow and confusing way to learn about
/// it.
///
/// Parameterised by hasher because the transcript is: `Test` and `Poseidon`
/// hash a transcript step with their single domain, BLAKE3 with the `"LFMT"`
/// tag, and the host has to agree with whichever one the proof is under.
pub struct HostSponge {
    state: LfmWord,
    squeeze_index: u32,
    hasher: HasherKind,
}

impl Default for HostSponge {
    fn default() -> Self {
        Self::new()
    }
}

impl HostSponge {
    /// The chain under the machine's default hasher.
    pub fn new() -> Self {
        Self::with_hasher(HasherKind::default())
    }

    pub fn with_hasher(hasher: HasherKind) -> Self {
        HostSponge {
            state: [FE::zero(); 4],
            squeeze_index: 0,
            hasher,
        }
    }

    /// `SQ(i) = [SQUEEZE_MARK, i, 0, 0]` — the advance operand.
    pub fn squeeze_operand(i: u32) -> LfmWord {
        [
            FE::from(u64::from(SQUEEZE_MARK)),
            FE::from(u64::from(i)),
            FE::zero(),
            FE::zero(),
        ]
    }

    /// The state as it stands — for the KATs, which pin it after every step.
    pub fn state(&self) -> LfmWord {
        self.state
    }

    pub fn absorb(&mut self, c: &LfmWord) {
        self.state = self.hasher.transcript(&self.state, c);
    }

    pub fn absorb2(&mut self, c0: &LfmWord, c1: &LfmWord) {
        self.absorb(c0);
        self.absorb(c1);
    }

    /// Output the current state, then advance past it with `SQ(i)`.
    pub fn squeeze_cell(&mut self) -> LfmWord {
        let out = self.state;
        let sq = Self::squeeze_operand(self.squeeze_index);
        self.state = self.hasher.transcript(&self.state, &sq);
        self.squeeze_index += 1;
        out
    }

    pub fn squeeze_ext(&mut self) -> FEE {
        let c = self.squeeze_cell();
        FEE::new([c[0], c[1], c[2]])
    }

    pub fn squeeze_index(&mut self, nbits: usize) -> u64 {
        let c = self.squeeze_cell();
        GoldilocksField::canonical(c[0].value()) & ((1 << nbits) - 1)
    }
}

/// A binary Merkle tree over word digests (TestPermutation compress).
pub struct HostTree {
    /// levels[0] = leaves … levels.last() = [root].
    pub levels: Vec<Vec<LfmWord>>,
}

impl HostTree {
    pub fn build(leaves: Vec<LfmWord>) -> Self {
        assert!(leaves.len().is_power_of_two());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let next: Vec<LfmWord> = prev
                .chunks_exact(2)
                .map(|pair| TestPermutation.compress(&pair[0], &pair[1]))
                .collect();
            levels.push(next);
        }
        HostTree { levels }
    }

    pub fn root(&self) -> LfmWord {
        self.levels.last().unwrap()[0]
    }

    /// Sibling digests along the path from leaf `index`, level 0 first.
    pub fn open(&self, mut index: usize) -> Vec<LfmWord> {
        let mut siblings = Vec::new();
        for level in &self.levels[..self.levels.len() - 1] {
            siblings.push(level[index ^ 1]);
            index >>= 1;
        }
        siblings
    }
}

/// The fixture proof, already in the machine's arena layout:
/// arena 0 = `[main_root, l1_root, t0, t1]`; arena 1 = per-query openings
/// (`shape::WORDS_PER_QUERY` words each, order pinned by the emitter).
pub struct FriToyProof {
    pub commitments: Vec<LfmWord>,
    pub openings: Vec<LfmWord>,
}

/// The committed columns: fixed low-degree polynomials evaluated over the
/// LDE coset. Deterministic — the honest witness.
pub fn fixture_columns() -> [Vec<FE>; shape::NUM_COLS] {
    let omega = GoldilocksField::get_primitive_root_of_unity(shape::LOG_LDE as u64)
        .expect("32nd root of unity");
    let offset = FE::from(shape::COSET_OFFSET);
    core::array::from_fn(|k| {
        // degree < TRACE_LEN coefficients, fixed per column.
        let coeffs: Vec<FE> = (0..shape::TRACE_LEN)
            .map(|j| FE::from(1_000 * (k as u64 + 1) + j as u64 + 1))
            .collect();
        (0..shape::LDE_SIZE)
            .map(|i| {
                let x = &offset * omega.pow(i as u64);
                coeffs.iter().rev().fold(FE::zero(), |acc, c| acc * &x + c)
            })
            .collect()
    })
}

fn row_word(cols: &[Vec<FE>; shape::NUM_COLS], i: usize) -> LfmWord {
    core::array::from_fn(|k| cols[k][i])
}

/// Runs the fixture prover over the honest columns.
pub fn fixture_prove() -> FriToyProof {
    fixture_prove_columns(&fixture_columns())
}

/// The prover proper, over arbitrary columns (tests tamper these).
pub fn fixture_prove_columns(cols: &[Vec<FE>; shape::NUM_COLS]) -> FriToyProof {
    let omega = GoldilocksField::get_primitive_root_of_unity(shape::LOG_LDE as u64)
        .expect("32nd root of unity");
    let offset = FE::from(shape::COSET_OFFSET);
    let half = shape::LDE_SIZE / 2; // 16

    // Main tree: row-pair leaves, leaf l = compress(row 2l, row 2l+1).
    let leaves: Vec<LfmWord> = (0..shape::LDE_SIZE / 2)
        .map(|l| TestPermutation.compress(&row_word(cols, 2 * l), &row_word(cols, 2 * l + 1)))
        .collect();
    let main_tree = HostTree::build(leaves);

    let mut sponge = HostSponge::new();
    sponge.absorb(&main_tree.root());
    let alpha = sponge.squeeze_ext();
    let zeta0 = sponge.squeeze_ext();

    // g0 = α-combination of the columns, over the full LDE domain.
    let g0: Vec<FEE> = (0..shape::LDE_SIZE)
        .map(|i| {
            let row = row_word(cols, i);
            row.iter().rev().fold(FEE::zero(), |acc, v| {
                acc * &alpha + FEE::new([*v, FE::zero(), FE::zero()])
            })
        })
        .collect();

    // Fold 0 (unnormalized): g1[j] = (g0[j]+g0[j+16]) + x_j⁻¹·ζ0·(g0[j]−g0[j+16]).
    let g1: Vec<FEE> = (0..half)
        .map(|j| {
            let x = &offset * omega.pow(j as u64);
            let inv_x = x.inv().expect("nonzero domain point");
            let (lo, hi) = (&g0[j], &g0[j + half]);
            (lo + hi) + (&zeta0 * (lo - hi)) * FEE::new([inv_x, FE::zero(), FE::zero()])
        })
        .collect();

    // L1 tree co-locates fold partners: leaf j = compress(g1[j], g1[j+8]).
    let quarter = half / 2; // 8
    let l1_leaves: Vec<LfmWord> = (0..quarter)
        .map(|j| TestPermutation.compress(&ext_word(&g1[j]), &ext_word(&g1[j + quarter])))
        .collect();
    let l1_tree = HostTree::build(l1_leaves);

    sponge.absorb(&l1_tree.root());
    let zeta1 = sponge.squeeze_ext();

    // Fold 1 over the size-16 domain c²·⟨ω²⟩: y_j = c²ω^{2j}.
    let g2: Vec<FEE> = (0..quarter)
        .map(|j| {
            let y = offset.square() * omega.pow(2 * j as u64);
            let inv_y = y.inv().expect("nonzero domain point");
            let (lo, hi) = (&g1[j], &g1[j + quarter]);
            (lo + hi) + (&zeta1 * (lo - hi)) * FEE::new([inv_y, FE::zero(), FE::zero()])
        })
        .collect();

    // Terminal polynomial (degree < 2) over c⁴·⟨ω⁴⟩, from two points; the
    // remaining points must agree — the honest-witness sanity check.
    let y_a = offset.square().square();
    let y_b = &y_a * omega.pow(4u64);
    let embed = |x: &FE| FEE::new([*x, FE::zero(), FE::zero()]);
    let t1 = (&g2[1] - &g2[0]) * (embed(&(&y_b - &y_a))).inv().expect("distinct points");
    let t0 = &g2[0] - &t1 * embed(&y_a);
    for (j, v) in g2.iter().enumerate() {
        let y = &y_a * omega.pow(4 * j as u64);
        debug_assert_eq!(*v, &t0 + &t1 * embed(&y), "terminal degree bound violated");
    }

    sponge.absorb2(&ext_word(&t0), &ext_word(&t1));

    // Queries.
    let mut openings = Vec::new();
    for _ in 0..shape::NUM_QUERIES {
        let q0 = sponge.squeeze_index(shape::QUERY_BITS) as usize; // [0, 16)
        let leaf_a = q0 >> 1;
        let leaf_b = leaf_a + shape::LDE_SIZE / 4; // + 8

        // Main leaf A: its two rows + path.
        openings.push(row_word(cols, 2 * leaf_a));
        openings.push(row_word(cols, 2 * leaf_a + 1));
        openings.extend(main_tree.open(leaf_a));
        // Main leaf B.
        openings.push(row_word(cols, 2 * leaf_b));
        openings.push(row_word(cols, 2 * leaf_b + 1));
        openings.extend(main_tree.open(leaf_b));
        // L1 leaf pair + path.
        let j = q0 % quarter;
        openings.push(ext_word(&g1[j]));
        openings.push(ext_word(&g1[j + quarter]));
        openings.extend(l1_tree.open(j));
    }
    debug_assert_eq!(openings.len(), shape::NUM_QUERIES * shape::WORDS_PER_QUERY);

    FriToyProof {
        commitments: vec![
            main_tree.root(),
            l1_tree.root(),
            ext_word(&t0),
            ext_word(&t1),
        ],
        openings,
    }
}

/// A word with lane 0 bumped — the tamper helper.
pub fn bump_lane0(w: &LfmWord) -> LfmWord {
    [&w[0] + FE::one(), w[1], w[2], w[3]]
}

/// Re-exported for the emitter: ω and the coset offset as constants.
pub fn domain_constants() -> (FE, FE) {
    let omega = GoldilocksField::get_primitive_root_of_unity(shape::LOG_LDE as u64)
        .expect("32nd root of unity");
    (omega, FE::from(shape::COSET_OFFSET))
}

// Small helper so `base_word` isn't unused when records are built elsewhere.
#[allow(dead_code)]
fn _base(v: FE) -> LfmWord {
    base_word(v)
}
