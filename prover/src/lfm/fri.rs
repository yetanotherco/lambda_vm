//! FRI: the compile-time shape of the emitted verifier.
//!
//! Slice 1 of the FRI leg — the arithmetic only. `others/lfm-fri-verify-spec.md`
//! is the verified account of the production verify path this mirrors; §2 is
//! the section this file implements.
//!
//! ## Why the shape is a struct and not a runtime computation
//!
//! Production derives the fold layout at verify time from the AIR's options and
//! domain (`FriFoldLayout::new`, `fri/terminal.rs:45`). The machine cannot: it
//! is straight-line, so the layer count fixes how many walks and folds are
//! EMITTED. Every field below is therefore program shape in the sense of
//! `others/lfm-target-shape.md`, and a program that read any of it from an
//! arena would let the prover choose how much FRI to verify — the degenerate
//! case being "none".
//!
//! ## What this module is checked against
//!
//! `FriFoldLayout` is `pub(crate)` inside `crypto/stark`, so this mirror cannot
//! be differentialled against the struct itself. The oracle is production's
//! observable BEHAVIOUR instead — the vector lengths a real proof carries and
//! the verifier structurally enforces before its query loop
//! (`verifier.rs:426-448`): `fri_layers_merkle_roots.len() == num_committed`
//! and `fri_final_poly_coeffs.len() == 1 << effective_k`. That is a stronger
//! check than reading the struct would be, because those are the lengths the
//! verifier actually rejects on.
//!
//! ## What this cannot see
//!
//! It is arithmetic over a shape; it says nothing about whether the emitted
//! walk or fold is correct, only about how many of each there should be. It
//! also mirrors the CPU layout only — `fri/mod.rs` has cuda fast paths that
//! claim the same layout, unverified here and never run by the machine.

use stark::fri::schedule::FriFormat;
use stark::leaf_layout::LeafLayout;
use stark::proof::options::{FriMode, OneRowMode, ProofFormat, ProofOptions};

use crate::tables::types::FE;

use super::builder::{Bit, Cell, Ext, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::instr::ArenaId;
use super::merkle_cap::CapCells;
use super::sub_proof::{self, GroupShape};

/// The compile-time shape of one sub-proof's FRI verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FriShape {
    /// `log2` of the LDE (deep-composition) codeword length.
    pub log2_lde_length: u32,
    /// `log2` of the blowup factor.
    pub blowup_log: u32,
    /// The requested terminal log-degree, `ProofOptions::fri_final_poly_log_degree`.
    pub final_poly_log_degree: u32,
    /// The LDE coset offset. Carried rather than assumed: the emitter bakes
    /// domain constants derived from it, and the standing deferral on
    /// `coset_offset != 3` is about test COVERAGE, not about a hardcoded 3 —
    /// so the value has to come from the options, and [`Self::from_options`] is
    /// the only constructor that reads it.
    pub coset_offset: u64,
    /// Queries the sub-proof carries.
    pub num_queries: usize,
    /// The inner proof's FORMAT (design/CAP.md, design/FRI.md): its Merkle cap
    /// policy caps every committed layer tree. A verifier constant, taken from
    /// the inner proof's options — never from the proof.
    ///
    /// `format.one_row` is the table's RESOLVED leaf layout (S2): `Off` (row
    /// pairs) or `On` (one row), never `Auto` — `auto` is resolved per table
    /// from the AIR's widths ([`Self::for_layout`]) before a shape exists, and
    /// [`Self::check`] refuses an unresolved one.
    pub format: ProofFormat,
}

impl FriShape {
    /// Derive the shape from the inner proof's own options.
    ///
    /// Every FRI-relevant parameter comes from `options` — including the coset
    /// offset, which discharges the plumbing half of the `coset_offset != 3`
    /// deferral recorded in `others/lfm-assembly-obligations.md`.
    ///
    /// # Panics
    ///
    /// On `one_row = auto`: the layout of an `auto` table is resolved from its
    /// AIR's committed widths (`stark::leaf_layout::table_leaf_layout`), which
    /// the options alone do not carry — use [`Self::for_layout`] with the
    /// table's resolved layout. `Off` and `On` resolve themselves.
    pub fn from_options(options: &ProofOptions, log2_lde_length: u32) -> Self {
        let layout = match options.format.one_row {
            OneRowMode::Off => LeafLayout::RowPair,
            OneRowMode::On => LeafLayout::Row,
            OneRowMode::Auto => panic!(
                "one_row = auto resolves per table from the AIR's widths: build the \
                 FRI shape with FriShape::for_layout(options, lde, table_leaf_layout(air, n))"
            ),
        };
        Self::for_layout(options, log2_lde_length, layout)
    }

    /// The shape of a table proved under `options` whose trace trees use the
    /// RESOLVED leaf `layout` (the table's `stark::leaf_layout::table_leaf_layout`
    /// — what the host prover and verifier lay the proof out with). The
    /// resolved layout is stored in `format.one_row` (`Off` / `On`).
    pub fn for_layout(options: &ProofOptions, log2_lde_length: u32, layout: LeafLayout) -> Self {
        let mut format = options.format;
        format.one_row = if layout.is_one_row() {
            OneRowMode::On
        } else {
            OneRowMode::Off
        };
        Self {
            log2_lde_length,
            blowup_log: (options.blowup_factor as u32).trailing_zeros(),
            final_poly_log_degree: options.fri_final_poly_log_degree as u32,
            coset_offset: options.coset_offset,
            num_queries: options.fri_number_of_queries,
            format,
        }
    }

    /// Whether the table's trace trees hold one row per leaf (S2): the DEEP
    /// codeword is then committed as FRI layer 0 (the input tree), the query
    /// index has `log2(lde)` bits and no fold precedes layer 0.
    pub fn one_row(self) -> bool {
        match self.format.one_row {
            OneRowMode::Off => false,
            OneRowMode::On => true,
            OneRowMode::Auto => {
                panic!("a FRI shape carries a RESOLVED layout, never one_row = auto")
            }
        }
    }

    /// The trace trees' leaf layout this shape verifies.
    pub fn leaf_layout(self) -> LeafLayout {
        LeafLayout::from_one_row(self.one_row())
    }

    /// `log2` of the terminal codeword length, clamped to the full LDE for
    /// traces too small to fold that far (`terminal.rs:46`'s `.min(lde_log)`).
    pub fn terminal_log(self) -> u32 {
        (self.blowup_log + self.final_poly_log_degree).min(self.log2_lde_length)
    }

    /// Folds from the LDE codeword down to the terminal codeword.
    pub fn total_folds(self) -> u32 {
        self.log2_lde_length - self.terminal_log()
    }

    /// Whether the proof uses today's FRI encoding: pair layers, one sibling
    /// value per committed layer (`fri = pair` with row-pair openings).
    /// Decided by the FORMAT, never by the schedule's values: a `dp` schedule
    /// of all ones still uses the group encoding, and so does every one-row
    /// table (`FriFormat::is_legacy`: its layer 0 is the committed DEEP
    /// codeword, opened as a full group). Every non-legacy path below is the
    /// group path; the legacy emission is today's, instruction for instruction.
    pub fn is_legacy(self) -> bool {
        self.format.fri_mode == FriMode::Pair && !self.one_row()
    }

    /// The host's own FRI format for this shape (the fold-schedule DP's
    /// inputs): the mode, the query count (every FRI tree is opened once per
    /// query), the cap policy and the test-only schedule override.
    fn fri_format(self) -> FriFormat {
        FriFormat {
            mode: self.format.fri_mode,
            one_row: self.one_row(),
            num_queries: self.num_queries as u64,
            cap: self.format.merkle_cap,
            schedule_override: self.format.fri_schedule_override,
        }
    }

    /// ★ The committed layers' fold exponents, first committed layer first —
    /// the SAME function the host prover and verifier lay out with
    /// (`stark::fri::schedule::FriFormat::schedule`: the all-ones schedule
    /// under `pair`, the RULINGS-13 cost-law DP under `dp`). A format
    /// constant: nothing here reads a proof.
    ///
    /// ⚠ `num_queries` is a DP input (and a cap-policy input): a program that
    /// verifies a SUBSET of a proof's queries has a different `dp` schedule
    /// and `auto` caps than the proof unless the query count is kept.
    pub fn schedule(self) -> Vec<u8> {
        self.fri_format()
            .schedule(self.log2_lde_length, self.terminal_log())
    }

    /// Committed (Merkle-rooted) layers — one root, one auth path per query,
    /// and one Merkle walk to emit, each: the schedule's length.
    ///
    /// **`total_folds − 1` under `pair`, not `total_folds`.** The final fold is
    /// performed and never committed (`fri/mod.rs:114-118`), so a query folds
    /// once more than it authenticates. This off-by-one is the readiest way to
    /// build a verifier that looks right and checks one layer too few. Under
    /// one-row leaves the chain starts at the DEEP codeword itself (layer 0 =
    /// the input tree), so the pair schedule is `total_folds` ones.
    pub fn num_committed(self) -> usize {
        self.schedule().len()
    }

    /// Folding challenges the proof draws (FRI.md §7.3, `FriFoldLayout::num_zetas`):
    /// one per committed layer plus the final fold's for row pairs (fold 0
    /// consumes the first), one per committed layer under one row (layer 0 is
    /// committed before any challenge); none when nothing folds.
    pub fn num_zetas(self) -> usize {
        if self.total_folds() == 0 {
            0
        } else {
            self.num_committed() + usize::from(!self.one_row())
        }
    }

    /// Index of committed layer `j`'s challenge in the ζ list: `j + 1` for row
    /// pairs (`ζ₀` drove the uncommitted fold 0), `j` under one row.
    pub fn layer_zeta_index(self, layer: usize) -> usize {
        layer + usize::from(!self.one_row())
    }

    /// Fold exponent `d_j` of committed layer `j`: a leaf groups `2^{d_j}`
    /// consecutive values (1 = today's pair).
    pub fn layer_fold(self, layer: usize) -> u32 {
        u32::from(self.schedule()[layer])
    }

    /// Index bits consumed before committed layer `j`: `G_j = Σ_{i<j} d_i`.
    /// Layer `j`'s slot is `bits[G_j .. G_j + d_j]` and its tree's leaf index
    /// `bits[G_j + d_j ..]` (FRI.md §3.2).
    pub fn layer_bit_offset(self, layer: usize) -> usize {
        self.schedule()[..layer].iter().map(|&d| d as usize).sum()
    }

    /// Opened values one query's opening of committed layer `j` carries: the
    /// sibling alone under `pair`, the whole `2^{d_j}` group otherwise
    /// (FRI.md §3.4 — the query's own value included).
    pub fn layer_values(self, layer: usize) -> usize {
        if self.is_legacy() {
            1
        } else {
            1usize << self.layer_fold(layer)
        }
    }

    /// Felts committed layer `j`'s leaf hashes: `2^{d_j}` extension values of
    /// three felts — six (the pair) under `pair`.
    pub fn layer_leaf_felts(self, layer: usize) -> usize {
        3 << self.layer_fold(layer)
    }

    /// Leaf permutations one query costs across every committed layer under
    /// `hash`'s own block rule (`epoch_verify::blocks_for`).
    pub fn leaf_permutations_per_query(self, hash: super::edsl::WrapHash) -> usize {
        (0..self.num_committed())
            .map(|j| super::epoch_verify::blocks_for(self.layer_leaf_felts(j), hash))
            .sum()
    }

    /// Folds a query performs: `num_committed + 1` whenever anything folds at
    /// all, and 0 when the codeword is already terminal.
    pub fn num_folds(self) -> usize {
        self.total_folds() as usize
    }

    /// Terminal codeword length.
    pub fn terminal_len(self) -> usize {
        1usize << self.terminal_log()
    }

    /// The terminal log-degree actually used — `min(k, trace_bits)`. Equals
    /// `final_poly_log_degree` except under the clamp.
    pub fn effective_k(self) -> u32 {
        self.terminal_log() - self.blowup_log
    }

    /// Coefficients the proof carries for the terminal polynomial.
    pub fn num_terminal_coeffs(self) -> usize {
        1usize << self.effective_k()
    }

    /// Tree depth of committed layer `j`: the layer is `2^(n − 1 − G_j)`
    /// values long and its leaves group `2^{d_j}`, so the tree has
    /// `2^(n − 1 − G_j − d_j)` leaves — `n − j − 2` under `pair`.
    pub fn layer_depth(self, layer: usize) -> usize {
        let schedule = self.schedule();
        assert!(
            layer < schedule.len(),
            "layer index must be below num_committed"
        );
        let consumed: usize = schedule[..=layer].iter().map(|&d| d as usize).sum();
        self.index_bits() - consumed
    }

    /// Merkle-cap height of committed layer `i`'s tree under the format's cap
    /// policy: every layer tree is opened once per query (`0` = uncapped).
    /// The same function the host prover and verifier use
    /// (`stark::merkle_caps::StarkCaps`), at the same depth.
    pub fn layer_cap(self, layer: usize) -> usize {
        self.format
            .merkle_cap
            .height(self.num_queries, self.layer_depth(layer))
    }

    /// Merkle path length a query's opening of committed layer `i` carries:
    /// the tree's depth less its cap height (the owner path's cap is split off
    /// into the caps arena).
    pub fn layer_path_len(self, layer: usize) -> usize {
        self.layer_depth(layer) - self.layer_cap(layer)
    }

    /// Arena words the committed layers' caps occupy, once per sub-proof.
    pub fn cap_words(self, digest_words: usize) -> usize {
        (0..self.num_committed())
            .map(|i| match self.layer_cap(i) {
                0 => 0,
                c => (1usize << c) * digest_words,
            })
            .sum()
    }

    /// Permutations the committed layers' cap checks cost, once per
    /// sub-proof: `2^c − 1` parents per capped layer.
    pub fn cap_permutations(self) -> usize {
        (0..self.num_committed())
            .map(|i| super::merkle_cap::cap_root_permutations(self.layer_cap(i)))
            .sum()
    }

    /// Merkle path steps one query walks across every committed layer.
    pub fn path_steps_per_query(self) -> usize {
        (0..self.num_committed())
            .map(|i| self.layer_path_len(i))
            .sum()
    }

    /// Permutations one query costs under the production wrap hash: every
    /// committed layer's leaf (a 48-byte pair is one block under every hash;
    /// a `2^d` group is `⌈3·2^d / 8⌉` at the rate-8 algebraic sponge) plus one
    /// per path step (a parent is one compression under every hash).
    pub fn permutations_per_query(self) -> usize {
        self.leaf_permutations_per_query(super::edsl::WrapHash::production())
            + self.path_steps_per_query()
    }

    /// Index bits a query carries — `log2(lde) − 1` for row pairs (the pair
    /// index `iota`), `log2(lde)` under one-row leaves (`r` over the whole LDE,
    /// FRI.md §7.2) — which is both the TRACE trees' Merkle depth and the bit
    /// width of the index.
    ///
    /// The FRI layers consume SUFFIXES of this one decomposition rather than
    /// decompositions of their own, which is what makes the emitted walks
    /// address the same query the trace openings did. Layer `i` reads `bits[i]`
    /// as its leaf-ordering parity and `bits[i+1..]` as its walk, and
    /// `bits[i+1..].len() = n − i − 2 = layer_path_len(i)` exactly — the layer
    /// tree's depth is not a separate fact to keep in sync, it is what is left
    /// of the index after the folds already performed. (Under a Merkle cap the
    /// top `layer_cap(i)` of those bits pick the cap node instead of being
    /// walked; the split is the cap's own, [`CapCells::verify_path`].)
    pub fn index_bits(self) -> usize {
        self.leaf_layout().tree_depth(self.log2_lde_length as usize)
    }

    /// Arena words one query's FRI opening occupies: per committed layer its
    /// opened values ([`Self::layer_values`]: the symmetric evaluation, or the
    /// whole group) and its path (`digest_words` per level).
    ///
    /// `digest_words` is the BUILDER's digest width on the machine side
    /// (`edsl::digest_words(b)`) and `proof_arena::words_per_root()` on the
    /// host side — see `SubProofShape::query_words` for why it is an argument.
    pub fn query_words(self, digest_words: usize) -> usize {
        // The path stride is the DIGEST's width, not a literal two.
        let values: usize = (0..self.num_committed())
            .map(|j| self.layer_values(j))
            .sum();
        values + digest_words * self.path_steps_per_query()
    }

    /// Keccak permutations the whole sub-proof's FRI costs.
    pub fn permutations(self) -> usize {
        self.num_queries * self.permutations_per_query()
    }

    /// Invariants a caller cannot assemble their way out of.
    pub fn check(self) {
        assert!(
            self.format.one_row != OneRowMode::Auto,
            "a FRI shape carries a RESOLVED layout (FriShape::for_layout), never one_row = auto"
        );
        // The schedule covers exactly the committed folds (`FriFoldLayout`'s
        // constructor invariant, which refuses a proof otherwise).
        let schedule = self.schedule();
        let covered: u32 = schedule.iter().map(|&d| u32::from(d)).sum();
        assert!(
            schedule
                .iter()
                .all(|&d| (1..=stark::fri::schedule::FRI_SCHEDULE_DMAX).contains(&u32::from(d))),
            "every fold exponent is in 1..=DMAX: {schedule:?}"
        );
        // Row pairs: fold 0 is binary and uncommitted. One row: every fold is
        // a committed layer's (layer 0 is the DEEP codeword).
        let committed_folds = if self.one_row() {
            self.total_folds()
        } else {
            self.total_folds().saturating_sub(1)
        };
        assert_eq!(
            covered, committed_folds,
            "the schedule {schedule:?} must cover the committed folds"
        );
        assert!(
            self.blowup_log >= 1,
            "a blowup of 1 is not a low-degree extension"
        );
        assert!(
            self.log2_lde_length > self.blowup_log,
            "the LDE must be strictly larger than the blowup: a trace of one \
             row has no FRI to do"
        );
        assert!(
            self.terminal_log() <= self.log2_lde_length,
            "the terminal codeword cannot exceed the LDE"
        );
        assert!(
            self.effective_k() <= self.final_poly_log_degree,
            "the clamp can only lower the terminal degree, never raise it"
        );
        assert_eq!(
            self.terminal_len(),
            1usize << (self.blowup_log + self.effective_k()),
            "terminal_len must equal 2^(blowup_log + effective_k)"
        );
    }
}

// ============================ the emitter ============================
//
// Slice 2: the per-layer walk, the fold chain, and the terminal check. The
// shape above says how many of each; this says what each one is.
//
// ## What the machine emits, against what production runs
//
// Production's `verify_query_and_sym_openings` (`verifier.rs:660-748`) is a loop
// over committed layers with a running `(v, index)` pair. Here the loop is
// unrolled at build time and `index` never exists as a value: every use of it is
// a use of some suffix of the query's bit decomposition. The three uses map as
//
// ```text
//   production                        machine
//   ----------                        -------
//   iota % 2      (leaf order)        bits[i]
//   iota >> 1     (leaf position)     bits[i+1..]      (the walk)
//   index >>= 1   (next layer)        i += 1           (a host-side index)
// ```
//
// so the halving that production performs per layer is, here, reading one bit
// further along a vector that was decomposed once — by the trace leg, for its
// own walk. That is the join: there is no second index in the program to
// disagree with the first.
//
// ## What this cannot see
//
// It emits ONE sub-proof's FRI. Nothing here says the terminal coefficients or
// the folding challenges are the ones the transcript produced — they arrive as
// arena values, exactly as `γ` and `ζ` do in [`super::sub_proof`], and binding
// them to a transcript replay is assembly's obligation, covered by the standing
// clause in `others/lfm-assembly-obligations.md`. It also says nothing about
// whether `p₀` is the DEEP value of the authenticated opening; that is the
// previous leg's join, consumed here as cells.

/// The group shape of a FRI layer leaf: ONE extension column, so a leaf covers
/// `ROWS_PER_LEAF = 2` values and 48 bytes.
///
/// Reusing [`super::sub_proof::emit_leaf_hash`] rather than writing a second
/// gadget is deliberate and was checked rather than assumed — see
/// `fri_tests::the_fri_leaf_is_byte_identical_to_productions_own_backends`,
/// which runs the machine's leaf against BOTH production backends on vectors
/// that differ in every one of the 48 bytes.
///
/// It is worth stating why the shapes coincide at all, because the two
/// commitments are built by different code: a trace leaf applies
/// `reverse_index` INSIDE the leaf builder and concatenates column-by-column
/// across a row pair (`commitment.rs:81-91`), while a FRI layer leaf is
/// `evals.chunks_exact(2)` of an ALREADY bit-reversed single codeword
/// (`fri/mod.rs:96-99`). At one column those two descriptions produce the same
/// byte string from the same pair — the permutation a trace leaf applies is the
/// permutation a FRI codeword already carries — and at more than one column
/// they do not. So this constant is not "the trace shape with a 1 in it"; it is
/// the point where the two layouts happen to meet.
pub const FRI_LEAF_GROUP: GroupShape = GroupShape {
    num_columns: 1,
    is_ext: true,
};

/// One committed FRI layer's root, unpacked once per sub-proof.
///
/// The hoist matters at production query counts for the same reason
/// [`super::sub_proof::GroupCommitment`]'s does: a root is a per-sub-proof value
/// and a 219-query proof would otherwise pay 219 redundant `Unpack`s per layer.
pub struct LayerCommitment {
    /// The root's two words as lanes.
    /// ⚠ A `Vec`, ONE ENTRY PER DIGEST CELL, not `[_; 2]`: the array wrote the
    /// byte digest's cell COUNT into the type, and an algebraic root is one cell
    /// of four felts. `edsl::assert_digest_eq_lanes` zips a digest against these
    /// and asserts the widths agree, so it works at either width unchanged.
    pub root_lanes: Vec<[Felt; 4]>,
    /// The layer tree's authenticated Merkle cap, when the format caps it
    /// (see [`super::sub_proof::GroupCommitment::cap`]). `None` = today.
    pub cap: Option<CapCells>,
}

impl LayerCommitment {
    /// Read a layer root out of the arena and hoist its unpack.
    pub fn hint(b: &mut LfmBuilder, arena: ArenaId, base: u32) -> Self {
        // ⚠ The configuration's word count, not two: an algebraic root is ONE
        // arena word of four felts. `RootCells::words_per_root` is the single
        // definition; a second `+ 1` here would be a second definition.
        let n = super::epoch::RootCells::words_per_root(b);
        let root_lanes: Vec<[Felt; 4]> = (0..n)
            .map(|i| {
                let w = b.hint_word(arena, base + i);
                b.unpack(w)
            })
            .collect();
        LayerCommitment {
            root_lanes,
            cap: None,
        }
    }

    /// A layer commitment over lanes the caller already holds.
    ///
    /// The assembled verifier's route: a FRI layer root is absorbed by the
    /// transcript in Round 4 (right after its own `ζ`) and compared against here,
    /// and those two consumers must read one cell. See
    /// [`super::sub_proof::GroupCommitment::from_lanes`] for the same argument at
    /// the trace trees.
    pub fn from_lanes(root_lanes: Vec<[Felt; 4]>) -> Self {
        LayerCommitment {
            root_lanes,
            cap: None,
        }
    }

    /// Hint this layer tree's height-`c` cap out of `arena` at `base` and
    /// authenticate it against the root lanes, once per tree (see
    /// [`super::sub_proof::GroupCommitment::hint_cap`]). Returns the next free
    /// word; `c = 0` hints nothing.
    pub fn hint_cap(&mut self, b: &mut LfmBuilder, arena: ArenaId, base: u32, c: usize) -> u32 {
        if c == 0 {
            return base;
        }
        let (cap, next) =
            super::merkle_cap::hint_and_authenticate(b, arena, base, c, &self.root_lanes);
        self.cap = Some(cap);
        next
    }

    /// Authenticate an opened leaf of this layer at the tree's WHOLE leaf
    /// index: against the cap when capped, else against the root lanes (the
    /// uncapped emission is today's, instruction for instruction).
    fn authenticate(
        &self,
        b: &mut LfmBuilder,
        leaf: WrapDigest,
        index_bits: &[Bit],
        siblings: &[WrapDigest],
    ) {
        match &self.cap {
            None => {
                let root = edsl::wrap_merkle_walk(b, leaf, index_bits, siblings);
                edsl::assert_digest_eq_lanes(b, root, &self.root_lanes);
            }
            Some(cap) => cap.verify_path(b, leaf, index_bits, siblings),
        }
    }

    fn cap_height(&self) -> usize {
        self.cap.as_ref().map_or(0, CapCells::height)
    }
}

/// Hint and authenticate every capped committed layer's cap, in layer order,
/// out of `arena` from word 0 — once per sub-proof. The words it reads are
/// exactly [`FriShape::cap_words`].
pub fn hint_layer_caps(
    b: &mut LfmBuilder,
    shape: FriShape,
    arena: ArenaId,
    layers: &mut [LayerCommitment],
) {
    assert_eq!(
        layers.len(),
        shape.num_committed(),
        "one commitment per layer"
    );
    let mut at = 0u32;
    for (i, layer) in layers.iter_mut().enumerate() {
        at = layer.hint_cap(b, arena, at, shape.layer_cap(i));
    }
    assert_eq!(
        at as usize,
        shape.cap_words(edsl::digest_words(b) as usize),
        "the FRI caps fill exactly what the shape declares"
    );
}

/// A sub-proof's FRI data that does not depend on the query.
pub struct FriCommitments {
    /// One per committed layer, in fold order.
    pub layers: Vec<LayerCommitment>,
    /// The folding challenges `ζ₀ .. ζ_C` — `num_committed + 1` of them, or
    /// none when nothing folds. The asymmetry is the whole off-by-one of this
    /// leg: the first fold consumes the DEEP pair and is not committed, so
    /// folds exceed layers by one (`fri/mod.rs:114-118`). Under one-row
    /// leaves there is no such fold: `num_committed` challenges
    /// ([`FriShape::num_zetas`]).
    pub zetas: Vec<Ext>,
    /// The terminal polynomial's `2^effective_k` coefficients, low-to-high.
    pub coeffs: Vec<Ext>,
    /// Under the group encoding (S3, and every one-row table): per committed
    /// layer `j`, the challenges its `d_j` binary folds use — `ζ, ζ², …,
    /// ζ^{2^{d_j−1}}` for `ζ = ζ_{j+1}` (row pairs) or `ζ_j` (one row,
    /// [`FriShape::layer_zeta_index`]) (FRI.md §1.2) — squared ONCE per
    /// sub-proof, not per query. Empty under the legacy encoding, where each
    /// layer folds once with `ζ_{j+1}` itself.
    pub zeta_powers: Vec<Vec<Ext>>,
}

impl FriCommitments {
    /// The commitments of one sub-proof's FRI, with the group encoding's
    /// challenge powers hoisted ([`Self::zeta_powers`]; nothing is emitted
    /// under `pair`, so today's program is unchanged).
    pub fn new(
        b: &mut LfmBuilder,
        shape: FriShape,
        layers: Vec<LayerCommitment>,
        zetas: Vec<Ext>,
        coeffs: Vec<Ext>,
    ) -> Self {
        let zeta_powers = if shape.is_legacy() || zetas.is_empty() {
            Vec::new()
        } else {
            (0..shape.num_committed())
                .map(|j| {
                    let mut z = zetas[shape.layer_zeta_index(j)];
                    let mut powers = vec![z];
                    for _ in 1..shape.layer_fold(j) {
                        z = b.emul(z, z);
                        powers.push(z);
                    }
                    powers
                })
                .collect()
        };
        FriCommitments {
            layers,
            zetas,
            coeffs,
            zeta_powers,
        }
    }
}

/// One query's opening of one committed layer.
///
/// There is deliberately no constructor that hints — like
/// [`super::sub_proof::GroupOpening`], the values are the caller's, so what the
/// walk authenticates is what the fold consumes.
pub struct LayerOpening {
    /// Under `pair`: ONE value, `pᵢ(−υ^(2ⁱ))` — the conjugate the prover
    /// supplies. Its partner `pᵢ(υ^(2ⁱ))` is not in the proof at all: the
    /// verifier computed it as the previous fold's output, which is why a pair
    /// layer opening is one value and not two.
    ///
    /// Under the group encoding: the whole group of `2^{d_j}` values in
    /// position (bit-reversed) order, the query's own value at its slot
    /// included (FRI.md §3.4) — the leaf is hashed straight from them and the
    /// slot check `values[slot] == v` ties them to the previous fold.
    pub values: Vec<Ext>,
    /// Sibling digests, LEAF LEVEL FIRST.
    pub siblings: Vec<WrapDigest>,
}

/// What the FRI leg needs from a query the trace legs already verified.
///
/// Every field is a CELL the previous leg produced, never a fresh hint or a
/// re-derivation. [`super::sub_proof::QueryOutput`] is exactly this shape's
/// supplier.
pub struct FriQuery<'a> {
    /// `p₀(υ)` — the DEEP reconstruction at the query point (`DEEP(x_r)` under
    /// one-row leaves).
    pub p0: Ext,
    /// `p₀(−υ)` for a row-pair shape; `None` under one-row leaves, which open
    /// one point.
    pub p0_sym: Option<Ext>,
    /// `υ` (or `x_r`). Not Merkle-checked here and not hinted: it is the point
    /// the authenticated opening was folded at.
    pub point: Felt,
    /// `−υ`, needed only by the row-pair zero-fold shape; `None` under one row.
    pub point_sym: Option<Felt>,
    /// The query index low-to-high, `shape.index_bits()` of them — the cells
    /// the trace walk consumed.
    pub bits: &'a [Bit],
}

impl FriQuery<'_> {
    /// `p₀(−υ)` — a row-pair shape's.
    fn p0_sym(&self) -> Ext {
        self.p0_sym
            .expect("a row-pair FRI query carries the symmetric DEEP value")
    }
}

/// The arenas one sub-proof's FRI verification reads, in declaration order.
pub struct FriArenas {
    /// Two words per committed layer root, in fold order.
    pub roots: ArenaId,
    /// `ζ₀ .. ζ_C`, one word each. Empty when nothing folds.
    pub zetas: ArenaId,
    /// The terminal polynomial's coefficients, low-to-high.
    pub coeffs: ArenaId,
    /// Per query, per committed layer: the symmetric evaluation, then the
    /// sibling digests (two words per level).
    pub queries: ArenaId,
    /// Per capped committed layer, its `2^c` cap digests — declared only when
    /// the format caps some layer ([`FriShape::cap_words`] `> 0`).
    pub caps: Option<ArenaId>,
}

/// Declare the FRI arenas and hoist everything a query does not depend on.
pub fn declare_fri(
    b: &mut LfmBuilder,
    shape: FriShape,
    num_queries: usize,
) -> (FriArenas, FriCommitments) {
    shape.check();
    assert!(num_queries > 0, "a proof carries at least one query");
    let c = shape.num_committed();
    let num_zetas = shape.num_zetas();

    let roots = b.declare_arena(edsl::digest_words(b) * c as u32);
    let zetas = b.declare_arena(num_zetas as u32);
    let coeffs = b.declare_arena(shape.num_terminal_coeffs() as u32);
    let queries =
        b.declare_arena((num_queries * shape.query_words(edsl::digest_words(b) as usize)) as u32);
    let cap_words = shape.cap_words(edsl::digest_words(b) as usize);
    let caps = (cap_words > 0).then(|| b.declare_arena(cap_words as u32));

    let mut layers: Vec<LayerCommitment> = (0..c)
        .map(|i| LayerCommitment::hint(b, roots, edsl::digest_words(b) * i as u32))
        .collect();
    if let Some(caps) = caps {
        hint_layer_caps(b, shape, caps, &mut layers);
    }
    let zeta_cells: Vec<Ext> = (0..num_zetas as u32)
        .map(|i| b.hint_word(zetas, i).as_ext())
        .collect();
    let coeff_cells: Vec<Ext> = (0..shape.num_terminal_coeffs() as u32)
        .map(|i| b.hint_word(coeffs, i).as_ext())
        .collect();

    (
        FriArenas {
            roots,
            zetas,
            coeffs,
            queries,
            caps,
        },
        FriCommitments::new(b, shape, layers, zeta_cells, coeff_cells),
    )
}

/// Hint one query's layer openings out of the query arena.
pub fn hint_layer_openings(
    b: &mut LfmBuilder,
    shape: FriShape,
    arenas: &FriArenas,
    query: usize,
) -> Vec<LayerOpening> {
    hint_layer_openings_from(b, shape, arenas.queries, query)
}

/// [`hint_layer_openings`] against a query arena the caller declared itself.
///
/// The assembled verifier declares one arena per sub-proof and takes the roots,
/// the folding challenges and the terminal coefficients from the transcript
/// replay rather than from [`declare_fri`]'s three other arenas — so it needs
/// this one without the other three.
pub fn hint_layer_openings_from(
    b: &mut LfmBuilder,
    shape: FriShape,
    arena: ArenaId,
    query: usize,
) -> Vec<LayerOpening> {
    let stride = shape.query_words(edsl::digest_words(b) as usize);
    let mut cursor = (query * stride) as u32;
    let openings: Vec<LayerOpening> = (0..shape.num_committed())
        .map(|layer| {
            let values: Vec<Ext> = (0..shape.layer_values(layer))
                .map(|_| {
                    let v = b.hint_word(arena, cursor).as_ext();
                    cursor += 1;
                    v
                })
                .collect();
            let siblings: Vec<WrapDigest> = (0..shape.layer_path_len(layer))
                .map(|_| {
                    // The stride follows the DIGEST's width, not a literal.
                    let d = edsl::hint_digest(b, arena, cursor);
                    cursor += edsl::digest_words(b);
                    d
                })
                .collect();
            LayerOpening { values, siblings }
        })
        .collect();
    assert_eq!(
        cursor as usize,
        (query + 1) * stride,
        "the emitter's cursor must agree with the declared query stride"
    );
    openings
}

/// `P(x)` for the terminal polynomial — Horner over the coefficients the proof
/// carries, low-to-high.
///
/// See [`emit_query_fri`] for why this is an evaluation and not a lookup into a
/// materialized codeword, which is what production does.
fn emit_terminal_eval(b: &mut LfmBuilder, fri: &FriCommitments, x: Felt) -> Ext {
    edsl::horner_ext(b, x.as_ext(), &fri.coeffs)
}

/// Emit one query's FRI verification: fold, authenticate each committed layer,
/// and check the terminal polynomial.
///
/// Returns the terminal-layer value `v` — the quantity production compares
/// against its terminal codeword — so a caller can publish it. Nothing depends
/// on the caller doing so: the check is `assert_eq_ext` INSIDE the program, so a
/// query that failed would not execute.
///
/// # The terminal check is an EVALUATION, not a codeword lookup — a deliberate
/// deviation from the spec
///
/// `others/lfm-fri-verify-spec.md` §5 says to emit the FFT, on the strength of a
/// measurement (sim/24) that replacing production's terminal FFT with per-point
/// Horner cost +20M cycles in the RV64 guest verifier. That measurement is
/// sound and it does not transfer, because the two machines disagree about the
/// price of an array index.
///
/// Production materializes the terminal codeword once per proof and then does
/// `terminal_codeword.get(index)` per query — one load. This machine is
/// straight-line with no addressable memory, so the same lookup is a `Select`
/// tree over `terminal_len` cells: `terminal_len − 1` `Select`s per query. At
/// blowup 8 (`terminal_len = 1024`, 73 queries) that is 74,679 `Select`s,
/// against which the FFT itself — `(terminal_len/2)·log₂(terminal_len) = 5,120`
/// butterflies at ~3 rows each — is the smaller half of the bill.
///
/// Evaluating instead costs `2^effective_k − 1 = 127` ext `MulAdd`s per query
/// plus `total_folds` squarings for the point, and no FFT at all: 140 rows per
/// query, 10,220 at 73 queries, against ~90,000. The direction reverses because
/// the guest amortizes one FFT across queries while paying nothing per lookup,
/// and this machine pays nothing for the FFT it does not run and everything for
/// the lookup it cannot do.
///
/// The two checks are the same check, and the argument is short. The terminal
/// codeword is `P` evaluated over the terminal coset in bit-reversed order
/// (`terminal.rs:134-155`), so position `index` holds
/// `P(terminal_offset · ω_T^{br(index)})`. With `index = iota >> C`,
/// `terminal_offset = coset_offset^(2^total_folds)` and `ω_T = g^(2^total_folds)`,
/// that point is exactly `υ^(2^total_folds)` — the bits of `iota` that survive
/// the shift are the bits `br` puts inside `ω_T`'s order. So the machine raises
/// the query point to `2^total_folds` and evaluates, which also makes the
/// terminal point BOUND to the query point by construction rather than by a
/// second derivation. `fri_tests::the_terminal_point_is_the_query_point_folded`
/// checks that identity against production's own FFT at every index of several
/// shapes; a wrong exponent, a missing coset offset or a dropped bit reversal
/// all fail it.
///
/// # The zero-fold shape
///
/// When `total_folds == 0` no challenge was ever drawn and the terminal codeword
/// IS `p₀` (`verifier.rs:683-690`), so the check becomes `terminal[2·iota] = p₀`
/// and `terminal[2·iota+1] = p₀ˢ`. Under evaluation the two branches unify:
/// `2^total_folds = 1`, the two positions are `υ` and `−υ`, and the shape simply
/// evaluates `P` twice instead of once. This is not a dead branch to pin — it is
/// the real proof fixture's own shape (`min` preset over a 2^4-step epoch), and
/// a real production path for any table small enough that its LDE is already
/// terminal.
pub fn emit_query_fri(
    b: &mut LfmBuilder,
    shape: FriShape,
    fri: &FriCommitments,
    q: &FriQuery<'_>,
    openings: &[LayerOpening],
) -> Ext {
    let c = shape.num_committed();
    assert_eq!(
        q.bits.len(),
        shape.index_bits(),
        "the FRI leg reads suffixes of the trace walk's own decomposition, so \
         it needs all of its index bits (log2(lde) − 1 for row pairs, log2(lde) \
         for one row)"
    );
    assert_eq!(fri.layers.len(), c, "one commitment per committed layer");
    assert_eq!(openings.len(), c, "one opening per committed layer");
    for (i, layer) in fri.layers.iter().enumerate() {
        assert_eq!(
            layer.cap_height(),
            shape.layer_cap(i),
            "layer {i} is capped at the shape's height"
        );
    }
    assert_eq!(
        fri.coeffs.len(),
        shape.num_terminal_coeffs(),
        "the terminal polynomial carries 2^effective_k coefficients"
    );

    assert_eq!(
        q.p0_sym.is_none(),
        shape.one_row(),
        "a one-row query opens ONE point, a row-pair query two"
    );
    assert_eq!(q.point_sym.is_none(), shape.one_row());

    if shape.total_folds() == 0 {
        assert!(
            fri.zetas.is_empty(),
            "a codeword that never folds draws no folding challenge"
        );
        // One row: the terminal codeword IS the DEEP codeword and
        // `terminal[r] == DEEP(x_r)` is the whole check (host
        // `verify_query_groups`); row pairs check both points.
        let at = emit_terminal_eval(b, fri, q.point);
        b.assert_eq_ext(at, q.p0);
        if let (Some(p0_sym), Some(point_sym)) = (q.p0_sym, q.point_sym) {
            let at_sym = emit_terminal_eval(b, fri, point_sym);
            b.assert_eq_ext(at_sym, p0_sym);
        }
        return q.p0;
    }
    assert_eq!(
        fri.zetas.len(),
        shape.num_zetas(),
        "folds exceed committed layers by one (row pairs), equal them (one row)"
    );

    // `υ⁻¹`, once. Production batch-inverts across queries and REJECTS on a
    // zero point (`verifier.rs:465`, fails closed on a malformed index); the
    // machine's `Div` errors on a zero divisor, which is the same disposition —
    // an unprovable program rather than a wrong answer.
    let one = b.felt_const(FE::one());
    let inv = b.div(one, q.point);

    if shape.one_row() {
        // ★ S2 (design/FRI.md §7.3-§7.4): layer 0 IS the committed DEEP
        // codeword, so no fold precedes it. The query's value there is
        // `DEEP(x_r)` itself and the point's inverse is `x_r⁻¹`; the layer-0
        // slot check of `emit_group_layer` is then the INPUT-SLOT check
        // `group₀[slot] == DEEP(x_r)` — the only thing tying the FRI chain to
        // the authenticated trace openings (host `verify_query_groups`, M1).
        assert_eq!(
            fri.zeta_powers.len(),
            c,
            "the challenge powers are hoisted once per committed layer"
        );
        let mut v = q.p0;
        let mut y_inv = inv;
        for (j, opening) in openings.iter().enumerate() {
            (v, y_inv) = emit_group_layer(
                b,
                shape,
                j,
                &fri.layers[j],
                &fri.zeta_powers[j],
                v,
                y_inv,
                opening,
                q.bits,
            );
        }
        return emit_terminal_check(b, shape, fri, q.point, v);
    }

    // Fold 0 consumes the DEEP pair and authenticates nothing: there is no
    // layer under it, which is why `zetas` is one longer than `layers`.
    let mut v = edsl::fri_fold(b, q.p0, q.p0_sym(), fri.zetas[0], inv);

    // The point chain is one squaring per layer and nothing else — no bit
    // reversal, no domain lookup, no coset offset past the first point
    // (spec §6). And no parity branch, because the sign the odd slot introduces
    // into `x⁻¹` is the same sign it introduces into `v − sym`, so the two
    // cancel (spec §3). Parity is consulted ONLY for the leaf byte order below.
    if shape.is_legacy() {
        let mut inv_pow = inv;
        for (i, opening) in openings.iter().enumerate() {
            (v, inv_pow) = emit_pair_layer(
                b,
                i,
                &fri.layers[i],
                fri.zetas[i + 1],
                v,
                inv_pow,
                opening,
                q.bits,
            );
        }
    } else {
        // The group encoding (S3): committed layer `j` opens a whole coset of
        // `2^{d_j}` values. `y⁻¹` at committed layer 0 is `υ^{−2}`, and each
        // layer hands the next its own point (`x_g^{2^d}`, FRI.md §1.1).
        assert_eq!(
            fri.zeta_powers.len(),
            c,
            "the challenge powers are hoisted once per committed layer"
        );
        let mut y_inv = b.mul(inv, inv);
        for (j, opening) in openings.iter().enumerate() {
            (v, y_inv) = emit_group_layer(
                b,
                shape,
                j,
                &fri.layers[j],
                &fri.zeta_powers[j],
                v,
                y_inv,
                opening,
                q.bits,
            );
        }
    }

    emit_terminal_check(b, shape, fri, q.point, v)
}

/// `x = υ^(2^total_folds)`: where the fold chain has arrived, and the terminal
/// codeword's point at position `iota >> C` (`r >> total_folds` under one row
/// — the same point, since `x_r` IS the query point at layer 0). See
/// [`emit_query_fri`]'s doc comment. Asserts `P(x) == v` and returns `v`.
fn emit_terminal_check(
    b: &mut LfmBuilder,
    shape: FriShape,
    fri: &FriCommitments,
    point: Felt,
    v: Ext,
) -> Ext {
    let mut x = point;
    for _ in 0..shape.total_folds() {
        x = b.mul(x, x);
    }
    let at = emit_terminal_eval(b, fri, x);
    b.assert_eq_ext(at, v);
    v
}

/// One committed layer under TODAY's pair encoding (`FriShape::is_legacy`):
/// the opening carries the sibling value only. With the query's value `v` at
/// this layer and `x⁻¹` of its point one layer up (`inv_pow`), order the pair
/// by the parity bit `bits[layer]`, hash it as the leaf, authenticate it at the
/// tree's leaf index `bits[layer + 1..]`, square the point and fold with
/// `zeta`. Returns `(v, inv_pow)` at the next layer. The rows it emits are
/// `stark::fri::schedule::fri_pair_layer_rows` (pinned in `fri_group_tests`).
#[allow(clippy::too_many_arguments)]
pub fn emit_pair_layer(
    b: &mut LfmBuilder,
    layer: usize,
    commitment: &LayerCommitment,
    zeta: Ext,
    v: Ext,
    inv_pow: Felt,
    opening: &LayerOpening,
    bits: &[Bit],
) -> (Ext, Felt) {
    assert_eq!(opening.values.len(), 1, "a pair layer opens its sibling");
    let sym = opening.values[0];
    // `if index % 2 == 1 { [sym, v] } else { [v, sym] }` (`verifier.rs:637`)
    // — the even codeword slot leads. `select(bit, l, r)` returns `(l, r)`
    // at 0 and `(r, l)` at 1, so this IS that conditional.
    let (first, second) = b.select(bits[layer], v.as_cell(), sym.as_cell());
    let leaf = sub_proof::emit_leaf_hash(b, FRI_LEAF_GROUP, &[first, second]);
    // `bits[layer+1..]` is this layer tree's whole leaf index; a cap walks its
    // low bits and muxes the top ones.
    commitment.authenticate(b, leaf, &bits[layer + 1..], &opening.siblings);

    // `evaluation_point_vec[i] = υ^(−2^(i+1))` — `inv.square()` then one
    // squaring per layer (`verifier.rs:692-697`).
    let inv_pow = b.mul(inv_pow, inv_pow);
    (edsl::fri_fold(b, v, sym, zeta, inv_pow), inv_pow)
}

/// The program constants of one group fold of exponent `d` (FRI.md §1.3), in
/// the host verifier's own terms (`fri::group::group_fold`, whose table is
/// `ω_{2^d}^t` for `ω_{2^d} = get_primitive_root_of_unity(d)`):
///
/// - `slot[ℓ] = ω_{2^d}^{2^{d−1−ℓ}}`, so `x_g⁻¹ = y⁻¹·Π_ℓ slot[ℓ]^{s_ℓ}
///   = y⁻¹·ω_{2^d}^{br_d(s)}` for the slot `s` (bits `s_ℓ`, low first);
/// - `kappa[ℓ][j] = ω_{2^d}^{−2^ℓ·br_{d−ℓ−1}(j)}`: fold level `ℓ`'s pair `j`
///   sits at `(X, −X)` with `X⁻¹ = x_g^{−2^ℓ}·kappa[ℓ][j]` (`kappa[ℓ][0] = 1`).
fn group_fold_constants(d: u32) -> (Vec<FE>, Vec<Vec<FE>>) {
    use math::fft::bit_reversing::reverse_index;
    use math::field::traits::IsFFTField;

    let n = 1usize << d;
    let w = <crate::tables::types::GoldilocksField as IsFFTField>::get_primitive_root_of_unity(
        u64::from(d),
    )
    .expect("2^d divides the two-adicity for d <= DMAX");
    let pow = |e: usize| w.pow(e as u64);
    let slot = (0..d as usize)
        .map(|l| pow(1 << (d as usize - 1 - l)))
        .collect();
    let kappa = (0..d as usize)
        .map(|l| {
            let half = n >> (l + 1);
            (0..half)
                .map(|j| {
                    let br = if half > 1 {
                        reverse_index(j, half as u64)
                    } else {
                        0
                    };
                    pow((n - (br << l)) % n)
                })
                .collect()
        })
        .collect();
    (slot, kappa)
}

// The load-bearing test of the slot check (the in-guest M1): a test build can
// emit without it and watch a moved `p₀` execute. Production has no switch.
#[cfg(test)]
thread_local! {
    pub(super) static SKIP_SLOT_CHECK: core::cell::Cell<bool> =
        const { core::cell::Cell::new(false) };
}

#[inline]
fn skip_slot_check() -> bool {
    #[cfg(test)]
    {
        SKIP_SLOT_CHECK.with(|c| c.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// `values[slot]` for the slot's bits, LOW first: a balanced mux of
/// `2^d − 1` `Select`s over ext cells, pairs `(2t, 2t + 1)` level by level.
fn emit_value_mux(b: &mut LfmBuilder, values: &[Ext], slot_bits: &[Bit]) -> Ext {
    assert_eq!(
        values.len(),
        1usize << slot_bits.len(),
        "one mux level per slot bit"
    );
    let mut level: Vec<Cell> = values.iter().map(|v| v.as_cell()).collect();
    for bit in slot_bits {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(b.select(*bit, pair[0], pair[1]).0);
        }
        level = next;
    }
    level[0].as_ext()
}

/// ★ One committed layer under the group encoding (S3; FRI.md §1.3, §3.2, §6).
///
/// With `d = d_j`, `G = G_j`, the query's bits `bits` (low first, all
/// `index_bits`), its value `v` at this layer (the previous fold's output) and
/// the inverse `y⁻¹` of its point here:
///
/// 1. **slot check** — `values[bits[G..G+d]] == v`, a `2^d − 1`-select mux and
///    an `assert_eq_ext`: the round-consistency check tying the opened group
///    to the value the previous fold produced (M1 on the host);
/// 2. **the group is the leaf** — hashed in full, position order (a
///    `GroupShape` of `2^{d−1}` ext columns covers `2^d` values; REVIEW-FRI
///    F6), and authenticated at the tree's leaf index `bits[G+d..]` against the
///    layer's root or cap;
/// 3. **the group fold** with `ζ, ζ², …, ζ^{2^{d−1}}`: `x_g⁻¹ = y⁻¹·ω_{2^d}^{br_d(s)}`
///    (`d` selects of constants and `d` base muls), then `d` levels of
///    `fri_fold` over the pairs, the level's `x_g^{−2^ℓ}` squared once per
///    level. A level with more than two pairs folds `ζ^{2^ℓ}·x_g^{−2^ℓ}` into
///    the challenge once (one `emul_base`) and multiplies each pair by its
///    constant inside the fold; a level with one or two pairs multiplies the
///    point instead (at most one base mul). After `d` levels the point is
///    `x_g^{−2^d}`, the NEXT layer's `y⁻¹`.
///
/// Returns `(v, y⁻¹)` at the next layer.
#[allow(clippy::too_many_arguments)]
pub fn emit_group_layer(
    b: &mut LfmBuilder,
    shape: FriShape,
    layer: usize,
    commitment: &LayerCommitment,
    zeta_powers: &[Ext],
    v: Ext,
    y_inv: Felt,
    opening: &LayerOpening,
    bits: &[Bit],
) -> (Ext, Felt) {
    let d = shape.layer_fold(layer);
    let g = shape.layer_bit_offset(layer);
    let n = 1usize << d;
    assert_eq!(opening.values.len(), n, "a group layer opens 2^d values");
    assert_eq!(
        zeta_powers.len(),
        d as usize,
        "one challenge power per fold level"
    );
    assert_eq!(
        bits.len() - (g + d as usize),
        shape.layer_depth(layer),
        "the tree's leaf index is what is left of the query after the slot"
    );
    let slot_bits = &bits[g..g + d as usize];

    // (1) the slot check.
    let v_slot = emit_value_mux(b, &opening.values, slot_bits);
    if !skip_slot_check() {
        b.assert_eq_ext(v_slot, v);
    }

    // (2) the group is the leaf.
    let cells: Vec<Cell> = opening.values.iter().map(|x| x.as_cell()).collect();
    let leaf = sub_proof::emit_leaf_hash(
        b,
        GroupShape {
            num_columns: n / 2,
            is_ext: true,
        },
        &cells,
    );
    commitment.authenticate(b, leaf, &bits[g + d as usize..], &opening.siblings);

    // (3) the group fold.
    let (slot_factors, kappa) = group_fold_constants(d);
    let mut xinv = y_inv;
    for (bit, factor) in slot_bits.iter().zip(&slot_factors) {
        let one = b.felt_const(FE::one());
        let f = b.felt_const(*factor);
        let (chosen, _) = b.select(*bit, one.as_cell(), f.as_cell());
        xinv = b.mul(xinv, Felt(chosen.0));
    }
    let mut vals = opening.values.clone();
    for (l, zeta) in zeta_powers.iter().enumerate() {
        let half = vals.len() / 2;
        let scaled = (half > 2).then(|| b.emul_base(*zeta, xinv));
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            let (lo, hi) = (vals[2 * j], vals[2 * j + 1]);
            let folded = match scaled {
                Some(zx) => {
                    let k = b.felt_const(kappa[l][j]);
                    edsl::fri_fold(b, lo, hi, zx, k)
                }
                None => {
                    let x = if j == 0 {
                        xinv
                    } else {
                        let k = b.felt_const(kappa[l][j]);
                        b.mul(xinv, k)
                    };
                    edsl::fri_fold(b, lo, hi, *zeta, x)
                }
            };
            next.push(folded);
        }
        vals = next;
        xinv = b.mul(xinv, xinv);
    }
    (vals[0], xinv)
}

/// A whole sub-proof, both legs: every query's openings authenticated and folded
/// to `p₀` ([`super::sub_proof::emit_sub_proof_with_bits`]), then that `p₀`
/// folded down FRI's layers to the terminal check.
///
/// This is where the two legs become one program rather than two. Every seam is
/// a shared CELL, not a shared convention: `p₀`/`p₀ˢ` are the DEEP outputs, `υ`
/// is the point they were evaluated at, and the index bits are the ones the
/// trace walk selected on. Returns the per-query terminal values.
pub fn emit_sub_proof_with_fri(
    b: &mut LfmBuilder,
    sub: &super::sub_proof::SubProofShape,
    shape: FriShape,
    num_queries: usize,
) -> (super::sub_proof::SubProofArenas, FriArenas, Vec<Ext>) {
    assert_eq!(
        sub.log2_lde_length, shape.log2_lde_length,
        "both legs verify the same sub-proof over the same LDE domain"
    );
    assert_eq!(
        sub.merkle_depth,
        shape.index_bits(),
        "the FRI layers consume suffixes of the trace walk's decomposition, so \
         the two shapes must agree about how long it is"
    );
    assert_eq!(
        shape.num_queries, num_queries,
        "the query count is one shape, declared once"
    );
    assert_eq!(
        sub.layout,
        shape.leaf_layout(),
        "both legs verify one table at one leaf layout"
    );

    let (sub_arenas, queries) = super::sub_proof::emit_sub_proof_with_bits(b, sub, num_queries);
    let (fri_arenas, fri) = declare_fri(b, shape, num_queries);

    let terminal = queries
        .iter()
        .enumerate()
        .map(|(i, out)| {
            let openings = hint_layer_openings(b, shape, &fri_arenas, i);
            emit_query_fri(
                b,
                shape,
                &fri,
                &FriQuery {
                    p0: out.deep,
                    p0_sym: out.deep_sym,
                    point: out.point,
                    point_sym: out.point_sym,
                    bits: &out.bits,
                },
                &openings,
            )
        })
        .collect();

    (sub_arenas, fri_arenas, terminal)
}
