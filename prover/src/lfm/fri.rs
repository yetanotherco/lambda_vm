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

use stark::proof::options::ProofOptions;

use crate::tables::types::FE;

use super::builder::{Bit, Ext, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::instr::ArenaId;
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
}

impl FriShape {
    /// Derive the shape from the inner proof's own options.
    ///
    /// Every FRI-relevant parameter comes from `options` — including the coset
    /// offset, which discharges the plumbing half of the `coset_offset != 3`
    /// deferral recorded in `others/lfm-assembly-obligations.md`.
    pub fn from_options(options: &ProofOptions, log2_lde_length: u32) -> Self {
        Self {
            log2_lde_length,
            blowup_log: (options.blowup_factor as u32).trailing_zeros(),
            final_poly_log_degree: options.fri_final_poly_log_degree as u32,
            coset_offset: options.coset_offset,
            num_queries: options.fri_number_of_queries,
        }
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

    /// Committed (Merkle-rooted) layers — one root, one auth path per query,
    /// and one Merkle walk to emit, each.
    ///
    /// **`total_folds − 1`, not `total_folds`.** The final fold is performed
    /// and never committed (`fri/mod.rs:114-118`), so a query folds once more
    /// than it authenticates. This off-by-one is the readiest way to build a
    /// verifier that looks right and checks one layer too few.
    pub fn num_committed(self) -> usize {
        self.total_folds().saturating_sub(1) as usize
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

    /// Merkle path length for committed layer `i`: that layer's codeword is
    /// `2^(n−i−1)` long and its leaves are pairs, so the tree has `2^(n−i−2)`
    /// leaves.
    pub fn layer_path_len(self, layer: usize) -> usize {
        (self.log2_lde_length as usize)
            .checked_sub(layer + 2)
            .expect("layer index must be below num_committed")
    }

    /// Merkle path steps one query walks across every committed layer.
    pub fn path_steps_per_query(self) -> usize {
        (0..self.num_committed())
            .map(|i| self.layer_path_len(i))
            .sum()
    }

    /// Keccak permutations one query costs: one leaf hash per committed layer
    /// (a 48-byte pair, one rate block) plus one per path step (64 bytes, one
    /// rate block).
    pub fn permutations_per_query(self) -> usize {
        self.num_committed() + self.path_steps_per_query()
    }

    /// Index bits a query carries — `log2(lde) − 1`, which is both the TRACE
    /// trees' Merkle depth and the bit width of `iota`.
    ///
    /// The FRI layers consume SUFFIXES of this one decomposition rather than
    /// decompositions of their own, which is what makes the emitted walks
    /// address the same query the trace openings did. Layer `i` reads `bits[i]`
    /// as its leaf-ordering parity and `bits[i+1..]` as its walk, and
    /// `bits[i+1..].len() = n − i − 2 = layer_path_len(i)` exactly — the layer
    /// tree's depth is not a separate fact to keep in sync, it is what is left
    /// of the index after the folds already performed.
    pub fn index_bits(self) -> usize {
        self.log2_lde_length as usize - 1
    }

    /// Arena words one query's FRI opening occupies: per committed layer the
    /// symmetric evaluation (one word) and its path (two words per level).
    pub fn query_words(self) -> usize {
        self.num_committed() + 2 * self.path_steps_per_query()
    }

    /// Keccak permutations the whole sub-proof's FRI costs.
    pub fn permutations(self) -> usize {
        self.num_queries * self.permutations_per_query()
    }

    /// Invariants a caller cannot assemble their way out of.
    pub fn check(self) {
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
    pub root_lanes: [[Felt; 4]; 2],
}

impl LayerCommitment {
    /// Read a layer root out of the arena and hoist its unpack.
    pub fn hint(b: &mut LfmBuilder, arena: ArenaId, base: u32) -> Self {
        let w0 = b.hint_word(arena, base);
        let w1 = b.hint_word(arena, base + 1);
        LayerCommitment {
            root_lanes: [b.unpack(w0), b.unpack(w1)],
        }
    }

    /// A layer commitment over lanes the caller already holds.
    ///
    /// The assembled verifier's route: a FRI layer root is absorbed by the
    /// transcript in Round 4 (right after its own `ζ`) and compared against here,
    /// and those two consumers must read one cell. See
    /// [`super::sub_proof::GroupCommitment::from_lanes`] for the same argument at
    /// the trace trees.
    pub fn from_lanes(root_lanes: [[Felt; 4]; 2]) -> Self {
        LayerCommitment { root_lanes }
    }
}

/// A sub-proof's FRI data that does not depend on the query.
pub struct FriCommitments {
    /// One per committed layer, in fold order.
    pub layers: Vec<LayerCommitment>,
    /// The folding challenges `ζ₀ .. ζ_C` — `num_committed + 1` of them, or
    /// none when nothing folds. The asymmetry is the whole off-by-one of this
    /// leg: the first fold consumes the DEEP pair and is not committed, so
    /// folds exceed layers by one (`fri/mod.rs:114-118`).
    pub zetas: Vec<Ext>,
    /// The terminal polynomial's `2^effective_k` coefficients, low-to-high.
    pub coeffs: Vec<Ext>,
}

/// One query's opening of one committed layer.
///
/// There is deliberately no constructor that hints — like
/// [`super::sub_proof::GroupOpening`], the values are the caller's, so what the
/// walk authenticates is what the fold consumes.
pub struct LayerOpening {
    /// `pᵢ(−υ^(2ⁱ))` — the conjugate the prover supplies. Its partner
    /// `pᵢ(υ^(2ⁱ))` is not in the proof at all: the verifier computed it as the
    /// previous fold's output, which is why a FRI layer opening is one value and
    /// not two.
    pub sym: Ext,
    /// Sibling digests, LEAF LEVEL FIRST.
    pub siblings: Vec<WrapDigest>,
}

/// What the FRI leg needs from a query the trace legs already verified.
///
/// Every field is a CELL the previous leg produced, never a fresh hint or a
/// re-derivation. [`super::sub_proof::QueryOutput`] is exactly this shape's
/// supplier.
pub struct FriQuery<'a> {
    /// `p₀(υ)` — the DEEP reconstruction at the query point.
    pub p0: Ext,
    /// `p₀(−υ)`.
    pub p0_sym: Ext,
    /// `υ`. Not Merkle-checked here and not hinted: it is the point the
    /// authenticated opening was folded at.
    pub point: Felt,
    /// `−υ`, needed only by the zero-fold shape.
    pub point_sym: Felt,
    /// The query index low-to-high, `shape.index_bits()` of them — the cells
    /// the trace walk consumed.
    pub bits: &'a [Bit],
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
    let num_zetas = if shape.total_folds() > 0 { c + 1 } else { 0 };

    let roots = b.declare_arena(2 * c as u32);
    let zetas = b.declare_arena(num_zetas as u32);
    let coeffs = b.declare_arena(shape.num_terminal_coeffs() as u32);
    let queries = b.declare_arena((num_queries * shape.query_words()) as u32);

    let layers = (0..c)
        .map(|i| LayerCommitment::hint(b, roots, 2 * i as u32))
        .collect();
    let zeta_cells = (0..num_zetas as u32)
        .map(|i| b.hint_word(zetas, i).as_ext())
        .collect();
    let coeff_cells = (0..shape.num_terminal_coeffs() as u32)
        .map(|i| b.hint_word(coeffs, i).as_ext())
        .collect();

    (
        FriArenas {
            roots,
            zetas,
            coeffs,
            queries,
        },
        FriCommitments {
            layers,
            zetas: zeta_cells,
            coeffs: coeff_cells,
        },
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
    let mut cursor = (query * shape.query_words()) as u32;
    let openings: Vec<LayerOpening> = (0..shape.num_committed())
        .map(|layer| {
            let sym = b.hint_word(arena, cursor).as_ext();
            cursor += 1;
            let siblings: Vec<WrapDigest> = (0..shape.layer_path_len(layer))
                .map(|_| {
                    let lo = b.hint_word(arena, cursor);
                    let hi = b.hint_word(arena, cursor + 1);
                    cursor += 2;
                    [lo, hi]
                })
                .collect();
            LayerOpening { sym, siblings }
        })
        .collect();
    assert_eq!(
        cursor as usize,
        (query + 1) * shape.query_words(),
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
         it needs all log2(lde) − 1 index bits"
    );
    assert_eq!(fri.layers.len(), c, "one commitment per committed layer");
    assert_eq!(openings.len(), c, "one opening per committed layer");
    assert_eq!(
        fri.coeffs.len(),
        shape.num_terminal_coeffs(),
        "the terminal polynomial carries 2^effective_k coefficients"
    );

    if shape.total_folds() == 0 {
        assert!(
            fri.zetas.is_empty(),
            "a codeword that never folds draws no folding challenge"
        );
        let at = emit_terminal_eval(b, fri, q.point);
        b.assert_eq_ext(at, q.p0);
        let at_sym = emit_terminal_eval(b, fri, q.point_sym);
        b.assert_eq_ext(at_sym, q.p0_sym);
        return q.p0;
    }
    assert_eq!(
        fri.zetas.len(),
        c + 1,
        "folds exceed committed layers by one"
    );

    // `υ⁻¹`, once. Production batch-inverts across queries and REJECTS on a
    // zero point (`verifier.rs:465`, fails closed on a malformed index); the
    // machine's `Div` errors on a zero divisor, which is the same disposition —
    // an unprovable program rather than a wrong answer.
    let one = b.felt_const(FE::one());
    let inv = b.div(one, q.point);

    // Fold 0 consumes the DEEP pair and authenticates nothing: there is no
    // layer under it, which is why `zetas` is one longer than `layers`.
    let mut v = edsl::fri_fold(b, q.p0, q.p0_sym, fri.zetas[0], inv);

    // The point chain is one squaring per layer and nothing else — no bit
    // reversal, no domain lookup, no coset offset past the first point
    // (spec §6). And no parity branch, because the sign the odd slot introduces
    // into `x⁻¹` is the same sign it introduces into `v − sym`, so the two
    // cancel (spec §3). Parity is consulted ONLY for the leaf byte order below.
    let mut inv_pow = inv;
    for (i, opening) in openings.iter().enumerate() {
        // `if index % 2 == 1 { [sym, v] } else { [v, sym] }` (`verifier.rs:637`)
        // — the even codeword slot leads. `select(bit, l, r)` returns `(l, r)`
        // at 0 and `(r, l)` at 1, so this IS that conditional.
        let (first, second) = b.select(q.bits[i], v.as_cell(), opening.sym.as_cell());
        let leaf = sub_proof::emit_leaf_hash(b, FRI_LEAF_GROUP, &[first, second]);
        let root = edsl::wrap_merkle_walk(b, leaf, &q.bits[i + 1..], &opening.siblings);
        edsl::assert_digest_eq_lanes(b, root, &fri.layers[i].root_lanes);

        // `evaluation_point_vec[i] = υ^(−2^(i+1))` — `inv.square()` then one
        // squaring per layer (`verifier.rs:692-697`).
        inv_pow = b.mul(inv_pow, inv_pow);
        v = edsl::fri_fold(b, v, opening.sym, fri.zetas[i + 1], inv_pow);
    }

    // `x = υ^(2^total_folds)`: where the fold chain has arrived, and the
    // terminal codeword's point at position `iota >> C`. See the doc comment.
    let mut x = q.point;
    for _ in 0..shape.total_folds() {
        x = b.mul(x, x);
    }
    let at = emit_terminal_eval(b, fri, x);
    b.assert_eq_ext(at, v);
    v
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
                    p0: out.deep.0,
                    p0_sym: out.deep.1,
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
