//! The DEEP leg: reconstructing the deep-composition polynomial at one query
//! point, in LFM instructions.
//!
//! This is where the values opened by a FRI query meet the out-of-domain frame
//! the [constraint leg](super::constraints) evaluates. For a query point `υ` the
//! verifier computes
//!
//! ```text
//!   DEEP(υ) = Σ_r  (Σ_c coeff[c][r]·opened[c] − oodRowSum[r]) / (υ − g^r·z)
//!           +      (Σ_j γ^{T+j}·H_j(υ) − hSumZpow) / (υ − z^P)
//! ```
//!
//! and the same at the symmetric point `−υ`, which shares every query-invariant
//! term. `crypto/stark/src/verifier.rs`'s
//! `reconstruct_deep_composition_poly_evaluation_pair` is the definition and the
//! oracle; nothing here re-derives it.
//!
//! # The coefficients are powers, so nothing has to be stored
//!
//! `replay_rounds_after_round_1` samples ONE challenge γ and expands a single
//! geometric run of `num_surviving() + num_parts` powers, handing the leading
//! `num_surviving()` to `build_trace_term_coeffs` and the rest to `gammas`. So
//! `coeff[c][r]` is `γ^p` for a position-determined `p`, and every sum here is a
//! **Horner fold** — one `MulAdd` per opened value, with no coefficient table to
//! materialize, hint or authenticate. That is the single biggest structural
//! saving in this leg, and it comes from a property of the transcript rather
//! than from anything the machine does.
//!
//! The power index runs COLUMN-MAJOR within each block: for every column, each
//! row of the block. So along a fixed row the exponent advances by the block's
//! row count, and a row folds as a Horner in `γ^stride` scaled by `γ^start`.
//! Both strides are one for every production AIR, which is exactly why the
//! stride is carried explicitly rather than assumed — see [`DeepShape::block`].
//!
//! The query POINT is an input here, not something this leg derives. Production
//! reaches it through a bit-reversal of the query index into the LDE coset
//! (`query_challenge_to_evaluation_point`), which belongs to the FRI/query leg.
//!
//! # Base openings need no conversion
//!
//! Precomputed and main-trace openings are BASE field elements; aux openings are
//! extension. Both enter the same `MulAdd`: a base word `(v, 0, 0, 0)` already
//! IS its extension embedding, so a base column costs exactly what an aux column
//! costs and no `MulBase` routing or repacking appears anywhere in this leg.
//!
//! # Reciprocals, not divisions
//!
//! Production batch-inverts the denominators and REJECTS the proof if any is
//! zero (`inplace_batch_inverse(...).ok()?`). Under the machine's `0/0 = 1`
//! convention a direct divide would instead return 1 whenever the numerator
//! vanished too — accepting exactly the malformed proof the production guard
//! exists to reject. So each denominator is inverted against the interned one,
//! which is unprovable at zero, and the quotient is a multiply.

use math::field::traits::IsFFTField;

use crate::tables::types::{FEE, GoldilocksField};

use super::builder::{Ext, Felt, LfmBuilder};

/// The compile-time shape of one sub-proof's DEEP reconstruction.
///
/// Every field is program SHAPE: it fixes how many `MulAdd` rows a query costs
/// and which columns are folded. A machine that read any of it from an arena
/// would let the prover choose the sum it is checked against.
#[derive(Clone, Debug)]
pub struct DeepShape {
    /// `AIR::step_size`. Rows below this open every column; rows at or above it
    /// open only the transition window.
    pub step_size: usize,
    /// `num_transition_offsets · step_size` — rows in the full OOD grid.
    pub num_eval_points: usize,
    /// Full `[main | aux]` trace width, precomputed columns included.
    pub num_total_cols: usize,
    /// The transition-window columns, sorted — the only ones a next row opens.
    pub next_row_cols: Vec<usize>,
    /// Composition-polynomial parts.
    pub num_composition_parts: usize,
    /// `log2` of the trace length, for `g^r` and `z^P`.
    pub log2_trace_length: u32,
}

impl DeepShape {
    /// Terms the trace-term coefficient run covers — `OodLayout::num_surviving`.
    /// Also the exponent γ is raised to for the first composition gamma.
    pub fn num_surviving(&self) -> usize {
        let next_rows = self.num_eval_points - self.step_size;
        self.num_total_cols * self.step_size + self.next_row_cols.len() * next_rows
    }

    /// Row `r`'s coefficient run: which columns it opens, the γ exponent of its
    /// first term, and the STRIDE between consecutive terms' exponents.
    ///
    /// The stride is not always one. `build_pruned_trace_term_coeffs` walks
    /// column-major within each block — for each column, every row of the block
    /// — so along a fixed row the exponent advances by the block's row count,
    /// not by one. Every production AIR has `step_size = 1` and a single next
    /// row, which collapses both strides to one; folding a row as a plain Horner
    /// in γ would therefore pass every test we have and be wrong for the first
    /// AIR that widened a step. Carrying the stride costs one extra power per
    /// distinct value and removes the assumption.
    pub(crate) fn block(&self, row: usize) -> (Vec<usize>, usize, usize) {
        let next_rows = self.num_eval_points - self.step_size;
        if row < self.step_size {
            ((0..self.num_total_cols).collect(), row, self.step_size)
        } else {
            let start = self.num_total_cols * self.step_size + (row - self.step_size);
            (self.next_row_cols.clone(), start, next_rows)
        }
    }

    /// [`Self::block`], exposed so the differential can check the emitter's
    /// exponent formula against the verifier's own coefficient table.
    #[cfg(test)]
    pub fn block_for_test(&self, row: usize) -> (Vec<usize>, usize, usize) {
        self.block(row)
    }
}

/// Terms shared by every query of one sub-proof, and by a query's two points.
///
/// Hoisting these is not an optimization the machine invents: production hoists
/// exactly this set into `QueryInvariantDeepTerms`, for the same reason (they do
/// not depend on the query index). Emitting them once per sub-proof rather than
/// once per query is what keeps a 219-query proof affordable.
pub struct DeepInvariants {
    /// `Σ_c coeff[c][r]·oodFull[r][c]`, one per OOD row.
    pub ood_row_sum: Vec<Ext>,
    /// `Σ_j γ^{T+j}·H_j(z^P)` over the claimed composition parts.
    pub h_sum_zpow: Ext,
    /// `z^P`.
    pub z_pow: Ext,
    /// `g^r·z`, one per OOD row.
    pub row_points: Vec<Ext>,
    /// `γ^T`, the exponent the composition gammas start at.
    pub gamma_pow_surviving: Ext,
    /// `γ^{start}` for each OOD row's block, so a block folds relative to its
    /// own start and is scaled once.
    pub gamma_pow_block: Vec<Ext>,
    /// `γ^{stride}` for each OOD row's block — the base its Horner runs in.
    pub gamma_stride: Vec<Ext>,
}

/// `Σ_k terms[k]·γ^k`, terms low-to-high — one `MulAdd` per term after the
/// first, and no power table.
fn horner(b: &mut LfmBuilder, gamma: Ext, terms: &[Ext]) -> Ext {
    let mut iter = terms.iter().rev();
    let mut acc = *iter.next().expect("a fold needs at least one term");
    for t in iter {
        acc = b.emul_add(acc, gamma, *t);
    }
    acc
}

/// `x^n` for a compile-time `n`, square-and-multiply. `n` is shape, so the row
/// count is program text rather than data-dependent.
fn pow_const(b: &mut LfmBuilder, x: Ext, n: usize) -> Ext {
    assert!(n > 0, "pow_const is not defined at zero here");
    let mut result: Option<Ext> = None;
    let mut base = x;
    let mut bits = n;
    while bits > 0 {
        if bits & 1 == 1 {
            result = Some(match result {
                None => base,
                Some(acc) => b.emul(acc, base),
            });
        }
        bits >>= 1;
        if bits > 0 {
            base = b.emul(base, base);
        }
    }
    result.expect("n > 0 sets at least one bit")
}

/// Emit the per-sub-proof terms every query reuses.
///
/// `ood_steps[r]` is the reconstructed OOD grid's row `r` — the same full-width
/// `[main | aux]` row the constraint leg reads, with pruned next-row entries
/// already the pooled zero. Those zeros are load-bearing here too: the verifier
/// pairs them with zero COEFFICIENTS, so folding the window alone is exact, and
/// a machine that hinted values into pruned slots would compute a different sum.
pub fn emit_deep_invariants(
    b: &mut LfmBuilder,
    shape: &DeepShape,
    gamma: Ext,
    zeta: Ext,
    ood_steps: &[Vec<Ext>],
    claimed_parts: &[Ext],
) -> DeepInvariants {
    assert_eq!(
        ood_steps.len(),
        shape.num_eval_points,
        "the OOD grid must have one row per evaluation point"
    );
    assert_eq!(
        claimed_parts.len(),
        shape.num_composition_parts,
        "the part count is shape and is never read off the proof"
    );

    let generator = <GoldilocksField as IsFFTField>::get_primitive_root_of_unity(
        shape.log2_trace_length as u64,
    )
    .expect("a power-of-two trace length has a root of unity");

    // Block scalars: γ^start for each row's coefficient run.
    let mut gamma_pow_block = Vec::with_capacity(shape.num_eval_points);
    let mut gamma_stride = Vec::with_capacity(shape.num_eval_points);
    let mut ood_row_sum = Vec::with_capacity(shape.num_eval_points);
    let mut row_points = Vec::with_capacity(shape.num_eval_points);

    #[allow(clippy::needless_range_loop)] // `row` indexes three parallel vectors, not one
    for row in 0..shape.num_eval_points {
        let (cols, start, stride) = shape.block(row);
        let scale = if start == 0 {
            None
        } else {
            Some(pow_const(b, gamma, start))
        };
        let base = if stride == 1 {
            gamma
        } else {
            pow_const(b, gamma, stride)
        };
        let terms: Vec<Ext> = cols.iter().map(|&c| ood_steps[row][c]).collect();
        let folded = horner(b, base, &terms);
        ood_row_sum.push(match scale {
            None => folded,
            Some(s) => b.emul(folded, s),
        });
        gamma_pow_block.push(scale.unwrap_or_else(|| b.ext_const(&FEE::one())));
        gamma_stride.push(base);

        // g^r·z. `g^r` is a program constant, so this is one MulBase per row
        // rather than a chain of multiplies whose length is data.
        let g_r = generator.pow(row as u64);
        row_points.push(if row == 0 {
            zeta
        } else {
            let c = b.felt_const(g_r);
            b.emul_base(zeta, c)
        });
    }

    let gamma_pow_surviving = pow_const(b, gamma, shape.num_surviving());
    let folded_parts = horner(b, gamma, claimed_parts);
    let h_sum_zpow = b.emul(folded_parts, gamma_pow_surviving);
    let z_pow = pow_const(b, zeta, shape.num_composition_parts);

    DeepInvariants {
        ood_row_sum,
        h_sum_zpow,
        z_pow,
        row_points,
        gamma_pow_surviving,
        gamma_pow_block,
        gamma_stride,
    }
}

/// One query point's opened values, in the order the proof carries them.
///
/// `trace` is the concatenation `precomputed ‖ main ‖ aux` — the same order
/// `reconstruct_deep_composition_poly_evaluation_pair`'s `base_at` walks, with
/// base and aux openings alike presented as extension cells because a base word
/// is already one.
pub struct DeepOpening {
    /// The query's domain point, a base-field element.
    pub point: Felt,
    /// `precomputed ‖ main ‖ aux`, one cell per full-width column.
    pub trace: Vec<Ext>,
    /// The composition parts opened at this point.
    pub parts: Vec<Ext>,
}

/// Emit `DEEP(υ)` for one query point.
///
/// Returns the reconstructed value; the caller feeds it to the FRI leg, which
/// is where it is finally checked. Cost is `num_surviving() + num_parts`
/// `MulAdd` rows plus a small constant per row, so this is the leg that scales
/// with query count and trace width.
pub fn emit_deep_point(
    b: &mut LfmBuilder,
    shape: &DeepShape,
    gamma: Ext,
    inv: &DeepInvariants,
    opening: &DeepOpening,
) -> Ext {
    assert_eq!(
        opening.trace.len(),
        shape.num_total_cols,
        "an opening covers every trace column"
    );
    assert_eq!(opening.parts.len(), shape.num_composition_parts);

    let one = b.ext_const(&FEE::one());
    let point = opening.point.as_ext();

    let mut trace_term: Option<Ext> = None;
    for row in 0..shape.num_eval_points {
        let (cols, start, _) = shape.block(row);
        let terms: Vec<Ext> = cols.iter().map(|&c| opening.trace[c]).collect();
        let folded = horner(b, inv.gamma_stride[row], &terms);
        let scaled = if start == 0 {
            folded
        } else {
            b.emul(folded, inv.gamma_pow_block[row])
        };
        let numerator = b.esub(scaled, inv.ood_row_sum[row]);
        // υ − g^r·z, inverted against one so a vanishing denominator is
        // unprovable rather than silently 0/0 = 1.
        let denominator = b.esub(point, inv.row_points[row]);
        let den_inv = b.ediv(one, denominator);
        trace_term = Some(match trace_term {
            None => b.emul(numerator, den_inv),
            Some(acc) => b.emul_add(numerator, den_inv, acc),
        });
    }

    let folded_parts = horner(b, gamma, &opening.parts);
    let h_sum = b.emul(folded_parts, inv.gamma_pow_surviving);
    let h_numerator = b.esub(h_sum, inv.h_sum_zpow);
    let h_denominator = b.esub(point, inv.z_pow);
    let h_den_inv = b.ediv(one, h_denominator);

    let trace_term = trace_term.expect("at least one evaluation point");
    b.emul_add(h_numerator, h_den_inv, trace_term)
}

/// The two points of one query — the domain point and its symmetric partner —
/// sharing every invariant.
///
/// `υ_sym = −υ`, which is why the pair costs no extra invariants: production
/// splits the term as `denom·(coeff·base − coeff·ood)` precisely so the OOD walk
/// and the coefficient run are done once for both.
pub fn emit_deep_query(
    b: &mut LfmBuilder,
    shape: &DeepShape,
    gamma: Ext,
    inv: &DeepInvariants,
    regular: &DeepOpening,
    symmetric: &DeepOpening,
) -> (Ext, Ext) {
    (
        emit_deep_point(b, shape, gamma, inv, regular),
        emit_deep_point(b, shape, gamma, inv, symmetric),
    )
}
