//! Polynomial primitives the WHIR verifier's legs share.
//!
//! `eq` is the one everything reaches for: the GKR layer relation multiplies by
//! it once per layer, the stacked-evaluation weight is a product of two of
//! them per column, and every sumcheck closes against one. So it is worth
//! emitting once, in the cheapest shape, with its cost written down.

use crate::tables::types::FEE;

use super::builder::{Ext, LfmBuilder};

/// INSTRUCTIONS [`emit_eq_eval`] emits over `n` variables.
///
/// Four rows per variable and one product to fold each into the running
/// accumulator, except the first, which *is* the accumulator. Plus the interned
/// `1`, counted the way [`super::preprocessed::bitwise_preprocessed_rows`]
/// counts it — an `LFM_CONST` row, paid once per program rather than once per
/// leg, so a second leg in the same program pays nothing for it and this number
/// is an upper bound there.
///
/// Pinned against the emitter by
/// `whir_poly_tests::the_eq_leg_emits_its_closed_form`.
pub const fn eq_eval_rows(n: usize) -> usize {
    if n == 0 {
        // `eq` over no variables is the empty product. Nothing is emitted but
        // the constant it returns.
        return 1;
    }
    let interned_one = 1;
    let per_variable = 4 * n; // r·x, 1−r, 1−x, and the MulAdd that joins them
    let fold = n - 1; // one product per variable past the first
    interned_one + per_variable + fold
}

/// ★ `eq(r, x) = Π_i [ r_i·x_i + (1−r_i)(1−x_i) ]`, emitted.
///
/// The factor is written as `(1−r)(1−x) + r·x` rather than as the sum of two
/// products, so the join is one `MulAdd` and a variable costs four rows instead
/// of five. `MulAdd` is the same one row as `Mul` on `LFM_XALU`, which is what
/// makes the shape free to choose.
///
/// Panics if the two points differ in length — the host returns an error there,
/// but a straight-line program's lengths are emit-time constants, so a mismatch
/// is a bug in the emitter rather than a condition to carry at runtime.
pub fn emit_eq_eval(b: &mut LfmBuilder, r: &[Ext], x: &[Ext]) -> Ext {
    assert_eq!(
        r.len(),
        x.len(),
        "eq's two points must have the same number of variables"
    );
    let one = b.ext_const(&FEE::one());
    let mut acc = one;
    for (i, (&ri, &xi)) in r.iter().zip(x).enumerate() {
        let rx = b.emul(ri, xi);
        let not_r = b.esub(one, ri);
        let not_x = b.esub(one, xi);
        let term = b.emul_add(not_r, not_x, rx);
        acc = if i == 0 { term } else { b.emul(acc, term) };
    }
    acc
}
