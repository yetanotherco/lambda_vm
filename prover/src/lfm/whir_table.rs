//! The per-table verify as a machine leg.
//!
//! The mirror is `stark::multilinear_table::verify`
//! (`multilinear_table.rs:721-785`): the bus output absorbed, the GKR ladder,
//! the row challenges, the bus statements, the zerocheck rule, one batched
//! sumcheck over all three, and `claim_reduce` to leave every COLUMN claimed at
//! one point.
//!
//! This module starts with the one kernel none of the earlier legs has: the row
//! SELECTOR, which is what a public factor's value is.

use multilinear::selector::Selector;

use crate::tables::types::FEE;

use super::builder::{Ext, LfmBuilder};
use super::whir_bus::Cost;

/// ★ `Selector::evaluate` (`crypto/multilinear/src/selector.rs:61-100`),
/// emitted.
///
/// A constraint that reads the next row cannot apply on the last one, so it
/// carries `Selector { end_exemptions }` and the verifier needs that
/// indicator's value at the point its sumcheck settled on. The host computes it
/// in `O(num_vars)` rather than over the cube:
///
/// ```text
///     s(x) = 1 − geq(x, cutoff),      cutoff = 2^n − end_exemptions
/// ```
///
/// where `geq` is the multilinear indicator of `index(x) >= cutoff`: either `x`
/// matches `cutoff` bit for bit, or the two first differ at a position where
/// `cutoff` has a zero and `x` a one. `prefix` is the running "everything above
/// this bit matched"; VARIABLE 0 IS THE MOST SIGNIFICANT BIT.
///
/// # ⚠ Why the literal values are tracked rather than emitted
///
/// `prefix` starts at ONE and `geq` at ZERO. A row that multiplies a wire by a
/// literal one, or adds a literal zero to it, computes nothing — so this
/// carries each as `Option<Ext>`, where `None` IS the literal the host starts
/// from, and spends a row only once the value is a wire. It is not an
/// optimisation bolted onto the host's expression; it is the host's expression
/// with the identity steps left out, and the cost form below counts the same
/// cases.
///
/// Panics when there are more exemptions than rows — the host returns
/// `TooManyExemptions`, and here the two are emit-time numbers, so a mismatch
/// is a bug in the caller.
pub fn emit_selector(b: &mut LfmBuilder, selector: Selector, point: &[Ext]) -> Ext {
    let num_vars = point.len();
    let size = 1usize << num_vars;
    assert!(
        selector.end_exemptions <= size,
        "a selector cannot exempt more rows than the table has: {} of {size}",
        selector.end_exemptions
    );

    let one = b.ext_const(&FEE::one());
    if selector.is_trivial() {
        return one;
    }
    let cutoff = size - selector.end_exemptions;
    if cutoff == 0 {
        // Every row is exempt: the constraint applies nowhere.
        return b.ext_const(&FEE::zero());
    }

    // `None` is the literal the host starts from: `prefix = 1`, `geq = 0`.
    let mut prefix: Option<Ext> = None;
    let mut geq: Option<Ext> = None;
    for (i, x) in point.iter().enumerate() {
        let bit = (cutoff >> (num_vars - 1 - i)) & 1;
        if bit == 0 {
            // `x` exceeds `cutoff` here when everything above matched and
            // `x_i = 1`.
            geq = Some(match (prefix, geq) {
                (None, None) => *x,
                (None, Some(running)) => b.eadd(*x, running),
                (Some(above), None) => b.emul(above, *x),
                (Some(above), Some(running)) => b.emul_add(above, *x, running),
            });
            let one_minus = b.esub(one, *x);
            prefix = Some(match prefix {
                None => one_minus,
                Some(above) => b.emul(above, one_minus),
            });
        } else {
            prefix = Some(match prefix {
                None => *x,
                Some(above) => b.emul(above, *x),
            });
        }
    }

    // Both branches write `prefix`, and a selector that is neither trivial nor
    // all-exempt has at least one variable — `num_vars = 0` leaves `cutoff = 0`
    // for every non-trivial exemption count, which returned above.
    let prefix = prefix.expect("a live selector reads at least one variable");
    // The remaining prefix is the `x == cutoff` case.
    let total = match geq {
        None => prefix,
        Some(running) => b.eadd(running, prefix),
    };
    b.esub(one, total)
}

/// INSTRUCTIONS [`emit_selector`] emits over `num_vars` variables, added to
/// `cost`.
///
/// The same case analysis, counted rather than emitted:
///
/// - a trivial selector is the interned `1` and nothing else;
/// - an all-exempt one is the interned `0` and nothing else;
/// - otherwise, per variable, a `1 − x_i` subtract at every zero bit of the
///   cutoff, plus one row for each of `geq` and `prefix` once that value has
///   stopped being a literal;
/// - then the `geq + prefix` add, which the all-ones cutoff does not need, and
///   the final subtract from one.
pub fn selector_cost(selector: Selector, num_vars: usize, cost: &mut Cost) {
    let size = 1usize << num_vars;
    assert!(selector.end_exemptions <= size, "more exemptions than rows");

    cost.constant(FEE::one());
    if selector.is_trivial() {
        return;
    }
    let cutoff = size - selector.end_exemptions;
    if cutoff == 0 {
        cost.constant(FEE::zero());
        return;
    }

    let mut prefix_live = false;
    let mut geq_live = false;
    for i in 0..num_vars {
        let bit = (cutoff >> (num_vars - 1 - i)) & 1;
        if bit == 0 {
            if prefix_live || geq_live {
                cost.op();
            }
            geq_live = true;
            // `1 − x_i` is a row whatever the running product holds.
            cost.op();
            if prefix_live {
                cost.op();
            }
            prefix_live = true;
        } else {
            if prefix_live {
                cost.op();
            }
            prefix_live = true;
        }
    }

    if geq_live {
        cost.op();
    }
    cost.op();
}
