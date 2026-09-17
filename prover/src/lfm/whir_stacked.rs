//! The stacked-evaluation weight as a machine leg.
//!
//! `stacked_eval::verify` hands `whir_chain::verify_weighted` a closure — the
//! weight of one stacked polynomial at the chain's final point — and that
//! closure is `weight_at` (`crypto/multilinear/src/stacked_eval.rs:287-317`):
//!
//! ```text
//!     Σ_c  w_c · eq(corner_c, high) · eq(point_c, low)
//! ```
//!
//! where `corner_c` is the column's prefix bits and `(high, low)` splits the
//! point at the prefix length. Written naively that is two `eq`s per column per
//! chain, which the sizing note priced as the term to collapse. Two structural
//! facts collapse it.

use multilinear::stacking::StackedLayout;

use crate::tables::types::FEE;

use super::builder::{Ext, LfmBuilder};
use super::instr::Addr;
use super::whir_poly::{emit_eq_eval, eq_eval_rows_again};

/// One column's claim, as wires: the point it is claimed at and the batching
/// weight it carries (`gamma^column`).
///
/// ⚠ The weights are INPUTS. `challenge_powers` is `pub(crate)` in the
/// multilinear crate, so there is no host function to gate an emitted copy
/// against, and an ungated copy is not worth having — whoever assembles the
/// group emits the `n − 1` multiplies.
#[derive(Clone, Copy)]
pub struct ColumnClaim<'a> {
    pub point: &'a [Ext],
    pub weight: Ext,
}

/// INSTRUCTIONS [`emit_weight_at`] emits for one stacked polynomial, given
/// which columns share a claimed point.
///
/// `group_of[column]` names that point: one table's sumcheck leaves every one
/// of its columns at the same point (`stacked_eval.rs:185-188`), so in a real
/// epoch this is the table index and the group is a table's columns inside this
/// polynomial.
///
/// The three terms, each by its shape:
/// - one `Sub` per prefix position that ANY of the polynomial's columns reads as
///   a zero bit — `1 − at_j` depends on the position alone, so it is emitted
///   once and shared;
/// - per distinct claimed point, one `eq` over that group's columns' own
///   variables plus one `MulAdd` to join the group into the running total;
/// - per column, `p − 1` products for its prefix indicator plus one `MulAdd`
///   for its weighted fold, where `p = n_stack − num_vars`; a column that fills
///   the whole stack has no indicator and pays only the fold, hence `max(p, 1)`.
///
/// An empty polynomial is one interned zero.
pub fn weight_at_rows(layout: &StackedLayout, poly: usize, group_of: &[usize]) -> usize {
    let placements = layout.placements();
    let columns: Vec<usize> = (0..placements.len())
        .filter(|&column| placements[column].poly == poly)
        .collect();
    if columns.is_empty() {
        return 1;
    }

    let mut reads_a_zero = vec![false; layout.n_stack()];
    for &column in &columns {
        for (position, bit) in placements[column].prefix_bits().iter().enumerate() {
            if !bit {
                reads_a_zero[position] = true;
            }
        }
    }
    let shared_subs = reads_a_zero.iter().filter(|seen| **seen).count();

    let mut seen_groups: Vec<usize> = Vec::new();
    let mut group_rows = 0;
    for &column in &columns {
        if !seen_groups.contains(&group_of[column]) {
            seen_groups.push(group_of[column]);
            group_rows += eq_eval_rows_again(placements[column].num_vars) + 1;
        }
    }

    let column_rows: usize = columns
        .iter()
        .map(|&column| prefix_len(layout, column).max(1))
        .sum();

    shared_subs + group_rows + column_rows
}

/// `LFM_CONST` rows the leg interns, once per program: the `1`, which both
/// `eq` and a full-height column's empty indicator reach for.
pub const fn weight_at_consts() -> usize {
    1
}

/// ★ `weight_at` for one stacked polynomial, emitted.
///
/// ★ **Why this is not two `eq`s per column.** First, `corner_c` is a CONSTANT
/// zero/one vector, so its `eq` is just the prefix indicator
/// `Π_j (at_j or 1 − at_j)` — `p` factors, not a `5p`-row `eq` — and `1 − at_j`
/// depends only on the position, so every column that reads position `j` as
/// zero shares one `Sub`. Second, columns claimed at the same point share their
/// whole low-half `eq`; one table's sumcheck leaves all of its columns at one
/// point, so a real epoch pays one `eq` per (table, polynomial), not one per
/// column. Grouping is by the point's WIRES, which is exact at emit time and
/// needs no notion of a table.
///
/// The total is then `Σ_groups eq_low · Σ_{c in group} w_c · indicator_c`, which
/// also saves one multiply per column against multiplying each term out.
pub fn emit_weight_at(
    b: &mut LfmBuilder,
    layout: &StackedLayout,
    poly: usize,
    claims: &[ColumnClaim<'_>],
    at: &[Ext],
) -> Ext {
    assert_eq!(
        at.len(),
        layout.n_stack(),
        "the weight is evaluated on the stacked cube"
    );
    assert_eq!(
        claims.len(),
        layout.placements().len(),
        "one claim per column of the layout"
    );

    let placements = layout.placements();
    let columns: Vec<usize> = (0..placements.len())
        .filter(|&column| placements[column].poly == poly)
        .collect();
    if columns.is_empty() {
        return b.ext_const(&FEE::zero());
    }

    // Columns sharing a claimed point share its `eq`. The key is the point's
    // wires, in order.
    let mut groups: Vec<(Vec<Addr>, Vec<usize>)> = Vec::new();
    for &column in &columns {
        let key: Vec<Addr> = claims[column].point.iter().map(Ext::addr).collect();
        match groups.iter_mut().find(|(seen, _)| *seen == key) {
            Some((_, members)) => members.push(column),
            None => groups.push((key, vec![column])),
        }
    }

    let mut not_at: Vec<Option<Ext>> = vec![None; at.len()];
    let mut one: Option<Ext> = None;
    let mut total: Option<Ext> = None;

    for (_, members) in &groups {
        let leader = members[0];
        let prefix = prefix_len(layout, leader);
        assert_eq!(
            claims[leader].point.len(),
            placements[leader].num_vars,
            "a column is claimed on its own cube"
        );
        let eq_low = emit_eq_eval(b, claims[leader].point, &at[prefix..]);

        let mut inner: Option<Ext> = None;
        for &column in members {
            let mut indicator: Option<Ext> = None;
            for (position, bit) in placements[column].prefix_bits().iter().enumerate() {
                let factor = if *bit {
                    at[position]
                } else {
                    match not_at[position] {
                        Some(cached) => cached,
                        None => {
                            let unit = *one.get_or_insert_with(|| b.ext_const(&FEE::one()));
                            let complement = b.esub(unit, at[position]);
                            not_at[position] = Some(complement);
                            complement
                        }
                    }
                };
                indicator = Some(match indicator {
                    None => factor,
                    Some(acc) => b.emul(acc, factor),
                });
            }
            // A column filling the whole stack has an empty indicator, which is
            // the interned `1` — so the fold below is one row either way.
            let indicator =
                indicator.unwrap_or_else(|| *one.get_or_insert_with(|| b.ext_const(&FEE::one())));
            let weight = claims[column].weight;
            inner = Some(match inner {
                None => b.emul(indicator, weight),
                Some(acc) => b.emul_add(indicator, weight, acc),
            });
        }
        let inner = inner.expect("a group holds at least one column");
        total = Some(match total {
            None => b.emul(eq_low, inner),
            Some(acc) => b.emul_add(eq_low, inner, acc),
        });
    }

    total.expect("a non-empty polynomial has at least one group")
}

/// How many high variables of the stack a column's prefix selects.
fn prefix_len(layout: &StackedLayout, column: usize) -> usize {
    let place = &layout.placements()[column];
    place.n_stack - place.num_vars
}
