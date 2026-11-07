//! The LogUp closure: `Σ_tables L = expected_bus_balance`.
//!
//! Every other leg verifies one sub-proof. This is the only one that is about
//! the epoch as a whole: each table exposes the total of its LogUp terms, `L`,
//! and the bus balances when those totals sum to the target. Production's check
//! is `verifier.rs`'s final block — `Σ L over tables with trace interactions`,
//! compared against an `expected_bus_balance` the caller supplies.
//!
//! # The target is computed, not zero
//!
//! It would be zero if every bus participant were an in-trace table. One is
//! not: the COMMIT output bus has a receiver the verifier computes rather than
//! proves, so the target is that missing positive remainder
//! (`lib.rs`'s `compute_commit_bus_offset`):
//!
//! ```text
//!   expected = Σ_i  1 / (z − (BusId::Commit + (start + i)·α + byte_i·α²))
//! ```
//!
//! over the public output BYTES. So half this leg is a per-byte gadget over the
//! epoch's public output, not a comparison against a constant.
//!
//! # Reciprocals, and why the machine's division convention matters here
//!
//! Production batch-inverts the fingerprints and REJECTS on a zero divisor —
//! `inplace_batch_inverse(...).ok()?`, a fingerprint collision. The machine's
//! `x/0` is an error and `0/0` is one, so `1/fingerprint` is unprovable at a
//! collision and provable everywhere else: the convention already matches, but
//! only because the numerator is the constant one. The DEEP leg had to invert
//! against an interned one for exactly this reason and a direct divide would
//! have accepted what production rejects; the same care applies here.
//!
//! # What is shape and what is data
//!
//! Which tables carry a bus contribution is `AIR::has_trace_interaction()` —
//! AIR shape, so a program constant. The number of public output bytes is shape
//! too, because it fixes the gadget's length; the byte VALUES are data. `start`
//! is the carried commit index, data.

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, Felt, LfmBuilder};

/// The compile-time shape of one epoch's LogUp closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpShape {
    /// Sub-proofs whose `L` enters the sum — those with trace interactions.
    /// SHAPE: a program that read this off the proof would let the prover
    /// choose which tables are on the bus.
    pub num_contributing_tables: usize,
    /// Public output bytes the COMMIT-bus target folds over.
    pub num_output_bytes: usize,
}

/// The `BusId::Commit` discriminant, as the fingerprint's constant term.
///
/// Mirrored rather than imported so this module does not depend on the VM's bus
/// enum; [`bus_id_matches_production`] pins the two together.
pub const COMMIT_BUS_ID: u64 = crate::tables::types::BusId::Commit as u64;

/// The COMMIT-bus target: `Σ_i 1/(z − (busId + (start+i)·α + byte_i·α²))`.
///
/// `bytes` are the public output bytes as base cells, one byte per cell, in
/// order — the same order `compute_commit_bus_offset` enumerates them. `start`
/// is the carried commit index (`x254`): zero for a monolithic proof or a first
/// epoch, nonzero for an epoch continuing a prior one.
///
/// The index `start + i` is derived by ADDING ONE per byte rather than by
/// hinting each index, so a prover cannot renumber the output: `i` is position,
/// and position is program text.
pub fn emit_commit_bus_target(
    b: &mut LfmBuilder,
    shape: &LogUpShape,
    z: Ext,
    alpha: Ext,
    start: Felt,
    bytes: &[Felt],
) -> Ext {
    assert_eq!(
        bytes.len(),
        shape.num_output_bytes,
        "the output length is shape and fixes the gadget's size"
    );
    if shape.num_output_bytes == 0 {
        // `compute_commit_bus_offset` short-circuits to zero on empty output.
        return b.ext_const(&FEE::zero());
    }

    let one = b.ext_const(&FEE::one());
    let bus_id = b.ext_const(&FEE::from(COMMIT_BUS_ID));
    let alpha_sq = b.emul(alpha, alpha);
    let one_base = b.felt_const(FE::one());

    let mut acc: Option<Ext> = None;
    let mut index = start;
    for (i, byte) in bytes.iter().enumerate() {
        // linear = busId + index·α + byte·α².
        let index_term = b.emul_base(alpha, index);
        let byte_term = b.emul_base(alpha_sq, *byte);
        let linear = b.eadd(bus_id, index_term);
        let linear = b.eadd(linear, byte_term);
        let fingerprint = b.esub(z, linear);
        // Inverted against the interned one: a collision is `1/0`, which is
        // unprovable, matching production's rejection. A direct divide of a
        // vanishing numerator would instead give `0/0 = 1`.
        let term = b.ediv(one, fingerprint);
        acc = Some(match acc {
            None => term,
            Some(a) => b.eadd(a, term),
        });
        if i + 1 < bytes.len() {
            index = b.add(index, one_base);
        }
    }
    acc.expect("a nonempty output folds at least one term")
}

/// The closure: sum the per-table contributions and assert the bus balances.
///
/// `contributions` are the `L` cells — and they must be the SAME cells the
/// constraint leg divided by `N` to get its per-row offset (see
/// [`super::constraints::emit_table_offset`]). A program that hinted `L` here
/// and `L/N` there would let the prover pick both, and this assert would be a
/// statement about numbers bound to no trace.
///
/// Returns the published sum, so a verifier sees what balanced rather than only
/// that something did.
pub fn emit_bus_closure(
    b: &mut LfmBuilder,
    shape: &LogUpShape,
    contributions: &[Ext],
    target: Ext,
) -> Ext {
    assert_eq!(
        contributions.len(),
        shape.num_contributing_tables,
        "the contributing-table count is shape and is never read off the proof"
    );
    let mut total = match contributions.first() {
        Some(first) => *first,
        // No table carries a bus interaction: production skips the check
        // entirely, so the honest total is zero and the target must be too.
        None => b.ext_const(&FEE::zero()),
    };
    for c in contributions.iter().skip(1) {
        total = b.eadd(total, *c);
    }
    b.assert_eq_ext(total, target);
    total
}
