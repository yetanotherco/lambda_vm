//! Splitting the global wrap across programs that each verify a SLICE of its
//! tables and publish a PARTIAL bus sum.
//!
//! # Why the global wrap has to be split
//!
//! The interior tree exists because a flat aggregator over 19 epoch wraps would
//! be too big. The global wrap is a flat aggregator over the global proof's
//! tables — **the same shape of mistake, in the one place nobody sized.** It
//! aborts at `LFM_HASH` 2^21 x 329 cols, needing 26.22 GiB against a 32 GiB card:
//! byte for byte the fan-in-3 failure.
//!
//! # Why it is NOT a tree of nodes
//!
//! A node verifies a CHILD PROOF. The global proof is one `MultiProof` whose
//! tables cannot be verified in independent subsets:
//!
//! - its program closes **one bus over ALL its tables** against a zero target, so
//!   a subset does not balance;
//! - [`super::epoch::fork_table`] uses **`num_tables` as a domain separator**, so
//!   a slice believing it had five tables would derive challenges the proof was
//!   never made under.
//!
//! ⇒ Each slice replays the same Phase A over ALL main roots (an absorb, not a
//! walk — the cheap half), verifies only its own tables at their TRUE index
//! within the true `num_tables`, and publishes its partial sum. A parent sums the
//! partials, pins the partition, and asserts zero.

/// The single source of truth for how the global proof's tables are split.
///
/// ⛔ **ONE SOURCE, BECAUSE THE COMMENT CANNOT ENFORCE IT.** This campaign has
/// just spent three hours on a `DivByZero` that was a second copy of an encoding
/// width behind a plain `usize` in another module, with a comment explaining that
/// it had to match the first. `statement.rs`'s tripwire fired and was obeyed —
/// and it guarded only its own copy. **The comment knew; the compiler could
/// not.**
///
/// Every quantity here — `k`, each slice's bounds, the total — is derived from
/// `num_tables` and `k` by this type and read from it by BOTH the emitter and the
/// harness. There is no second place to write a bound, so there is nothing to
/// drift.
///
/// # The partition is the soundness argument
///
/// If the slices do not tile `0..num_tables` **exactly once**, a proof passes for
/// a weaker statement:
///
/// - omit table `y` ⇒ `Σ Pᵢ = T − c(y)`, caught by the zero-assert alone;
/// - duplicate table `x` ⇒ `Σ Pᵢ = T + c(x)`, caught alone;
/// - ⛔ **both ⇒ `Σ Pᵢ = c(x) − c(y)`, which is ZERO whenever `c(x) = c(y)`** —
///   and then `y` was never verified and every check passed.
///
/// ⇒ **The zero-assert is a check on the BUS, not on the PARTITION.** Using it to
/// detect a partition error relies on two errors not cancelling, which is a check
/// that cannot fail under the failure mode it is meant to catch. The bounds are
/// emit-time constants, pinned, with arms for a gap AND an overlap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlicePartition {
    num_tables: usize,
    bounds: Vec<(usize, usize)>,
}

impl SlicePartition {
    /// Split `num_tables` into `k` contiguous slices as evenly as possible.
    ///
    /// Contiguous and in order, so a slice's bounds are two numbers rather than a
    /// set — which is what lets the parent pin the partition with `2k` constants
    /// and check tiling by adjacency instead of by membership.
    pub fn even(num_tables: usize, k: usize) -> Self {
        assert!(num_tables >= 1, "the global proof has at least one table");
        assert!(k >= 1 && k <= num_tables, "k must be in 1..={num_tables}");
        let base = num_tables / k;
        let extra = num_tables % k;
        let mut bounds = Vec::with_capacity(k);
        let mut start = 0usize;
        for i in 0..k {
            let len = base + usize::from(i < extra);
            bounds.push((start, start + len));
            start += len;
        }
        let p = Self { num_tables, bounds };
        p.assert_tiles();
        p
    }

    /// ⛔ The invariant, checked at construction and re-checkable by a caller:
    /// the slices tile `0..num_tables` exactly once — no gap, no overlap, none
    /// empty.
    pub fn assert_tiles(&self) {
        let mut expect = 0usize;
        for (i, (lo, hi)) in self.bounds.iter().enumerate() {
            assert_eq!(
                *lo,
                expect,
                "slice {i} starts at {lo} but slice {} ended at {expect}: a gap or \
                 an overlap here lets a proof pass for a weaker statement",
                i.wrapping_sub(1)
            );
            assert!(hi > lo, "slice {i} is empty");
            expect = *hi;
        }
        assert_eq!(
            expect, self.num_tables,
            "the slices cover 0..{expect} but the proof has {} tables",
            self.num_tables
        );
    }

    pub fn k(&self) -> usize {
        self.bounds.len()
    }

    pub fn num_tables(&self) -> usize {
        self.num_tables
    }

    /// Slice `i`'s half-open table range.
    pub fn slice(&self, i: usize) -> (usize, usize) {
        self.bounds[i]
    }

    /// Every slice's bounds, in order — what the parent pins as constants.
    pub fn bounds(&self) -> &[(usize, usize)] {
        &self.bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ Every split tiles exactly, at every `k`, for every table count.
    #[test]
    fn every_partition_tiles_exactly_once() {
        for num_tables in 1usize..=40 {
            for k in 1..=num_tables {
                let p = SlicePartition::even(num_tables, k);
                assert_eq!(p.k(), k);
                let covered: usize = p.bounds().iter().map(|(lo, hi)| hi - lo).sum();
                assert_eq!(covered, num_tables, "{num_tables} tables at k={k}");
                // Sizes differ by at most one, so no slice is a straggler.
                let lens: Vec<usize> = p.bounds().iter().map(|(lo, hi)| hi - lo).collect();
                let (min, max) = (
                    *lens.iter().min().expect("k >= 1"),
                    *lens.iter().max().expect("k >= 1"),
                );
                assert!(max - min <= 1, "{num_tables} at k={k}: {lens:?}");
            }
        }
    }

    /// ⛔ A hand-built partition with a GAP must be refused.
    #[test]
    #[should_panic(expected = "a gap or")]
    fn a_gap_is_refused() {
        SlicePartition {
            num_tables: 6,
            bounds: vec![(0, 2), (3, 6)],
        }
        .assert_tiles();
    }

    /// ⛔ And one with an OVERLAP. This is the arm that matters: a gap alone and
    /// an overlap alone are each caught by the parent's zero-assert, but TOGETHER
    /// they cancel whenever the two contributions are equal — so neither may be
    /// left to arithmetic.
    #[test]
    #[should_panic(expected = "a gap or")]
    fn an_overlap_is_refused() {
        SlicePartition {
            num_tables: 6,
            bounds: vec![(0, 4), (3, 6)],
        }
        .assert_tiles();
    }

    /// ⛔ And one that stops short, leaving tables verified by nobody.
    #[test]
    #[should_panic(expected = "but the proof has")]
    fn a_short_cover_is_refused() {
        SlicePartition {
            num_tables: 6,
            bounds: vec![(0, 2), (2, 4)],
        }
        .assert_tiles();
    }
}
