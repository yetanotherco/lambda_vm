//! `check_dense_index_set` — the release-visible guard against a constraint
//! body emitting one index twice and another never.
//!
//! These tests are the guard's own honest control. A checker that never fires
//! would pass every "the real chip is fine" assertion in the workspace, so the
//! first thing established here is that it fires — on the exact shape that
//! motivated it (`COMMIT.md` §1.4.4 H1: a widened lane loop overrunning into
//! the pins that follow it) and on the degenerate cases either side.

use crate::constraints::builder::{
    ConstraintBuilder, ConstraintMeta, ConstraintSet, RootKind, check_dense_index_set,
};
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

fn base(constraint_idx: usize) -> ConstraintMeta {
    ConstraintMeta {
        constraint_idx,
        kind: RootKind::Base,
        end_exemptions: 0,
    }
}

/// A body shaped like the hazard: `lanes` lane identities starting at 6,
/// then 8 unused-output pins starting at `pin_base`, then one tail constraint.
/// At `lanes = 8, pin_base = 14` the blocks abut exactly; widening `lanes` to
/// 12 without moving `pin_base` makes 14..17 collide.
struct LaneBody {
    lanes: usize,
    pin_base: usize,
    tail: usize,
}

impl ConstraintSet<F, E> for LaneBody {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        for lane in 0..self.lanes {
            b.emit_base(6 + lane, b.zero());
        }
        for j in 0..8 {
            b.emit_base(self.pin_base + j, b.zero());
        }
        for i in 0..6 {
            b.emit_base(i, b.zero());
        }
        b.emit_base(self.tail, b.zero());
    }
}

/// ★ The H1 shape: widening the lane block over the pins that follow it.
///
/// The count is unchanged — the body still emits 8+8+6+1 = 23 constraints into
/// 23 declared slots — which is exactly why `NUM_CONSTRAINTS`, a predicted-count
/// test, and `assert_complete` all miss it. Four lane identities are silently
/// overwritten and nothing else notices.
#[test]
fn the_widened_lane_block_collides_and_the_checker_says_so() {
    let healthy = LaneBody {
        lanes: 8,
        pin_base: 14,
        tail: 22,
    };
    check_dense_index_set(&healthy.meta(), 23).expect("the un-widened body is dense");

    let widened = LaneBody {
        lanes: 12,
        pin_base: 14,
        tail: 22,
    };
    // Same declared count — the collision is invisible to counting.
    assert_eq!(widened.meta().len(), 27);
    let err = check_dense_index_set(&widened.meta(), 27)
        .expect_err("a lane block overrunning the pins must be caught");
    assert!(
        err.contains("emitted twice [14, 15, 16, 17]"),
        "the four colliding indices must be named, got: {err}"
    );
    assert!(
        err.contains("never emitted [23, 24, 25, 26]"),
        "the slots left unwritten must be named, got: {err}"
    );
}

/// A repeat with no compensating gap is still a repeat.
#[test]
fn a_plain_duplicate_is_caught() {
    let meta = vec![base(0), base(1), base(1), base(2)];
    let err = check_dense_index_set(&meta, 4).expect_err("1 emitted twice");
    assert!(err.contains("emitted twice [1]"), "got: {err}");
    assert!(err.contains("never emitted [3]"), "got: {err}");
}

/// A hole with no compensating duplicate cannot keep the count, so it surfaces
/// as a count mismatch — the half `assert_complete` used to cover. Worth
/// pinning: it is the reason a gap alone is the *easy* failure, and why H1
/// (which pairs a gap with a duplicate and so keeps the count) is the hard one.
#[test]
fn a_gap_without_a_duplicate_shows_up_as_a_count_mismatch() {
    let meta = vec![base(0), base(2), base(3), base(4)];
    let err = check_dense_index_set(&meta, 5).expect_err("1 never emitted");
    assert!(
        err.contains("emitted 4 constraints, declared 5"),
        "got: {err}"
    );
}

/// Wrong total is reported as wrong total, not as a confusing index list.
#[test]
fn a_count_mismatch_is_reported_plainly() {
    let meta = vec![base(0), base(1)];
    let err = check_dense_index_set(&meta, 3).expect_err("2 != 3");
    assert!(
        err.contains("emitted 2 constraints, declared 3"),
        "got: {err}"
    );
}

/// The honest control: a dense set passes, including the empty one.
#[test]
fn a_dense_set_passes() {
    check_dense_index_set(&[], 0).expect("the empty body is dense");
    let meta: Vec<_> = (0..64).map(base).collect();
    check_dense_index_set(&meta, 64).expect("0..64 with no repeats is dense");
}
