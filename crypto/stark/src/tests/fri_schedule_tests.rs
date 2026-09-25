//! Tests for the FRI fold schedule (`crate::fri::schedule`) and the generalised
//! `FriFoldLayout` (FRI.md §10 U1–U3).
//!
//! Two objectives appear here. The PRODUCTION one is the cost law (RULINGS 13,
//! `FRI_COST_WEIGHTS`): U1 pins its schedules as the Rust DP computes them, U2
//! checks it against brute force. The design model's PERMUTATION objective
//! (FRI.md §2.1, the §2.2 table) is kept as a second instance of the generic DP
//! (`fri_schedule_by`), pinned against the design document: it shows the DP
//! machinery reproduces an independent model exactly, and documents how far
//! the two objectives' schedules differ.

use crate::fri::schedule::{
    BALU_ROW_NS, FRI_COST_WEIGHTS, FRI_FOLD_XALU_ROWS, FRI_SCHEDULE_DMAX, FriFormat,
    FriFormatError, XALU_ROW_NS, fri_chain_start, fri_group_layer_rows, fri_layer_cost_q,
    fri_leaf_blocks, fri_pair_layer_cost_q, fri_pair_layer_rows, fri_schedule, fri_schedule_by,
    fri_schedule_cost_by, fri_schedule_cost_q, fri_schedule_with_cost, legacy_fri_schedule,
};
use crate::fri::terminal::FriFoldLayout;
use crate::proof::options::{
    CapPolicy, FriMode, FriScheduleOverride, OneRowMode, ProofFormat, ProofOptions,
};
use crypto::merkle_tree::cap::{AUTO_WEIGHTS, cap_gain};

const Q: u64 = 110;

// ---------------------------------------------------------------------------
// Cap-height functions for the permutation objective.
// ---------------------------------------------------------------------------

/// A cap-height function that caps nothing.
fn no_cap(_depth: u32) -> u32 {
    0
}

/// The cap rule FRI.md §2.2's table was computed with (PLAN §4):
/// `c = argmax_{0 ≤ c ≤ depth} (Q·c − (2^c − 1))`, ties to the smaller `c`.
fn cap_design_model(depth: u32) -> u32 {
    let (mut best, mut best_c) = (0i64, 0u32);
    for c in 0..=depth.min(62) {
        let v = Q as i64 * i64::from(c) - ((1i64 << c) - 1);
        if v > best {
            (best, best_c) = (v, c);
        }
    }
    best_c
}

/// The adopted policy (RULINGS 1): every FRI tree is opened once per query.
fn cap_auto(depth: u32) -> u32 {
    CapPolicy::Auto.height(Q as usize, depth as usize) as u32
}

#[test]
fn cap_auto_heights_match_cap_md() {
    // CAP.md §11 "CapPolicy pins", at a depth large enough not to clamp.
    for (openings, want) in [(1, 0), (3, 0), (4, 2), (19, 2), (20, 3), (110, 3), (224, 3)] {
        assert_eq!(
            CapPolicy::Auto.height(openings, 20),
            want,
            "openings {openings}"
        );
    }
    // Clamped to depth.
    for depth in 0..8 {
        assert_eq!(cap_auto(depth), depth.min(3), "depth {depth}");
    }
    // The design model's rule reaches 7 at Q = 110 (FRI.md §2.2 used it).
    assert_eq!(cap_design_model(20), 7);
    assert_eq!(cap_design_model(5), 5);
}

/// FRI.md §2.1's per-layer cost, `Q ×` permutations: `Q·leaf(d) + Q·(depth −
/// c) + 2^c − 1`.
fn perm_layer_q(d: u32, depth: u32, q: u64, cap: &dyn Fn(u32) -> u32) -> u64 {
    let c = cap(depth).min(depth);
    q * fri_leaf_blocks(d) + q * u64::from(depth - c) + (1u64 << c) - 1
}

fn perm_schedule(
    b0: u32,
    t: u32,
    cap: &dyn Fn(u32) -> u32,
) -> crate::fri::schedule::FriScheduleChoice {
    fri_schedule_by(b0, t, FRI_SCHEDULE_DMAX, &|d, depth| {
        perm_layer_q(d, depth, Q, cap)
    })
}

fn perm_cost(b0: u32, schedule: &[u8], cap: &dyn Fn(u32) -> u32) -> Option<u64> {
    fri_schedule_cost_by(b0, schedule, &|d, depth| perm_layer_q(d, depth, Q, cap))
}

// ---------------------------------------------------------------------------
// Cost-model primitives.
// ---------------------------------------------------------------------------

#[test]
fn leaf_blocks() {
    // ⌈3·2^d / 8⌉, at least 1.
    let want = [1u64, 1, 2, 3, 6, 12, 24];
    for (d, w) in want.iter().enumerate() {
        assert_eq!(fri_leaf_blocks(d as u32), *w, "d = {d}");
    }
}

/// The objective's weights are a format constant (RULINGS 13, 22): the cap
/// policy's weights plus the in-guest XALU and BALU row prices (a fold is 5
/// XALU rows, a twiddle one BALU row).
#[test]
fn cost_weights_are_pinned() {
    assert_eq!(FRI_COST_WEIGHTS.cap, AUTO_WEIGHTS);
    assert_eq!(
        (
            FRI_COST_WEIGHTS.cap.compress,
            FRI_COST_WEIGHTS.cap.select,
            FRI_COST_WEIGHTS.cap.unpack,
            FRI_COST_WEIGHTS.cap.hint,
            FRI_COST_WEIGHTS.cap.compare
        ),
        (2251, 567, 528, 460, 3789)
    );
    assert_eq!(
        (FRI_FOLD_XALU_ROWS, XALU_ROW_NS, BALU_ROW_NS),
        (5, 522, 477)
    );
    assert_eq!(
        (FRI_COST_WEIGHTS.fold, FRI_COST_WEIGHTS.twiddle),
        (2610, 477)
    );
    assert_eq!(
        (FRI_COST_WEIGHTS.xalu, FRI_COST_WEIGHTS.balu),
        (XALU_ROW_NS, BALU_ROW_NS)
    );
    assert_eq!(
        FRI_COST_WEIGHTS.fold,
        FRI_FOLD_XALU_ROWS * FRI_COST_WEIGHTS.xalu
    );
    assert_eq!(FRI_COST_WEIGHTS.twiddle, FRI_COST_WEIGHTS.balu);
    // The node cost law, 421 ns/instruction + 5.63 ns/cell, at the committed
    // widths (XALU 18, BALU 10 cells), rounded to the nearest ns.
    assert_eq!(((421.0f64 + 5.63 * 18.0).round()) as u64, XALU_ROW_NS);
    assert_eq!(((421.0f64 + 5.63 * 10.0).round()) as u64, BALU_ROW_NS);
}

/// One layer's cost written out by hand (RULINGS 22: every emitted row).
#[test]
fn layer_cost_by_hand() {
    // d = 3, depth 10, no cap:
    //   leaf 3·2251 + walk 10·(2251+567) + slot mux 7·567 + folds 7·2610
    //   + twiddles 3·477 + x_g 3·(567+477) + scaling 1·522 + 1·477
    //   + slot assert 2·522 + compare 8·477 + 528 + values 8·(528+460)
    //   + leaf packs 6·528 + siblings 10·460.
    let per_query = 3 * 2251
        + 10 * (2251 + 567)
        + 7 * 567
        + 7 * 2610
        + 3 * 477
        + 3 * (567 + 477)
        + 522
        + 477
        + 2 * 522
        + 8 * 477
        + 528
        + 8 * (528 + 460)
        + 6 * 528
        + 10 * 460;
    assert_eq!(
        fri_layer_cost_q(&FRI_COST_WEIGHTS, 3, 10, Q, CapPolicy::Off),
        Q * per_query
    );
    // Auto cap at Q = 110 is c = 3 on a 10-deep tree: minus its gain and the
    // three sibling hints per query the capped path omits.
    let gain = cap_gain(&AUTO_WEIGHTS, 110, 3);
    assert!(gain > 0);
    assert_eq!(
        fri_layer_cost_q(&FRI_COST_WEIGHTS, 3, 10, Q, CapPolicy::Auto),
        Q * per_query - gain as u64 - Q * 3 * 460
    );
    // d = 1 at depth 0 under the group encoding: leaf, mux select, fold,
    // twiddle, x_g select + mul, slot assert, compare, 2 values, 2 packs.
    assert_eq!(
        fri_layer_cost_q(&FRI_COST_WEIGHTS, 1, 0, 1, CapPolicy::Off),
        2251 + 567
            + 2610
            + 477
            + (567 + 477)
            + 2 * 522
            + (8 * 477 + 528)
            + 2 * (528 + 460)
            + 2 * 528
    );
    // Today's pair layer at depth 0: parity select, leaf, fold, squaring,
    // compare, two unpacks, two packs and one hinted sibling value.
    assert_eq!(
        fri_pair_layer_cost_q(&FRI_COST_WEIGHTS, 0, 1, CapPolicy::Off),
        567 + 2251 + 2610 + 477 + (8 * 477 + 528) + 2 * 528 + 2 * 528 + 460
    );
}

/// The row model's kinds at `d = 1..=6`, written out (the in-guest lane pins
/// the same numbers against the emitter, `lfm::fri_group_tests`).
#[test]
fn group_layer_rows_by_hand() {
    // (d, selects, XALU, BALU, hashes, unpacks, packs, hints) at depth 2,
    // uncapped.
    let want = [
        (1u32, 4u64, 7u64, 10u64, 3u64, 3u64, 2u64, 4u64),
        (2, 7, 17, 13, 4, 5, 3, 6),
        (3, 12, 38, 15, 5, 9, 6, 10),
        (4, 21, 79, 17, 8, 17, 12, 18),
        (5, 38, 160, 19, 14, 33, 24, 34),
        (6, 71, 321, 21, 26, 65, 48, 66),
    ];
    for (d, sel, xalu, balu, hashes, unpacks, packs, hints) in want {
        let r = fri_group_layer_rows(d, 2, 0);
        assert_eq!(
            (
                r.selects, r.xalu, r.balu, r.hashes, r.unpacks, r.packs, r.hints
            ),
            (sel, xalu, balu, hashes, unpacks, packs, hints),
            "d = {d}"
        );
    }
    // A cap of c on the same tree: c fewer walk levels and sibling hints,
    // 2^c − 1 cap-mux selects, one more unpack.
    let r = fri_group_layer_rows(3, 2, 2);
    assert_eq!(
        (r.selects, r.hashes, r.unpacks, r.hints),
        (12 - 2 + 3, 5 - 2, 9 + 1, 10 - 2)
    );
    // The cap height is clamped to the depth.
    assert_eq!(fri_group_layer_rows(3, 2, 9), fri_group_layer_rows(3, 2, 2));
    let p = fri_pair_layer_rows(2, 0);
    assert_eq!(
        (
            p.selects, p.xalu, p.balu, p.hashes, p.unpacks, p.packs, p.hints
        ),
        (3, 5, 9, 3, 3, 2, 3)
    );
}

#[test]
fn schedule_cost_rejects_malformed_schedules() {
    let cap = CapPolicy::Off;
    assert_eq!(fri_schedule_cost_q(10, &[], Q, cap), Some(0));
    assert_eq!(fri_schedule_cost_q(10, &[0, 1], Q, cap), None);
    assert_eq!(fri_schedule_cost_q(3, &[2, 2], Q, cap), None);
    assert!(fri_schedule_cost_q(4, &[2, 2], Q, cap).is_some());
}

// ---------------------------------------------------------------------------
// The design model (permutation objective): the FRI.md §2.2 table, reproduced.
// ---------------------------------------------------------------------------

/// (B, today, S3 from B−1, S2+S3 from B); each entry = (cost·Q, schedule).
/// Generated by an independent Python reproduction of FRI.md §2.1 in exact
/// integer units (lane I-FRI-H scratch), and cross-checked against
/// `lanes/D-FRI/model_output.txt` for the OFF and MODEL caps (cost / 110).
type Row = (
    u32,
    (u64, &'static [u8]),
    (u64, &'static [u8]),
    (u64, &'static [u8]),
);

/// T = 9, cap = OFF: (B, today, S3, S2+S3), each (cost·Q, schedule).
const PIN_T9_CAP_OFF: &[Row] = &[
    (6, (0, &[]), (0, &[]), (0, &[])),
    (7, (0, &[]), (0, &[]), (0, &[])),
    (8, (0, &[]), (0, &[]), (0, &[])),
    (9, (0, &[]), (0, &[]), (0, &[])),
    (10, (0, &[]), (0, &[]), (1100, &[1])),
    (11, (1100, &[1]), (1100, &[1]), (1210, &[2])),
    (12, (2310, &[1, 1]), (1210, &[2]), (1320, &[3])),
    (13, (3630, &[1, 1, 1]), (1320, &[3]), (1650, &[4])),
    (14, (5060, &[1, 1, 1, 1]), (1650, &[4]), (2310, &[5])),
    (15, (6600, &[1, 1, 1, 1, 1]), (2310, &[5]), (2970, &[3, 3])),
    (
        16,
        (8250, &[1, 1, 1, 1, 1, 1]),
        (2970, &[3, 3]),
        (3300, &[4, 3]),
    ),
    (
        17,
        (10010, &[1, 1, 1, 1, 1, 1, 1]),
        (3300, &[4, 3]),
        (3740, &[4, 4]),
    ),
    (
        18,
        (11880, &[1, 1, 1, 1, 1, 1, 1, 1]),
        (3740, &[4, 4]),
        (4400, &[5, 4]),
    ),
    (
        19,
        (13860, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4400, &[5, 4]),
        (5170, &[5, 5]),
    ),
    (
        20,
        (15950, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5170, &[5, 5]),
        (5720, &[4, 4, 3]),
    ),
    (
        21,
        (18150, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5720, &[4, 4, 3]),
        (6270, &[4, 4, 4]),
    ),
    (
        22,
        (20460, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (6270, &[4, 4, 4]),
        (6930, &[5, 4, 4]),
    ),
    (
        23,
        (22880, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (6930, &[5, 4, 4]),
        (7700, &[5, 5, 4]),
    ),
    (
        24,
        (25410, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (7700, &[5, 5, 4]),
        (8580, &[5, 5, 5]),
    ),
];
/// T = 10, cap = OFF: (B, today, S3, S2+S3), each (cost·Q, schedule).
const PIN_T10_CAP_OFF: &[Row] = &[
    (6, (0, &[]), (0, &[]), (0, &[])),
    (7, (0, &[]), (0, &[]), (0, &[])),
    (8, (0, &[]), (0, &[]), (0, &[])),
    (9, (0, &[]), (0, &[]), (0, &[])),
    (10, (0, &[]), (0, &[]), (0, &[])),
    (11, (0, &[]), (0, &[]), (1210, &[1])),
    (12, (1210, &[1]), (1210, &[1]), (1320, &[2])),
    (13, (2530, &[1, 1]), (1320, &[2]), (1430, &[3])),
    (14, (3960, &[1, 1, 1]), (1430, &[3]), (1760, &[4])),
    (15, (5500, &[1, 1, 1, 1]), (1760, &[4]), (2420, &[5])),
    (16, (7150, &[1, 1, 1, 1, 1]), (2420, &[5]), (3190, &[3, 3])),
    (
        17,
        (8910, &[1, 1, 1, 1, 1, 1]),
        (3190, &[3, 3]),
        (3520, &[4, 3]),
    ),
    (
        18,
        (10780, &[1, 1, 1, 1, 1, 1, 1]),
        (3520, &[4, 3]),
        (3960, &[4, 4]),
    ),
    (
        19,
        (12760, &[1, 1, 1, 1, 1, 1, 1, 1]),
        (3960, &[4, 4]),
        (4620, &[5, 4]),
    ),
    (
        20,
        (14850, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4620, &[5, 4]),
        (5390, &[5, 5]),
    ),
    (
        21,
        (17050, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5390, &[5, 5]),
        (6050, &[4, 4, 3]),
    ),
    (
        22,
        (19360, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (6050, &[4, 4, 3]),
        (6600, &[4, 4, 4]),
    ),
    (
        23,
        (21780, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (6600, &[4, 4, 4]),
        (7260, &[5, 4, 4]),
    ),
    (
        24,
        (24310, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (7260, &[5, 4, 4]),
        (8030, &[5, 5, 4]),
    ),
];
/// T = 9, cap = MODEL: (B, today, S3, S2+S3), each (cost·Q, schedule).
const PIN_T9_CAP_MODEL: &[Row] = &[
    (6, (0, &[]), (0, &[]), (0, &[])),
    (7, (0, &[]), (0, &[]), (0, &[])),
    (8, (0, &[]), (0, &[]), (0, &[])),
    (9, (0, &[]), (0, &[]), (0, &[])),
    (10, (0, &[]), (0, &[]), (457, &[1])),
    (11, (457, &[1]), (457, &[1]), (567, &[2])),
    (12, (1024, &[1, 1]), (567, &[2]), (677, &[3])),
    (13, (1701, &[1, 1, 1]), (677, &[3]), (1007, &[4])),
    (14, (2488, &[1, 1, 1, 1]), (1007, &[4]), (1464, &[3, 2])),
    (
        15,
        (3385, &[1, 1, 1, 1, 1]),
        (1464, &[3, 2]),
        (1684, &[3, 3]),
    ),
    (
        16,
        (4392, &[1, 1, 1, 1, 1, 1]),
        (1684, &[3, 3]),
        (2014, &[4, 3]),
    ),
    (
        17,
        (5509, &[1, 1, 1, 1, 1, 1, 1]),
        (2014, &[4, 3]),
        (2454, &[4, 4]),
    ),
    (
        18,
        (6736, &[1, 1, 1, 1, 1, 1, 1, 1]),
        (2454, &[4, 4]),
        (3021, &[3, 3, 3]),
    ),
    (
        19,
        (8073, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3021, &[3, 3, 3]),
        (3351, &[4, 3, 3]),
    ),
    (
        20,
        (9520, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3351, &[4, 3, 3]),
        (3791, &[4, 4, 3]),
    ),
    (
        21,
        (11077, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3791, &[4, 4, 3]),
        (4341, &[4, 4, 4]),
    ),
    (
        22,
        (12744, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4341, &[4, 4, 4]),
        (5001, &[5, 4, 4]),
    ),
    (
        23,
        (14521, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5001, &[5, 4, 4]),
        (5458, &[4, 4, 3, 3]),
    ),
    (
        24,
        (16408, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5458, &[4, 4, 3, 3]),
        (6008, &[4, 4, 4, 3]),
    ),
];
/// T = 10, cap = MODEL: (B, today, S3, S2+S3), each (cost·Q, schedule).
const PIN_T10_CAP_MODEL: &[Row] = &[
    (6, (0, &[]), (0, &[]), (0, &[])),
    (7, (0, &[]), (0, &[]), (0, &[])),
    (8, (0, &[]), (0, &[]), (0, &[])),
    (9, (0, &[]), (0, &[]), (0, &[])),
    (10, (0, &[]), (0, &[]), (0, &[])),
    (11, (0, &[]), (0, &[]), (567, &[1])),
    (12, (567, &[1]), (567, &[1]), (677, &[2])),
    (13, (1244, &[1, 1]), (677, &[2]), (787, &[3])),
    (14, (2031, &[1, 1, 1]), (787, &[3]), (1117, &[4])),
    (15, (2928, &[1, 1, 1, 1]), (1117, &[4]), (1684, &[3, 2])),
    (
        16,
        (3935, &[1, 1, 1, 1, 1]),
        (1684, &[3, 2]),
        (1904, &[3, 3]),
    ),
    (
        17,
        (5052, &[1, 1, 1, 1, 1, 1]),
        (1904, &[3, 3]),
        (2234, &[4, 3]),
    ),
    (
        18,
        (6279, &[1, 1, 1, 1, 1, 1, 1]),
        (2234, &[4, 3]),
        (2674, &[4, 4]),
    ),
    (
        19,
        (7616, &[1, 1, 1, 1, 1, 1, 1, 1]),
        (2674, &[4, 4]),
        (3334, &[5, 4]),
    ),
    (
        20,
        (9063, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3334, &[5, 4]),
        (3681, &[4, 3, 3]),
    ),
    (
        21,
        (10620, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3681, &[4, 3, 3]),
        (4121, &[4, 4, 3]),
    ),
    (
        22,
        (12287, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4121, &[4, 4, 3]),
        (4671, &[4, 4, 4]),
    ),
    (
        23,
        (14064, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4671, &[4, 4, 4]),
        (5331, &[5, 4, 4]),
    ),
    (
        24,
        (15951, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5331, &[5, 4, 4]),
        (5898, &[4, 4, 3, 3]),
    ),
];
/// T = 9, cap = AUTO: (B, today, S3, S2+S3), each (cost·Q, schedule).
const PIN_T9_CAP_AUTO: &[Row] = &[
    (6, (0, &[]), (0, &[]), (0, &[])),
    (7, (0, &[]), (0, &[]), (0, &[])),
    (8, (0, &[]), (0, &[]), (0, &[])),
    (9, (0, &[]), (0, &[]), (0, &[])),
    (10, (0, &[]), (0, &[]), (777, &[1])),
    (11, (777, &[1]), (777, &[1]), (887, &[2])),
    (12, (1664, &[1, 1]), (887, &[2]), (997, &[3])),
    (13, (2661, &[1, 1, 1]), (997, &[3]), (1327, &[4])),
    (14, (3768, &[1, 1, 1, 1]), (1327, &[4]), (1987, &[5])),
    (15, (4985, &[1, 1, 1, 1, 1]), (1987, &[5]), (2324, &[3, 3])),
    (
        16,
        (6312, &[1, 1, 1, 1, 1, 1]),
        (2324, &[3, 3]),
        (2654, &[4, 3]),
    ),
    (
        17,
        (7749, &[1, 1, 1, 1, 1, 1, 1]),
        (2654, &[4, 3]),
        (3094, &[4, 4]),
    ),
    (
        18,
        (9296, &[1, 1, 1, 1, 1, 1, 1, 1]),
        (3094, &[4, 4]),
        (3754, &[5, 4]),
    ),
    (
        19,
        (10953, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3754, &[5, 4]),
        (4311, &[4, 3, 3]),
    ),
    (
        20,
        (12720, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4311, &[4, 3, 3]),
        (4751, &[4, 4, 3]),
    ),
    (
        21,
        (14597, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4751, &[4, 4, 3]),
        (5301, &[4, 4, 4]),
    ),
    (
        22,
        (16584, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5301, &[4, 4, 4]),
        (5961, &[5, 4, 4]),
    ),
    (
        23,
        (18681, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5961, &[5, 4, 4]),
        (6731, &[5, 5, 4]),
    ),
    (
        24,
        (20888, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (6731, &[5, 5, 4]),
        (7288, &[4, 4, 4, 3]),
    ),
];
/// T = 10, cap = AUTO: (B, today, S3, S2+S3), each (cost·Q, schedule).
const PIN_T10_CAP_AUTO: &[Row] = &[
    (6, (0, &[]), (0, &[]), (0, &[])),
    (7, (0, &[]), (0, &[]), (0, &[])),
    (8, (0, &[]), (0, &[]), (0, &[])),
    (9, (0, &[]), (0, &[]), (0, &[])),
    (10, (0, &[]), (0, &[]), (0, &[])),
    (11, (0, &[]), (0, &[]), (887, &[1])),
    (12, (887, &[1]), (887, &[1]), (997, &[2])),
    (13, (1884, &[1, 1]), (997, &[2]), (1107, &[3])),
    (14, (2991, &[1, 1, 1]), (1107, &[3]), (1437, &[4])),
    (15, (4208, &[1, 1, 1, 1]), (1437, &[4]), (2097, &[5])),
    (16, (5535, &[1, 1, 1, 1, 1]), (2097, &[5]), (2544, &[3, 3])),
    (
        17,
        (6972, &[1, 1, 1, 1, 1, 1]),
        (2544, &[3, 3]),
        (2874, &[4, 3]),
    ),
    (
        18,
        (8519, &[1, 1, 1, 1, 1, 1, 1]),
        (2874, &[4, 3]),
        (3314, &[4, 4]),
    ),
    (
        19,
        (10176, &[1, 1, 1, 1, 1, 1, 1, 1]),
        (3314, &[4, 4]),
        (3974, &[5, 4]),
    ),
    (
        20,
        (11943, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3974, &[5, 4]),
        (4641, &[4, 3, 3]),
    ),
    (
        21,
        (13820, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (4641, &[4, 3, 3]),
        (5081, &[4, 4, 3]),
    ),
    (
        22,
        (15807, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5081, &[4, 4, 3]),
        (5631, &[4, 4, 4]),
    ),
    (
        23,
        (17904, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (5631, &[4, 4, 4]),
        (6291, &[5, 4, 4]),
    ),
    (
        24,
        (20111, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (6291, &[5, 4, 4]),
        (7061, &[5, 5, 4]),
    ),
];

fn check_pin(name: &str, terminal_log: u32, cap: &dyn Fn(u32) -> u32, rows: &[Row]) {
    assert_eq!(rows.len(), 19, "{name}: B = 6..=24");
    for &(b, (today_q, today), (s3_q, s3), (s2_q, s2)) in rows {
        let ctx = format!("{name} B={b}");
        // today: the all-ones chain from B − 1 (no committed layer when B − 1 ≤ T).
        let b0 = fri_chain_start(b, false);
        assert_eq!(legacy_fri_schedule(b0, terminal_log), today, "{ctx} today");
        assert_eq!(perm_cost(b0, today, cap), Some(today_q), "{ctx} today cost");
        // S3: the DP from B − 1.
        let got = perm_schedule(b0, terminal_log, cap);
        assert_eq!(got.schedule, s3, "{ctx} S3 schedule");
        assert_eq!(got.cost_q, s3_q, "{ctx} S3 cost");
        assert_eq!(got.trees as usize, s3.len(), "{ctx} S3 trees");
        // S2+S3: the DP from B (the DEEP codeword is committed).
        let b0 = fri_chain_start(b, true);
        let got = perm_schedule(b0, terminal_log, cap);
        assert_eq!(got.schedule, s2, "{ctx} S2+S3 schedule");
        assert_eq!(got.cost_q, s2_q, "{ctx} S2+S3 cost");
    }
}

#[test]
fn design_model_reproduces_the_fri_md_table() {
    check_pin("T9 cap off", 9, &no_cap, PIN_T9_CAP_OFF);
    check_pin("T10 cap off", 10, &no_cap, PIN_T10_CAP_OFF);
    check_pin("T9 cap model", 9, &cap_design_model, PIN_T9_CAP_MODEL);
    check_pin("T10 cap model", 10, &cap_design_model, PIN_T10_CAP_MODEL);
    check_pin("T9 cap auto", 9, &cap_auto, PIN_T9_CAP_AUTO);
    check_pin("T10 cap auto", 10, &cap_auto, PIN_T10_CAP_AUTO);
}

/// Spot checks tying the design-model pins to the printed FRI.md §2.2 table
/// (costs there are per query, i.e. cost·Q / 110, rounded to two decimals).
#[test]
fn design_model_pins_match_fri_md_table() {
    let per_query = |cost_q: u64| (cost_q as f64 / Q as f64 * 100.0).round() / 100.0;
    let today = legacy_fri_schedule(20, 9);
    assert_eq!(per_query(perm_cost(20, &today, &no_cap).unwrap()), 165.0);
    assert_eq!(
        per_query(perm_cost(20, &today, &cap_design_model).unwrap()),
        100.70
    );
    let c = perm_schedule(20, 9, &no_cap);
    assert_eq!((per_query(c.cost_q), c.schedule), (52.0, vec![4, 4, 3]));
    let c = perm_schedule(20, 9, &cap_design_model);
    assert_eq!((per_query(c.cost_q), c.schedule), (34.46, vec![4, 4, 3]));
    let c = perm_schedule(21, 9, &cap_design_model);
    assert_eq!((per_query(c.cost_q), c.schedule), (39.46, vec![4, 4, 4]));
    assert_eq!(perm_schedule(18, 9, &no_cap).schedule, vec![5, 4]);
    assert_eq!(
        perm_schedule(18, 9, &cap_design_model).schedule,
        vec![3, 3, 3]
    );
    let c = perm_schedule(20, 10, &cap_design_model);
    assert_eq!((per_query(c.cost_q), c.schedule), (33.46, vec![4, 3, 3]));
    let c = perm_schedule(21, 10, &cap_design_model);
    assert_eq!((per_query(c.cost_q), c.schedule), (37.46, vec![4, 4, 3]));
}

// ---------------------------------------------------------------------------
// U1: the PRODUCTION schedules (cost-law objective), pinned from the Rust DP.
// The schedule is a format constant: a change here is a format change.
// ---------------------------------------------------------------------------

/// (B, S3 schedule from B − 1, S2+S3 schedule from B) for B = 6..=24.
type CostRow = (u32, &'static [u8], &'static [u8]);

/// Generated by `print_cost_law_schedule_table` (below, `--ignored`) from the
/// Rust DP. Re-pinned for RULINGS 22 (every emitted row priced): the cap-auto
/// tables did NOT move; four cap-off entries did — T = 9: B = 16 S2 [4,3] →
/// [3,2,2], B = 17 S3 [4,3] → [3,2,2]; T = 10: B = 14 S2 [4] → [2,2],
/// B = 15 S3 [4] → [2,2]. The whole table was cross-checked against an
/// independent Python reproduction of the objective (lane I-PRICE scratch):
/// identical. REVIEW-FRI F2's cost-law column gives [2,2] / [3,3,3] /
/// [3,3,3,2] / [3,3,3,3,2] at B = 14 / 19 / 21 / 24, T = 9 — the Auto rows.
const PIN_COST_T9_CAP_OFF: &[CostRow] = &[
    (6, &[], &[]),
    (7, &[], &[]),
    (8, &[], &[]),
    (9, &[], &[]),
    (10, &[], &[1]),
    (11, &[1], &[2]),
    (12, &[2], &[3]),
    (13, &[3], &[2, 2]),
    (14, &[2, 2], &[3, 2]),
    (15, &[3, 2], &[3, 3]),
    (16, &[3, 3], &[3, 2, 2]),
    (17, &[3, 2, 2], &[3, 3, 2]),
    (18, &[3, 3, 2], &[3, 3, 3]),
    (19, &[3, 3, 3], &[4, 3, 3]),
    (20, &[4, 3, 3], &[3, 3, 3, 2]),
    (21, &[3, 3, 3, 2], &[3, 3, 3, 3]),
    (22, &[3, 3, 3, 3], &[4, 3, 3, 3]),
    (23, &[4, 3, 3, 3], &[3, 3, 3, 3, 2]),
    (24, &[3, 3, 3, 3, 2], &[3, 3, 3, 3, 3]),
];
const PIN_COST_T10_CAP_OFF: &[CostRow] = &[
    (6, &[], &[]),
    (7, &[], &[]),
    (8, &[], &[]),
    (9, &[], &[]),
    (10, &[], &[]),
    (11, &[], &[1]),
    (12, &[1], &[2]),
    (13, &[2], &[3]),
    (14, &[3], &[2, 2]),
    (15, &[2, 2], &[3, 2]),
    (16, &[3, 2], &[3, 3]),
    (17, &[3, 3], &[4, 3]),
    (18, &[4, 3], &[3, 3, 2]),
    (19, &[3, 3, 2], &[3, 3, 3]),
    (20, &[3, 3, 3], &[4, 3, 3]),
    (21, &[4, 3, 3], &[3, 3, 3, 2]),
    (22, &[3, 3, 3, 2], &[3, 3, 3, 3]),
    (23, &[3, 3, 3, 3], &[4, 3, 3, 3]),
    (24, &[4, 3, 3, 3], &[3, 3, 3, 3, 2]),
];
const PIN_COST_T9_CAP_AUTO: &[CostRow] = &[
    (6, &[], &[]),
    (7, &[], &[]),
    (8, &[], &[]),
    (9, &[], &[]),
    (10, &[], &[1]),
    (11, &[1], &[2]),
    (12, &[2], &[3]),
    (13, &[3], &[2, 2]),
    (14, &[2, 2], &[3, 2]),
    (15, &[3, 2], &[3, 3]),
    (16, &[3, 3], &[3, 2, 2]),
    (17, &[3, 2, 2], &[3, 3, 2]),
    (18, &[3, 3, 2], &[3, 3, 3]),
    (19, &[3, 3, 3], &[3, 3, 2, 2]),
    (20, &[3, 3, 2, 2], &[3, 3, 3, 2]),
    (21, &[3, 3, 3, 2], &[3, 3, 3, 3]),
    (22, &[3, 3, 3, 3], &[4, 3, 3, 3]),
    (23, &[4, 3, 3, 3], &[3, 3, 3, 3, 2]),
    (24, &[3, 3, 3, 3, 2], &[3, 3, 3, 3, 3]),
];
const PIN_COST_T10_CAP_AUTO: &[CostRow] = &[
    (6, &[], &[]),
    (7, &[], &[]),
    (8, &[], &[]),
    (9, &[], &[]),
    (10, &[], &[]),
    (11, &[], &[1]),
    (12, &[1], &[2]),
    (13, &[2], &[3]),
    (14, &[3], &[2, 2]),
    (15, &[2, 2], &[3, 2]),
    (16, &[3, 2], &[3, 3]),
    (17, &[3, 3], &[3, 2, 2]),
    (18, &[3, 2, 2], &[3, 3, 2]),
    (19, &[3, 3, 2], &[3, 3, 3]),
    (20, &[3, 3, 3], &[4, 3, 3]),
    (21, &[4, 3, 3], &[3, 3, 3, 2]),
    (22, &[3, 3, 3, 2], &[3, 3, 3, 3]),
    (23, &[3, 3, 3, 3], &[4, 3, 3, 3]),
    (24, &[4, 3, 3, 3], &[3, 3, 3, 3, 2]),
];

fn check_cost_pin(name: &str, terminal_log: u32, cap: CapPolicy, rows: &[CostRow]) {
    assert_eq!(rows.len(), 19, "{name}: B = 6..=24");
    for &(b, s3, s2) in rows {
        let ctx = format!("{name} B={b}");
        let got = fri_schedule(
            fri_chain_start(b, false),
            terminal_log,
            Q,
            cap,
            FRI_SCHEDULE_DMAX,
        );
        assert_eq!(got, s3, "{ctx} S3");
        let got = fri_schedule(
            fri_chain_start(b, true),
            terminal_log,
            Q,
            cap,
            FRI_SCHEDULE_DMAX,
        );
        assert_eq!(got, s2, "{ctx} S2+S3");
    }
}

#[test]
fn schedule_pinned_table() {
    check_cost_pin("T9 cap off", 9, CapPolicy::Off, PIN_COST_T9_CAP_OFF);
    check_cost_pin("T10 cap off", 10, CapPolicy::Off, PIN_COST_T10_CAP_OFF);
    check_cost_pin("T9 cap auto", 9, CapPolicy::Auto, PIN_COST_T9_CAP_AUTO);
    check_cost_pin("T10 cap auto", 10, CapPolicy::Auto, PIN_COST_T10_CAP_AUTO);
}

/// Prints the U1 table in the `fri_schedule_cost_pins.rs` format. Run with
/// `-- --ignored --nocapture` to regenerate after a DELIBERATE objective change.
#[test]
#[ignore = "generator for fri_schedule_cost_pins.rs"]
fn print_cost_law_schedule_table() {
    for (name, t, cap) in [
        ("T9_CAP_OFF", 9, CapPolicy::Off),
        ("T10_CAP_OFF", 10, CapPolicy::Off),
        ("T9_CAP_AUTO", 9, CapPolicy::Auto),
        ("T10_CAP_AUTO", 10, CapPolicy::Auto),
    ] {
        println!("const PIN_COST_{name}_DATA: [CostRow; 19] = [");
        for b in 6..=24u32 {
            let s3 = fri_schedule(fri_chain_start(b, false), t, Q, cap, FRI_SCHEDULE_DMAX);
            let s2 = fri_schedule(fri_chain_start(b, true), t, Q, cap, FRI_SCHEDULE_DMAX);
            println!("    ({b}, &{s3:?}, &{s2:?}),");
        }
        println!("];");
    }
}

/// B = 21, T = 9, no cap, S3: every candidate schedule's cost by hand, so the
/// pinned optimum is shown to be one, not merely reproduced.
#[test]
fn cost_law_b21_by_hand() {
    let layer = |d: u64, depth: u64| -> u64 {
        let leaf = (3u64 << d).div_ceil(8).max(1);
        let n = 1u64 << d;
        let g = n - 1;
        let extras = d * (567 + 477)
            + d.saturating_sub(2) * 522
            + u64::from(d >= 2) * 477
            + 2 * 522
            + 8 * 477
            + 528
            + n * (528 + 460)
            + (3 * n).div_ceil(4) * 528
            + depth * 460;
        Q * (leaf * 2251 + depth * (2251 + 567) + g * (567 + 2610) + d * 477 + extras)
    };
    let cost = |sched: &[u64]| {
        let mut b = 20u64;
        sched
            .iter()
            .map(|&d| {
                b -= d;
                layer(d, b)
            })
            .sum::<u64>()
    };
    let got = fri_schedule_with_cost(20, 9, Q, CapPolicy::Off, FRI_SCHEDULE_DMAX);
    let as_u64: Vec<u64> = got.schedule.iter().map(|&d| u64::from(d)).collect();
    assert_eq!(got.cost_q, cost(&as_u64));
    // It beats the permutation objective's choice and today's.
    assert!(got.cost_q <= cost(&[4, 4, 3]));
    assert!(got.cost_q < cost(&[1; 11]));
}

// ---------------------------------------------------------------------------
// U2: brute-force optimality for b₀ ≤ 16, under the production objective.
// ---------------------------------------------------------------------------

/// Independent oracle for one layer's cost-law price (the module docs'
/// formula, written out again from the CAPPED rows priced directly plus the
/// cap's once-per-tree cost, rather than the uncapped rows minus the gain).
fn oracle_layer_q(d: u32, depth: u32, q: u64, cap: CapPolicy) -> u64 {
    let (wc, ws, wu, wh, wq) = (2251i128, 567i128, 528i128, 460i128, 3789i128);
    let (xalu, balu) = (522i128, 477i128);
    let leaf = i128::from((3u64 << d).div_ceil(8).max(1) as u32);
    let n = 1i128 << d;
    let d = i128::from(d);
    let c = cap.height(q as usize, depth as usize) as i128;
    let walk = i128::from(depth) - c;
    let cap_nodes = 1i128 << c;
    let selects = (n - 1) + walk + (cap_nodes - 1) + d;
    let xalus = 5 * (n - 1) + 2 + (d - 2).max(0);
    let balus = d + d + i128::from(d >= 2) + 8;
    let hashes = leaf + walk;
    let unpacks = n + 1 + i128::from(c > 0);
    let packs = (3 * n + 3) / 4;
    let hints = n + walk;
    let per_query = selects * ws
        + xalus * xalu
        + balus * balu
        + hashes * wc
        + (unpacks + packs) * wu
        + hints * wh;
    let per_tree = if c == 0 {
        0
    } else {
        (cap_nodes - 1) * wc + cap_nodes * wh + wq
    };
    (q as i128 * per_query + per_tree) as u64
}

/// Every composition of `b0 − t` into parts in `1..=dmax`, with its cost;
/// returns the minimum under (cost, trees, schedule) lexicographic order.
fn brute_force(b0: u32, t: u32, q: u64, cap: CapPolicy, dmax: u32) -> (u64, Vec<u8>) {
    struct Search {
        t: u32,
        q: u64,
        cap: CapPolicy,
        dmax: u32,
        prefix: Vec<u8>,
        best: Option<(u64, usize, Vec<u8>)>,
    }
    impl Search {
        fn walk(&mut self, b: u32, cost: u64) {
            if b == self.t {
                let cand = (cost, self.prefix.len(), self.prefix.clone());
                if self.best.as_ref().is_none_or(|cur| cand < *cur) {
                    self.best = Some(cand);
                }
                return;
            }
            for d in 1..=self.dmax.min(b - self.t) {
                self.prefix.push(d as u8);
                let c = cost + oracle_layer_q(d, b - d, self.q, self.cap);
                self.walk(b - d, c);
                self.prefix.pop();
            }
        }
    }
    let mut s = Search {
        t,
        q,
        cap,
        dmax,
        prefix: Vec::new(),
        best: None,
    };
    s.walk(b0, 0);
    let (cost, _, sched) = s.best.expect("at least the empty / all-ones composition");
    (cost, sched)
}

#[test]
fn dp_is_optimal() {
    let caps = [
        CapPolicy::Off,
        CapPolicy::Auto,
        CapPolicy::Fixed(2),
        CapPolicy::Fixed(5),
    ];
    let mut checked = 0u32;
    for cap in caps {
        for q in [1u64, 3, 110] {
            for dmax in [1u32, 2, 3, FRI_SCHEDULE_DMAX] {
                for b0 in 0..=16u32 {
                    for t in 0..=b0 {
                        let got = fri_schedule_with_cost(b0, t, q, cap, dmax);
                        let (cost, sched) = brute_force(b0, t, q, cap, dmax);
                        let ctx = format!("cap={cap:?} q={q} dmax={dmax} b0={b0} t={t}");
                        assert_eq!(got.cost_q, cost, "{ctx}: cost");
                        // The tie rule makes the optimum unique: the smallest
                        // (trees, schedule) among the cost-optimal ones.
                        assert_eq!(got.schedule, sched, "{ctx}: schedule");
                        assert_eq!(got.trees as usize, sched.len(), "{ctx}: trees");
                        assert_eq!(
                            fri_schedule_cost_q(b0, &got.schedule, q, cap),
                            Some(got.cost_q),
                            "{ctx}: cost of the schedule"
                        );
                        let sum: u32 = got.schedule.iter().map(|&d| u32::from(d)).sum();
                        assert_eq!(sum, b0 - t, "{ctx}: lands on the terminal");
                        assert!(
                            got.schedule
                                .iter()
                                .all(|&d| (1..=dmax).contains(&u32::from(d)))
                        );
                        checked += 1;
                    }
                    // Above the terminal: nothing to commit.
                    for t in b0 + 1..=b0 + 2 {
                        assert!(fri_schedule(b0, t, q, cap, dmax).is_empty());
                    }
                }
            }
        }
    }
    assert_eq!(checked, 4 * 3 * 4 * (17 * 18 / 2));
}

#[test]
fn dmax_one_is_the_legacy_schedule() {
    for b0 in 0..=30u32 {
        for t in 0..=31u32 {
            for cap in [CapPolicy::Off, CapPolicy::Auto] {
                assert_eq!(fri_schedule(b0, t, Q, cap, 1), legacy_fri_schedule(b0, t));
                // dmax = 0 is treated as 1.
                assert_eq!(fri_schedule(b0, t, Q, cap, 0), legacy_fri_schedule(b0, t));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// U3: the legacy constructor is unchanged.
// ---------------------------------------------------------------------------

/// `FriFoldLayout::new` as it was before the schedule existed (terminal.rs @
/// 5d0b0a41a), copied verbatim: (total_folds, num_committed, terminal_len,
/// effective_k).
fn old_layout(lde_log: u32, blowup_log: u32, k: u32) -> (u32, usize, usize, u32) {
    let terminal_log = (blowup_log + k).min(lde_log);
    let total_folds = lde_log - terminal_log;
    (
        total_folds,
        total_folds.saturating_sub(1) as usize,
        1usize << terminal_log,
        terminal_log - blowup_log,
    )
}

fn dp_format(cap: CapPolicy) -> FriFormat {
    FriFormat {
        mode: FriMode::Dp,
        one_row: false,
        num_queries: Q,
        cap,
        schedule_override: None,
    }
}

#[test]
fn legacy_layout_equals_old_layout() {
    // one_row = true is exercised through the schedule arithmetic only (the
    // layout is still well defined); the prover refuses it (S2 not built).
    let dp_formats: Vec<FriFormat> = [false, true]
        .into_iter()
        .flat_map(|one_row| {
            [CapPolicy::Off, CapPolicy::Auto].map(|cap| FriFormat {
                one_row,
                ..dp_format(cap)
            })
        })
        .collect();
    let mut checked = 0u32;
    for blowup_log in 1..=4u32 {
        // The LDE is at least the blowup (trace length ≥ 1).
        for lde_log in blowup_log..=30u32 {
            for k in 0..=10u32 {
                let ctx = format!("lde_log={lde_log} blowup_log={blowup_log} k={k}");
                let (total_folds, num_committed, terminal_len, effective_k) =
                    old_layout(lde_log, blowup_log, k);
                let new = FriFoldLayout::new(lde_log, blowup_log, k);
                assert_eq!(new.total_folds, total_folds, "{ctx}");
                assert_eq!(new.num_committed, num_committed, "{ctx}");
                assert_eq!(new.terminal_len, terminal_len, "{ctx}");
                assert_eq!(new.effective_k, effective_k, "{ctx}");
                assert_eq!(new.schedule, vec![1u8; num_committed], "{ctx}");
                assert!(!new.one_row, "{ctx}");
                assert!(new.is_legacy(), "{ctx}");
                assert_eq!(new.opened_values_per_query(), num_committed, "{ctx}");

                // Pair mode ignores the query count, the cap policy and any
                // schedule override.
                for cap in [CapPolicy::Off, CapPolicy::Auto] {
                    let pair = FriFormat {
                        mode: FriMode::Pair,
                        schedule_override: FriScheduleOverride::new(&[3, 1]),
                        ..dp_format(cap)
                    };
                    assert_eq!(
                        FriFoldLayout::for_format(lde_log, blowup_log, k, &pair),
                        Some(new.clone()),
                        "{ctx}"
                    );
                }
                assert_eq!(
                    FriFoldLayout::from_schedule(
                        lde_log,
                        blowup_log,
                        k,
                        false,
                        new.schedule.clone()
                    ),
                    Some(new.clone()),
                    "{ctx}"
                );

                // Any format moves only the split of the folds into committed
                // layers (and, off the legacy format, the encoding).
                for fmt in &dp_formats {
                    let l = FriFoldLayout::for_format(lde_log, blowup_log, k, fmt)
                        .expect("the DP lands on the terminal");
                    assert_eq!(
                        (l.total_folds, l.terminal_len, l.effective_k, l.one_row),
                        (total_folds, terminal_len, effective_k, fmt.one_row),
                        "{ctx} {fmt:?}"
                    );
                    assert!(
                        !l.is_legacy(),
                        "{ctx} {fmt:?}: Dp is never the legacy encoding"
                    );
                    assert_eq!(l.num_committed, l.schedule.len(), "{ctx} {fmt:?}");
                    let covered: u32 = l.schedule.iter().map(|&d| u32::from(d)).sum();
                    let expected = match (total_folds, fmt.one_row) {
                        (0, _) => 0,
                        (n, true) => n,
                        (n, false) => n - 1,
                    };
                    assert_eq!(covered, expected, "{ctx} {fmt:?}");
                    assert_eq!(
                        l.opened_values_per_query(),
                        l.schedule.iter().map(|&d| 1usize << d).sum::<usize>(),
                        "{ctx} {fmt:?}"
                    );
                    for j in 0..l.num_committed {
                        let d = u32::from(l.schedule[j]);
                        assert_eq!(
                            l.layer_depth(lde_log, j) + d,
                            l.layer_log_len(lde_log, j),
                            "{ctx} {fmt:?} layer {j}"
                        );
                    }
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 11 * (30 + 29 + 28 + 27));
}

#[test]
fn from_schedule_rejects_a_schedule_that_does_not_cover_the_folds() {
    // lde_log 20, blowup 2, k 7: total_folds 11, row-pair chain covers 10 bits.
    assert!(FriFoldLayout::from_schedule(20, 2, 7, false, vec![4, 4, 2]).is_some());
    assert!(FriFoldLayout::from_schedule(20, 2, 7, false, vec![4, 4, 3]).is_none());
    assert!(FriFoldLayout::from_schedule(20, 2, 7, false, vec![4, 4, 1]).is_none());
    // One-row: the chain covers all 11.
    assert!(FriFoldLayout::from_schedule(20, 2, 7, true, vec![4, 4, 3]).is_some());
    assert!(FriFoldLayout::from_schedule(20, 2, 7, true, vec![4, 4, 2]).is_none());
    // Zero and over-DMAX exponents.
    assert!(FriFoldLayout::from_schedule(20, 2, 7, false, vec![0, 5, 5]).is_none());
    assert!(FriFoldLayout::from_schedule(20, 2, 7, false, vec![7, 3]).is_none());
    // No fold: only the empty schedule.
    assert!(FriFoldLayout::from_schedule(8, 2, 7, false, vec![]).is_some());
    assert!(FriFoldLayout::from_schedule(8, 2, 7, true, vec![1]).is_none());
    // One fold, row pair: no committed layer.
    assert!(FriFoldLayout::from_schedule(10, 2, 7, false, vec![]).is_some());
    assert!(FriFoldLayout::from_schedule(10, 2, 7, false, vec![1]).is_none());
    assert!(FriFoldLayout::from_schedule(10, 2, 7, true, vec![1]).is_some());
}

// ---------------------------------------------------------------------------
// The layout from `ProofOptions` (what the prover and verifier build).
// ---------------------------------------------------------------------------

fn options_with(format: ProofFormat) -> ProofOptions {
    ProofOptions {
        format,
        ..ProofOptions::default_test_options()
    }
}

#[test]
fn layout_from_options() {
    // Default format: today's layout.
    let o = options_with(ProofFormat::DEFAULT);
    let k = u32::from(o.fri_final_poly_log_degree);
    assert_eq!(
        FriFoldLayout::for_options(20, 1, &o, false),
        Ok(FriFoldLayout::new(20, 1, k))
    );
    // Dp: the DP's schedule under the options' query count and cap.
    let o = options_with(ProofFormat {
        fri_mode: FriMode::Dp,
        ..ProofFormat::DEFAULT
    });
    let l = FriFoldLayout::for_options(20, 1, &o, false).unwrap();
    let t = (1 + k).min(20);
    assert_eq!(
        l.schedule,
        fri_schedule(
            19,
            t,
            o.fri_number_of_queries as u64,
            CapPolicy::Off,
            FRI_SCHEDULE_DMAX
        )
    );
    assert!(!l.is_legacy());
    // An override that fits is taken verbatim; one that does not is an error.
    let span = 19 - t;
    let mut fit = vec![1u8; span as usize - 3];
    fit.insert(0, 3);
    let o = options_with(ProofFormat {
        fri_mode: FriMode::Dp,
        fri_schedule_override: FriScheduleOverride::new(&fit),
        ..ProofFormat::DEFAULT
    });
    assert_eq!(
        FriFoldLayout::for_options(20, 1, &o, false)
            .unwrap()
            .schedule,
        fit
    );
    let o = options_with(ProofFormat {
        fri_mode: FriMode::Dp,
        fri_schedule_override: FriScheduleOverride::new(&[3, 1]),
        ..ProofFormat::DEFAULT
    });
    assert_eq!(
        FriFoldLayout::for_options(20, 1, &o, false),
        Err(FriFormatError::ScheduleOverrideMismatch)
    );
    // An all-ones override under Dp keeps the GROUP encoding.
    let o = options_with(ProofFormat {
        fri_mode: FriMode::Dp,
        fri_schedule_override: FriScheduleOverride::new(&vec![1u8; span as usize]),
        ..ProofFormat::DEFAULT
    });
    let l = FriFoldLayout::for_options(20, 1, &o, false).unwrap();
    assert_eq!(l.schedule, vec![1u8; span as usize]);
    assert!(!l.is_legacy());
    // One row (S2): the chain starts at the LDE size, the encoding is the
    // group one even at fri = pair, and the all-ones schedule covers every
    // fold (no uncommitted fold 0).
    for one_row in [OneRowMode::On, OneRowMode::Auto] {
        let o = options_with(ProofFormat {
            one_row,
            ..ProofFormat::DEFAULT
        });
        let l = FriFoldLayout::for_options(20, 1, &o, true).unwrap();
        assert!(l.one_row && !l.is_legacy());
        assert_eq!(l.schedule, vec![1u8; (20 - t) as usize]);
        assert_eq!(l.num_committed as u32, l.total_folds);
        assert_eq!(l.num_zetas(), l.num_committed);
        assert_eq!(
            l.layer_depth(20, 0),
            19,
            "the input tree: 2^20 values in pairs"
        );
        // The same options at a resolved row-pair layout: today's.
        assert_eq!(
            FriFoldLayout::for_options(20, 1, &o, false),
            Ok(FriFoldLayout::new(20, 1, k))
        );
    }
    // An override longer than the fixed capacity is refused at construction.
    assert!(FriScheduleOverride::new(&[1u8; 33]).is_none());
    assert_eq!(
        FriScheduleOverride::new(&[2, 1]).unwrap().as_slice(),
        &[2, 1]
    );
}
