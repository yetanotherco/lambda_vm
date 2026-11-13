//! MEMW_A's timestamp order by the step's two 16-bit limbs
//! ([`MemwAlignedOrder::Limbs`]), on the built AIR: the real interaction
//! lists over hand-built rows, the real LT chip beside them. No proof.
//!
//! - The structure: no ALU lookup, the two limbs' `IS_HALF` lookups at the
//!   tokens' multiplicity, every old token's high word the row's own
//!   `TIMESTAMP_1`, 21 interactions; the old form is the per-table STARK's.
//! - The lemma: on one high word the LT chip and the limbs accept exactly the
//!   steps `1..=2^32`, as field differences of the low words.
//! - Forgeries, each refused with a control that passes: an old timestamp
//!   equal to the new one, later than it, a limb of 2^16, an old high word
//!   other than the row's, and an old low word of `p − k` (accepted by the row,
//!   carried by no token any producer makes).
//! - Mutants of the interaction list, one a piece: each lets its forgery
//!   through, so each piece is what refuses it.
//! - The prover's side: the routing sends an access on two high words (or not
//!   later) to MEMW, the row lays its limbs out, BITWISE counts them and LT
//!   takes no comparison of the row.

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::lt_multiplicity_poc::eval_mult;
use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::lt::{self, LtConstraints, cols as lt_cols};
use crate::tables::memw::MemwOperation;
use crate::tables::memw_aligned::{
    self, AlignedRow, MemwAlignedOrder, bus_interactions_for, cols, step_fits_limbs, step_limbs,
};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{busless_air, validate_busless};

type Trace = TraceTable<GoldilocksField, GoldilocksExtension>;

/// The Goldilocks prime.
const P: u64 = 0xFFFF_FFFF_0000_0001;
const HALF: u64 = 1 << 16;

// =============================================================================
// The built AIR, evaluated on a row
// =============================================================================

/// A MEMW_A row in the limbs' form: an aligned 8-byte memory read at 0x1000,
/// its timestamp `(ts_lo, ts_hi)` and limbs `d0`, `d1` as given (any field
/// elements), `μ_read`, `μ_write` as given.
fn limbs_row(ts_lo: FE, ts_hi: u64, d0: FE, d1: FE, mu_read: u64, mu_write: u64) -> Vec<FE> {
    let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
    row[cols::BASE_ADDRESS[0]] = FE::from(0x1000u64);
    for (i, c) in cols::VALUE.into_iter().enumerate() {
        row[c] = FE::from(i as u64 + 1);
    }
    for (i, c) in cols::OLD.into_iter().enumerate() {
        row[c] = FE::from(i as u64 + 1);
    }
    row[cols::TIMESTAMP_0] = ts_lo;
    row[cols::TIMESTAMP_1] = FE::from(ts_hi);
    row[cols::WRITE8] = FE::one();
    row[cols::STEP_0] = d0;
    row[cols::STEP_1] = d1;
    row[cols::MU_READ] = FE::from(mu_read);
    row[cols::MU_WRITE] = FE::from(mu_write);
    row
}

/// What one active row of `interactions` sends: per interaction with a nonzero
/// multiplicity, its bus, whether it sends, and its tuple.
fn messages(interactions: &[BusInteraction], row: &[FE]) -> Vec<(BusId, bool, Vec<FE>)> {
    let g = |c: usize| row[c];
    interactions
        .iter()
        .filter(|i| eval_mult(&i.multiplicity, &g) != FE::zero())
        .map(|i| {
            let tuple = i.values.iter().flat_map(|v| v.combine_from(g)).collect();
            let bus = BusId::try_from(i.bus_id).expect("a bus id");
            (bus, i.is_sender, tuple)
        })
        .collect()
}

/// The `(low, high)` timestamp every old token of the row carries (the
/// Memory bus's senders: elements 3 and 4).
fn old_timestamps(interactions: &[BusInteraction], row: &[FE]) -> Vec<(FE, FE)> {
    messages(interactions, row)
        .into_iter()
        .filter(|(bus, sender, _)| *bus == BusId::Memory && *sender)
        .map(|(_, _, t)| (t[3], t[4]))
        .collect()
}

/// Whether every `IS_HALF` message of the row is a 16-bit value: what the
/// BITWISE table's preprocessed half-word column holds and nothing else.
fn halves_in_range(interactions: &[BusInteraction], row: &[FE]) -> bool {
    messages(interactions, row)
        .into_iter()
        .filter(|(bus, sender, _)| *bus == BusId::IsHalfword && *sender)
        .all(|(_, _, t)| t.len() == 1 && t[0].canonical_u64() < HALF)
}

/// Whether the row of `interactions` accepts the access from `(old_lo, old_hi)`
/// to `(ts_lo, ts_hi)` with limbs `(d0, d1)` and multiplicities
/// `(μ_read, μ_write)`: every old token claims that old timestamp (eight of
/// them, a width-8 access) and every half-word message is in range.
#[allow(clippy::too_many_arguments)]
fn row_accepts(
    interactions: &[BusInteraction],
    old: (FE, FE),
    ts_lo: FE,
    ts_hi: u64,
    d0: FE,
    d1: FE,
    mu: (u64, u64),
) -> bool {
    let row = limbs_row(ts_lo, ts_hi, d0, d1, mu.0, mu.1);
    let olds = old_timestamps(interactions, &row);
    olds.len() == 8 && olds.iter().all(|t| *t == old) && halves_in_range(interactions, &row)
}

/// Whether the real limbs' AIR can accept an access from `(old_lo, old_hi)`
/// to `(ts_lo, ts_hi)`: the limbs the identity forces, if in range, are the
/// only witness (the identity is linear in them), so this tries them, and for
/// a refused pair also checks in-range limbs of every other kind fail.
fn limbs_accept(old_lo: FE, old_hi: u64, ts_lo: FE, ts_hi: u64) -> bool {
    let air = bus_interactions_for(MemwAlignedOrder::Limbs);
    let old = (old_lo, FE::from(old_hi));
    let forced = (ts_lo - old_lo - FE::one()).canonical_u64();
    let (d0, d1) = (FE::from(forced % HALF), FE::from(forced / HALF));
    let accepted = row_accepts(&air, old, ts_lo, ts_hi, d0, d1, (1, 0));
    if !accepted {
        // In-range limbs: the forced ones cut to 16 bits, and the range's ends.
        for (a, b) in [
            (forced % HALF, (forced / HALF) % HALF),
            (0, 0),
            (HALF - 1, HALF - 1),
            (0, HALF - 1),
            (HALF - 1, 0),
        ] {
            assert!(
                !row_accepts(&air, old, ts_lo, ts_hi, FE::from(a), FE::from(b), (1, 0)),
                "in-range limbs {a:#x}, {b:#x} took a refused access"
            );
        }
    }
    accepted
}

// =============================================================================
// The real LT chip (the breaker's probe of the review, re-pointed here)
// =============================================================================

/// A row of the real LT layout claiming unsigned `(a0, a_hi) < (b0, b_hi)` is
/// `lt`, with the difference limbs `sub_lo`, `sub_hi` (32-bit values, so every
/// range lookup the row sends is one the range tables answer).
fn lt_row_with(a0: FE, a_hi: u32, b0: FE, b_hi: u32, sub_lo: u32, sub_hi: u32, lt: u64) -> Trace {
    let mut trace = lt::generate_lt_trace(&[]);
    let mut set = |col: usize, value: FE| trace.set_main(0, col, value);
    let half = |word: u32, i: u32| FE::from(u64::from((word >> (16 * i)) & 0xFFFF));
    set(lt_cols::LHS_0, a0);
    set(lt_cols::LHS_1, half(a_hi, 0));
    set(lt_cols::LHS_2, half(a_hi, 1));
    set(lt_cols::RHS_0, b0);
    set(lt_cols::RHS_1, half(b_hi, 0));
    set(lt_cols::RHS_2, half(b_hi, 1));
    set(lt_cols::LT, FE::from(lt));
    set(lt_cols::LHS_SUB_RHS_0, half(sub_lo, 0));
    set(lt_cols::LHS_SUB_RHS_1, half(sub_lo, 1));
    set(lt_cols::LHS_SUB_RHS_2, half(sub_hi, 0));
    set(lt_cols::LHS_SUB_RHS_3, half(sub_hi, 1));
    set(lt_cols::LHS_MSB, FE::from(u64::from(a_hi >> 31)));
    set(lt_cols::RHS_MSB, FE::from(u64::from(b_hi >> 31)));
    set(lt_cols::OUT, FE::from(lt));
    set(lt_cols::MU, FE::one());
    trace
}

/// [`lt_row_with`] for carries `(c0, c1)`: `None` when the difference limbs the
/// two carry constraints force are not 32-bit values.
fn lt_row(a0: FE, a_hi: u32, b0: FE, b_hi: u32, c0: u64, c1: u64) -> Option<Trace> {
    let sub_lo = (a0 - b0 + FE::from(c0 << 32)).canonical_u64();
    let sub_hi = (FE::from(u64::from(a_hi)) - FE::from(u64::from(b_hi)) - FE::from(c0)
        + FE::from(c1 << 32))
    .canonical_u64();
    let (sub_lo, sub_hi) = (u32::try_from(sub_lo).ok()?, u32::try_from(sub_hi).ok()?);
    Some(lt_row_with(a0, a_hi, b0, b_hi, sub_lo, sub_hi, c1))
}

/// Whether the LT chip can answer `out = 1` for unsigned
/// `(a0, a_hi) < (b0, b_hi)`: the two carries are bits, so the four carry
/// choices are every row there is; `out = 1` needs `carry[1] = 1`.
fn lt_accepts(a0: FE, a_hi: u32, b0: FE, b_hi: u32) -> bool {
    let air = busless_air(lt_cols::NUM_COLUMNS, LtConstraints);
    let mut accepted = false;
    for c0 in 0..2 {
        if let Some(trace) = lt_row(a0, a_hi, b0, b_hi, c0, 1) {
            assert!(
                validate_busless(&air, &trace),
                "a row with in-range limbs and bit carries must satisfy LT"
            );
            accepted = true;
        }
    }
    accepted
}

// =============================================================================
// Structure
// =============================================================================

/// ★ The limbs' interactions: 21, no ALU lookup, exactly two `IS_HALF` sends
/// of the limb columns weighted `μ_read + μ_write` (the tokens' own), and every
/// old token's timestamp the derived low word and the row's own `TIMESTAMP_1`;
/// the eight old tokens' other elements and every other interaction are the
/// old form's, one for one. The old form is what `bus_interactions` (the
/// per-table STARK's) gives.
#[test]
fn the_limbs_form_replaces_the_lt_lookup_and_nothing_else() {
    let lt = bus_interactions_for(MemwAlignedOrder::Lt);
    let limbs = bus_interactions_for(MemwAlignedOrder::Limbs);
    assert!(
        lt == memw_aligned::bus_interactions(),
        "the default is LT's"
    );
    assert_eq!((lt.len(), limbs.len()), (20, 21));
    assert_eq!(
        lt.iter().filter(|i| i.bus_id == BusId::Alu as u64).count(),
        1
    );
    assert!(limbs.iter().all(|i| i.bus_id != BusId::Alu as u64));

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);
    let limb_send = |column| {
        BusInteraction::sender(
            BusId::IsHalfword,
            mu_sum.clone(),
            vec![BusValue::linear(vec![LinearTerm::Column {
                coefficient: 1,
                column,
            }])],
        )
    };
    assert!(
        limbs[1] == limb_send(cols::STEP_0),
        "interaction 1: IS_HALF[d0]"
    );
    assert!(
        limbs[2] == limb_send(cols::STEP_1),
        "interaction 2: IS_HALF[d1]"
    );
    let halves: Vec<_> = limbs
        .iter()
        .filter(|i| i.bus_id == BusId::IsHalfword as u64)
        .collect();
    assert_eq!(halves.len(), 3, "the address check and the two limbs");
    assert!(halves[0] == &lt[0], "the address check is the old form's");

    let old_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::TIMESTAMP_0,
        },
        LinearTerm::Column {
            coefficient: -1,
            column: cols::STEP_0,
        },
        LinearTerm::Column {
            coefficient: -(1 << 16),
            column: cols::STEP_1,
        },
        LinearTerm::Constant(-1),
    ]);
    let ts_hi = BusValue::Packed {
        start_column: cols::TIMESTAMP_1,
        packing: Packing::Direct,
    };
    // Past the ALU lookup, the limbs' list is the old one with each old
    // token's timestamp derived.
    let mut old_tokens = 0;
    for (k, (a, b)) in lt[2..].iter().zip(&limbs[3..]).enumerate() {
        if a.bus_id == BusId::Memory as u64 && a.is_sender {
            old_tokens += 1;
            assert_eq!(
                (&b.values[3], &b.values[4]),
                (&old_lo, &ts_hi),
                "old token {k}: the derived low word and the row's own high word"
            );
            let mut same = b.clone();
            same.values[3] = a.values[3].clone();
            same.values[4] = a.values[4].clone();
            assert!(
                &same == a,
                "old token {k}: its other elements are the old form's"
            );
        } else {
            assert!(a == b, "interaction {k} is the old form's");
        }
    }
    assert_eq!(old_tokens, 8, "eight old tokens, one a byte");
    // No column holds an old timestamp: nothing reads columns 25 and 26 but
    // the limb sends and the derived low words.
    for i in &limbs {
        for v in &i.values {
            if let BusValue::Packed { start_column, .. } = v {
                assert!(
                    ![cols::STEP_0, cols::STEP_1].contains(start_column),
                    "a limb column read as a value: {:?}",
                    i.values
                );
            }
        }
    }
}

/// MEMW keeps its eight LT lookups (one an old timestamp, `MEMW-C4…C7`): the
/// receiver of every aligned access the limbs' routing refuses (two high
/// words, or not later) and of every unaligned or mixed-timestamp access, with
/// the full comparison. Its 26 interactions are today's.
#[test]
fn memw_keeps_its_eight_lt_lookups() {
    use crate::tables::types::alu_op;
    let memw = crate::tables::memw::bus_interactions();
    assert_eq!(memw.len(), 26);
    let lt = memw
        .iter()
        .filter(|i| {
            i.bus_id == BusId::Alu as u64
                && i.is_sender
                && i.values.get(2) == Some(&BusValue::constant(alu_op::LT as u64))
        })
        .count();
    assert_eq!(lt, 8, "one LT lookup an old timestamp");
}

/// Each form is its own AIR prototype: built in any order in one process, the
/// limbs' AIR has 21 interactions and LT's 20, both named `MEMW_A`, with the
/// same columns and constraints.
#[test]
fn the_two_forms_never_share_an_air_prototype() {
    use stark::traits::AIR;
    let opts = stark::proof::options::ProofOptions::default_test_options();
    for order in [
        MemwAlignedOrder::Limbs,
        MemwAlignedOrder::Lt,
        MemwAlignedOrder::Limbs,
        MemwAlignedOrder::Lt,
    ] {
        let air = crate::test_utils::create_memw_aligned_air_for(&opts, order);
        assert!(
            air.bus_interactions() == bus_interactions_for(order).as_slice(),
            "{order:?}"
        );
        assert_eq!(air.name(), "MEMW_A", "{order:?}");
        assert_eq!(air.trace_layout().0, cols::NUM_COLUMNS, "{order:?}");
    }
}

// =============================================================================
// The lemma
// =============================================================================

/// ★ On one high word the real LT chip and the limbs' AIR accept the same
/// pairs: the steps `1..=2^32` as a field difference of the low words,
/// whatever field elements the low words are. LT range-checks neither low
/// word; the limbs' row has no old low word to check. So the limbs prove what
/// MEMW_A's LT lookup proves on the clock's timestamps, pair for pair.
#[test]
fn on_one_high_word_the_limbs_accept_exactly_what_lt_accepts() {
    let bases = [
        FE::from(0u64),
        FE::from(4u64),
        FE::from((1u64 << 32) - 1),
        FE::from(1u64 << 32),
        FE::from(1u64 << 63),
        FE::from(P - 5),
    ];
    let deltas = [
        0u64,
        1,
        2,
        0xFFFF,
        0x1_0000,
        0x1_0001,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        1 << 33,
        P - (1 << 32) - 1,
        P - (1 << 32),
        P - 1,
    ];
    for hi in [0u32, 1, 0x8000_0000, u32::MAX] {
        for old_lo in bases {
            for delta in deltas {
                let ts_lo = old_lo + FE::from(delta);
                let window = (1..=1u64 << 32).contains(&delta);
                assert_eq!(
                    lt_accepts(old_lo, hi, ts_lo, hi),
                    window,
                    "LT, hi {hi:#x}, old {old_lo:?}, delta {delta:#x}"
                );
                assert_eq!(
                    limbs_accept(old_lo, u64::from(hi), ts_lo, u64::from(hi)),
                    window,
                    "limbs, hi {hi:#x}, old {old_lo:?}, delta {delta:#x}"
                );
            }
        }
    }
}

/// On two high words the limbs' row cannot claim the access at all: every
/// old token carries the row's own high word, whatever its limbs. LT orders
/// such pairs ((2^32 − 1, 0) before (0, 1)); the routing sends them to MEMW,
/// which keeps LT.
#[test]
fn an_old_token_always_carries_the_rows_own_high_word() {
    let air = bus_interactions_for(MemwAlignedOrder::Limbs);
    for ts_hi in [0u64, 1, 7, u64::from(u32::MAX)] {
        for (d0, d1) in [(0, 0), (5, 9), (HALF - 1, HALF - 1)] {
            let row = limbs_row(
                FE::from(1u64 << 20),
                ts_hi,
                FE::from(d0),
                FE::from(d1),
                1,
                0,
            );
            let olds = old_timestamps(&air, &row);
            assert_eq!(olds.len(), 8);
            assert!(olds.iter().all(|(_, hi)| *hi == FE::from(ts_hi)));
        }
    }
    let fe = |v: u64| FE::from(v);
    // LT orders across high words; no limbs' row claims the old token (lo
    // 2^32 − 1, hi 0) at a timestamp of high word 1.
    assert!(lt_accepts(fe((1 << 32) - 1), 0, fe(0), 1));
    assert!(!limbs_accept(fe((1 << 32) - 1), 0, fe(0), 1));
    assert!(!limbs_accept(fe(0), 1, fe(10), 0));
}

// =============================================================================
// Forgeries and the boundary
// =============================================================================

/// ★ Forgeries on the built AIR, each refused, each beside a control that
/// passes: the old timestamp equal to the new one; later than it; a step past
/// 2^32; a limb of 2^16 (the honest split of the same step passes); and the
/// boundary: a step of exactly 1, 2^16, 2^16 + 1 and 2^32 (the window's
/// end), a nonzero high word shared by both.
#[test]
fn forged_orders_are_refused_and_the_boundary_holds() {
    let fe = FE::from;
    // (old_lo, ts_lo, hi, accepted)
    let cases: [(u64, u64, u64, bool); 12] = [
        (100, 100, 0, false),                      // equal timestamps
        (100, 101, 0, true),                       // control: one step
        (105, 100, 0, false),                      // old later than new
        (95, 100, 0, true),                        // control
        (100, 100 + (1 << 16), 0, true),           // a step of 2^16
        (100, 101 + (1 << 16), 0, true),           // 2^16 + 1: the high limb at 1
        (0, 1 << 32, 0, true), // a step of exactly 2^32 (lo 2^32 as a field element)
        (0, (1 << 32) + 1, 0, false), // past the window
        (7, 9, 3, true),       // a nonzero high word, shared
        (9, 7, 3, false),      // ... and backwards on it
        (0, (1 << 32) - 1, u32::MAX as u64, true), // the widest step a 32-bit low word makes
        (1, (1 << 32) - 1, 0, true), // HALT's widest: old 1, new 2^32 − 1
    ];
    for (old_lo, ts_lo, hi, want) in cases {
        assert_eq!(
            limbs_accept(fe(old_lo), hi, fe(ts_lo), hi),
            want,
            "old {old_lo:#x} → {ts_lo:#x} at high word {hi}"
        );
        assert_eq!(
            lt_accepts(fe(old_lo), hi as u32, fe(ts_lo), hi as u32),
            want,
            "LT: old {old_lo:#x} → {ts_lo:#x} at high word {hi}"
        );
    }
    // A limb of 2^16: the token is the one the honest split (0, 1) gives, and
    // only the half-word lookup refuses it.
    let air = bus_interactions_for(MemwAlignedOrder::Limbs);
    let old = (fe(100), fe(0));
    let ts = fe(101 + HALF);
    assert!(row_accepts(&air, old, ts, 0, fe(0), fe(1), (1, 0)));
    let row = limbs_row(ts, 0, fe(HALF), fe(0), 1, 0);
    assert!(old_timestamps(&air, &row).iter().all(|t| *t == old));
    assert!(
        !halves_in_range(&air, &row),
        "a limb of 2^16 is no half word"
    );
    // Writes are checked as reads are: the limbs go at μ_read + μ_write.
    assert!(row_accepts(&air, old, ts, 0, fe(0), fe(1), (0, 1)));
    let backwards = fe(P - 6); // old 105 → new 100: the forced step − 1
    assert!(!row_accepts(
        &air,
        (fe(105), fe(0)),
        fe(100),
        0,
        backwards,
        fe(0),
        (0, 1)
    ));
}

/// An old low word of `p − k` passes the row (the identity is over the field:
/// 10 − (p − 3) − 1 = 12), and LT's lookup took it too; what refuses it is the
/// Memory bus. Every token a producer makes has a timestamp below 2^32 (PAGE
/// at 0, REGISTER at 1, the CPU below 2^32 − 1 by the row bound, HALT at
/// 2^32 − 1), so a forged history that consumes `(…, p − 3, …)` leaves it
/// unmatched; the honest history at the same addresses balances.
#[test]
fn an_old_low_word_no_producer_makes_is_left_on_the_memory_bus() {
    let fe = FE::from;
    let air = bus_interactions_for(MemwAlignedOrder::Limbs);
    // The row accepts the field-wrapped pair, as LT does.
    assert!(limbs_accept(fe(P - 3), 0, fe(10), 0));
    assert!(lt_accepts(fe(P - 3), 0, fe(10), 0));

    // Net Memory-bus tuples of `rows` with PAGE's two tokens of the address
    // (at timestamp 0 and at `last`) on the other side.
    let net = |rows: &[Vec<FE>], last: u64| {
        let mut net: HashMap<Vec<u64>, i64> = HashMap::new();
        for row in rows {
            for (bus, sender, tuple) in messages(&air, row) {
                if bus == BusId::Memory {
                    let key = tuple.iter().map(FE::canonical_u64).collect();
                    *net.entry(key).or_default() += if sender { 1 } else { -1 };
                }
            }
        }
        for byte in 0..8u64 {
            // The page hands out each byte's token at 0 and takes it back at
            // the last access's timestamp.
            *net.entry(vec![0, 0x1000 + byte, 0, 0, 0, byte + 1])
                .or_default() -= 1;
            *net.entry(vec![0, 0x1000 + byte, 0, last, 0, byte + 1])
                .or_default() += 1;
        }
        net.retain(|_, m| *m != 0);
        net
    };
    // Honest: 0 → 8 → 10, limbs 7 and 1.
    let honest = [
        limbs_row(fe(8), 0, fe(7), fe(0), 1, 0),
        limbs_row(fe(10), 0, fe(1), fe(0), 1, 0),
    ];
    assert!(net(&honest, 10).is_empty(), "the honest history balances");
    // Forged: the second access claims old p − 3 (limbs 12, 0).
    let forged = [
        limbs_row(fe(8), 0, fe(7), fe(0), 1, 0),
        limbs_row(fe(10), 0, fe(12), fe(0), 1, 0),
    ];
    let left = net(&forged, 10);
    assert!(!left.is_empty(), "the forged history does not balance");
    assert!(
        left.keys().any(|t| t[3] == P - 3),
        "it consumes a token at p − 3: {left:?}"
    );
    assert!(
        left.keys().all(|t| t[3] == P - 3 || t[3] < 1 << 32),
        "what is left is that token and what it should have consumed"
    );
}

// =============================================================================
// Mutants: each piece of the limbs' list refuses its forgery
// =============================================================================

/// The limbs' list with `edit` applied: a mutant of one piece.
fn mutant(edit: impl FnOnce(&mut Vec<BusInteraction>)) -> Vec<BusInteraction> {
    let mut list = bus_interactions_for(MemwAlignedOrder::Limbs);
    edit(&mut list);
    list
}

/// Every old token's timestamp low word replaced by `lo`.
fn with_old_lo(list: &mut [BusInteraction], lo: &BusValue) {
    for i in list.iter_mut() {
        if i.bus_id == BusId::Memory as u64 && i.is_sender {
            i.values[3] = lo.clone();
        }
    }
}

/// ★ One mutant per piece, each letting through the forgery the real list
/// refuses (and the real list refusing it):
/// - the identity without its `− 1`: an old timestamp equal to the new one;
/// - without the high limb's lookup: an old timestamp later than the new one
///   (the high limb a field element);
/// - without the low limb's lookup: the same through the low limb;
/// - the limbs weighted by `μ_read` alone: a write row checks nothing;
/// - the old high word a free column (a 30th): old (2^32 + 0) before new 10;
/// - the high limb's weight 2^17: a step past 2^32, outside LT's window.
#[test]
fn each_piece_of_the_limbs_refuses_its_forgery() {
    let fe = FE::from;
    let real = bus_interactions_for(MemwAlignedOrder::Limbs);
    let lo_without = |constant: Option<i64>, high: i64| {
        let mut terms = vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::STEP_0,
            },
            LinearTerm::Column {
                coefficient: high,
                column: cols::STEP_1,
            },
        ];
        terms.extend(constant.map(LinearTerm::Constant));
        BusValue::linear(terms)
    };

    // 1. No `− 1`: old = new = 100 with limbs (0, 0).
    let no_minus_one = mutant(|l| with_old_lo(l, &lo_without(None, -(1 << 16))));
    let equal = ((fe(100), fe(0)), fe(100), fe(0), fe(0));
    assert!(row_accepts(
        &no_minus_one,
        equal.0,
        equal.1,
        0,
        equal.2,
        equal.3,
        (1, 0)
    ));
    assert!(!limbs_accept(fe(100), 0, fe(100), 0));

    // 2. No high limb lookup: old 105 → new 100 needs d0 + 2^16·d1 = p − 6;
    //    with d0 = 0, d1 = (p − 6)/2^16 in the field.
    let no_high = mutant(|l| {
        l.remove(2);
    });
    let inv = FE::from(HALF).inv().expect("2^16 is invertible");
    let d1 = fe(P - 6) * inv;
    assert!(row_accepts(
        &no_high,
        (fe(105), fe(0)),
        fe(100),
        0,
        fe(0),
        d1,
        (1, 0)
    ));
    assert!(!row_accepts(
        &real,
        (fe(105), fe(0)),
        fe(100),
        0,
        fe(0),
        d1,
        (1, 0)
    ));
    assert!(!limbs_accept(fe(105), 0, fe(100), 0));

    // 3. No low limb lookup: the same step through d0 = p − 6.
    let no_low = mutant(|l| {
        l.remove(1);
    });
    assert!(row_accepts(
        &no_low,
        (fe(105), fe(0)),
        fe(100),
        0,
        fe(P - 6),
        fe(0),
        (1, 0)
    ));
    assert!(!row_accepts(
        &real,
        (fe(105), fe(0)),
        fe(100),
        0,
        fe(P - 6),
        fe(0),
        (1, 0)
    ));

    // 4. The limbs at μ_read only: a write row's backwards step.
    let read_only = mutant(|l| {
        l[1].multiplicity = Multiplicity::Column(cols::MU_READ);
        l[2].multiplicity = Multiplicity::Column(cols::MU_READ);
    });
    assert!(row_accepts(
        &read_only,
        (fe(105), fe(0)),
        fe(100),
        0,
        fe(P - 6),
        fe(0),
        (0, 1)
    ));
    assert!(!row_accepts(
        &real,
        (fe(105), fe(0)),
        fe(100),
        0,
        fe(P - 6),
        fe(0),
        (0, 1)
    ));

    // 5. The old high word a free column (30): old (lo 0, hi 1) = 2^32 before
    //    new (lo 10, hi 0); the low words alone pass (10 − 0 − 1 = 9).
    let free_high = mutant(|l| {
        for i in l.iter_mut() {
            if i.bus_id == BusId::Memory as u64 && i.is_sender {
                i.values[4] = BusValue::Packed {
                    start_column: cols::NUM_COLUMNS,
                    packing: Packing::Direct,
                };
            }
        }
    });
    let mut row = limbs_row(fe(10), 0, fe(9), fe(0), 1, 0);
    row.push(fe(1));
    let olds = old_timestamps(&free_high, &row);
    assert!(
        olds.len() == 8 && olds.iter().all(|t| *t == (fe(0), fe(1))),
        "the mutant claims old 2^32"
    );
    assert!(halves_in_range(&free_high, &row));
    assert!(
        !limbs_accept(fe(0), 1, fe(10), 0),
        "the real row cannot claim it"
    );

    // 6. The high limb at 2^17: step 2^32 + 2, past LT's window, passes.
    let wide = mutant(|l| with_old_lo(l, &lo_without(Some(-1), -(1 << 17))));
    // ts − old − 1 = 2^32 + 1 = 1 + 2^17·(2^15).
    let (old, ts) = (fe(0), fe((1 << 32) + 2));
    assert!(row_accepts(
        &wide,
        (old, fe(0)),
        ts,
        0,
        fe(1),
        fe(1 << 15),
        (1, 0)
    ));
    assert!(!lt_accepts(old, 0, ts, 0), "LT refuses it");
    assert!(!limbs_accept(old, 0, ts, 0), "the real row refuses it");
}

// =============================================================================
// The prover's side
// =============================================================================

/// An aligned 8-byte memory op from `old` to `ts`.
fn aligned_op(old: u64, ts: u64) -> MemwOperation {
    MemwOperation::new(false, 0x2000, [1, 2, 3, 4, 5, 6, 7, 8], ts, 8, true)
        .with_old([9; 8], [old; 8])
}

/// ★ What the routing admits under the limbs: a step on one high word, the
/// old timestamp the earlier; anything else is MEMW's (refused here, never
/// wrapped). Under LT every aligned access is MEMW_A's, as before.
#[test]
fn the_routing_gives_the_limbs_only_a_step_that_fits() {
    use crate::tables::trace_builder::route_for_test as route;
    for (old, ts, fits) in [
        (1u64, 4u64, true),
        (4, 4 + HALF, true),
        (1, (1 << 32) - 1, true),             // HALT's widest
        ((7 << 32) + 3, (7 << 32) + 9, true), // a shared nonzero high word
        (5, 5 + (1 << 32), false),            // a step of exactly 2^32: two high words
        ((1 << 32) - 1, 1 << 32, false),      // one step, across the word
        (9, 9, false),                        // not later
        (10, 9, false),                       // backwards
    ] {
        assert_eq!(step_fits_limbs(old, ts), fits, "{old:#x} → {ts:#x}");
        let op = aligned_op(old, ts);
        assert_eq!(
            route(&op, MemwAlignedOrder::Limbs),
            if fits { "MEMW_A" } else { "MEMW" }
        );
        assert_eq!(
            route(&op, MemwAlignedOrder::Lt),
            "MEMW_A",
            "LT takes any order"
        );
    }
    // Registers stay MEMW_R's in either form; misaligned stays MEMW's.
    let reg = MemwOperation::new(true, 2 * 5, [1, 0, 0, 0, 0, 0, 0, 0], 40, 2, true)
        .with_old([0; 8], [36, 36, 0, 0, 0, 0, 0, 0]);
    assert_eq!(route(&reg, MemwAlignedOrder::Limbs), "MEMW_R");
    let odd = MemwOperation::new(false, 0x2001, [1; 8], 40, 4, true).with_old([0; 8], [3; 8]);
    assert_eq!(route(&odd, MemwAlignedOrder::Limbs), "MEMW");
}

/// ★ A limbs' row lays its step's limbs out in columns 25 and 26 (the old form
/// its old timestamp), BITWISE counts the address check and the two limbs
/// (the old form the address check alone), and LT takes no comparison of it
/// (the old form one): what the walk's row is, end to end, for the steps at
/// the window's ends.
#[test]
fn a_limbs_row_lays_out_counts_and_asks_lt_as_its_form_says() {
    for (old, ts) in [(1u64, 2u64), (4, 4 + HALF), (1, (1 << 32) - 1), (36, 100)] {
        let op = aligned_op(old, ts);
        let lt = AlignedRow::from_memw(&op, MemwAlignedOrder::Lt);
        let limbs = AlignedRow::from_memw(&op, MemwAlignedOrder::Limbs);
        let (d0, d1) = step_limbs(old, ts);
        assert_eq!(u64::from(d0) + (u64::from(d1) << 16), ts - old - 1);

        let lt_trace = memw_aligned::generate_memw_aligned_trace(&[lt]);
        let limbs_trace = memw_aligned::generate_memw_aligned_trace(&[limbs]);
        assert_eq!(
            *lt_trace.get_main(0, cols::OLD_TIMESTAMP_0),
            FE::from(old & 0xFFFF_FFFF)
        );
        assert_eq!(
            *lt_trace.get_main(0, cols::OLD_TIMESTAMP_1),
            FE::from(old >> 32)
        );
        assert_eq!(
            *limbs_trace.get_main(0, cols::STEP_0),
            FE::from(u64::from(d0))
        );
        assert_eq!(
            *limbs_trace.get_main(0, cols::STEP_1),
            FE::from(u64::from(d1))
        );
        for c in (0..cols::NUM_COLUMNS).filter(|c| ![cols::STEP_0, cols::STEP_1].contains(c)) {
            assert_eq!(
                lt_trace.get_main(0, c),
                limbs_trace.get_main(0, c),
                "column {c}"
            );
        }
        // The laid-out row's own interactions claim the op's old timestamp.
        let row: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *limbs_trace.get_main(0, c))
            .collect();
        let air = bus_interactions_for(MemwAlignedOrder::Limbs);
        let olds = old_timestamps(&air, &row);
        assert_eq!(olds.len(), 8);
        assert!(
            olds.iter()
                .all(|t| *t == (FE::from(old & 0xFFFF_FFFF), FE::from(old >> 32)))
        );
        assert!(halves_in_range(&air, &row));

        let counted = crate::tables::trace_builder::bitwise_of_memw_aligned_for_test(&[limbs]);
        let half = |v: u64| {
            BitwiseOperation::halfword(BitwiseOperationType::IsHalf, v as u8, (v >> 8) as u8)
        };
        assert_eq!(
            counted,
            vec![half(0x2000 + 7), half(u64::from(d0)), half(u64::from(d1))]
        );
        assert_eq!(
            crate::tables::trace_builder::bitwise_of_memw_aligned_for_test(&[lt]),
            vec![half(0x2000 + 7)]
        );
        assert_eq!(lt.lt_pair(), Some((old, ts)));
        assert_eq!(limbs.lt_pair(), None);
    }
}

// =============================================================================
// The range bus the limbs rest on
// =============================================================================

/// Every `IS_HALF` sender of the production block's tables, as
/// `(table, multiplicity)`: the block AIR set under the pinned format, one
/// table of each family.
fn half_word_senders() -> Vec<(String, String)> {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 1,
        memw: 1,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        copy: 1,
        blake3: 1,
        is_wh: 1,
        ecrec: 1,
    };
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        0,
        1 << 20,
    )
    .expect("pages");
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![1 << 10], &[]).expect("the AIR inputs");
    let format = crate::block_whir::BlockFormat::pinned();
    let airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
    );
    let mut senders: Vec<(String, String)> = airs
        .air_refs()
        .iter()
        .flat_map(|air| {
            let family = air.name().split('[').next().unwrap_or("").to_string();
            air.bus_interactions()
                .iter()
                .filter(|i| i.bus_id == BusId::IsHalfword as u64 && i.is_sender)
                .map(|i| (family.clone(), format!("{:?}", i.multiplicity)))
                .collect::<Vec<_>>()
        })
        .collect();
    senders.sort();
    senders.dedup();
    senders
}

/// ★ The range bus the limbs rest on takes no negative weight: every `IS_HALF`
/// sender of the production block's tables, by table and multiplicity, is
/// the review's census (a constant 1, a column or a sum constrained to a bit,
/// or a multiplicity bounded by a lookup of weight 1). A new sender, or a
/// sender's weight changed, fails here, to be classified before it lands: the
/// limbs' 16-bit bound is only as good as this bus's.
#[test]
fn every_half_word_sender_weighs_one_a_bit_or_a_bounded_count() {
    let senders = half_word_senders();
    for (table, m) in &senders {
        println!("IS_HALF SENDER {table}: {m}");
    }
    let mu_sum = format!("{:?}", Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE));
    assert!(
        senders.contains(&("MEMW_A".to_string(), mu_sum)),
        "MEMW_A's limbs at μ_read + μ_write"
    );
    let pinned: Vec<(String, String)> = PINNED_HALF_WORD_SENDERS
        .iter()
        .map(|(t, m, _)| (t.to_string(), m.to_string()))
        .collect();
    assert_eq!(senders, pinned, "the IS_HALF senders moved: classify them");
}

/// [`every_half_word_sender_weighs_one_a_bit_or_a_bounded_count`]'s census:
/// `(table, multiplicity, why it is never negative)`, read in each table's
/// constraints and interactions.
const PINNED_HALF_WORD_SENDERS: &[(&str, &str, &str)] = &[
    ("BLAKE3", "Column(3218)", "a bit column"),
    ("BLAKE3", "Column(3219)", "a bit column"),
    ("BLAKE3", "Column(3220)", "a bit column"),
    ("BLAKE3", "Column(3221)", "a bit column"),
    ("BLAKE3", "Column(3222)", "a bit column"),
    (
        "BRANCH",
        "Column(13)",
        "μ, a byte by ARE_BYTES[μ, 0] at weight 1",
    ),
    ("COPY", "Column(36)", "a bit column"),
    ("COPY", "Column(38)", "a bit column"),
    ("CPU", "One", "the constant 1"),
    ("CPU32", "Column(37)", "a bit column"),
    (
        "DVRM",
        "Sum(32, 33)",
        "μ_q + μ_r, each a byte by ARE_BYTES at weight 1",
    ),
    ("ECDAS", "Column(520)", "a bit column"),
    ("ECREC", "Column(10)", "a bit column"),
    ("ECSM", "Column(666)", "a bit column"),
    ("EQ", "Column(11)", "a bit column"),
    ("IS_WH", "Column(3)", "μ, 20 bits by IS_B20 at weight 1"),
    ("KECCAK", "Column(486)", "a bit column"),
    ("KECCAK", "Column(487)", "a bit column"),
    ("KECCAK", "Column(489)", "a bit column"),
    (
        "LT",
        "Column(16)",
        "μ, a byte by ARE_BYTES[μ, 0] at weight 1",
    ),
    (
        "MEMW_A",
        "Sum(27, 28)",
        "μ_read + μ_write, a bit (IS_BIT on the sum)",
    ),
    (
        "MEMW_R",
        "Sum(8, 9)",
        "μ_read + μ_write, a bit (IS_BIT on the sum)",
    ),
    (
        "MUL",
        "Sum(24, 25)",
        "μ_lo + μ_hi, each a byte by ARE_BYTES at weight 1",
    ),
    ("SHIFT", "Column(25)", "a bit column"),
];
