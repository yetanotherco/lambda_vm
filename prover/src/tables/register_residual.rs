//! The residual register table (MEMW_R's slot under the in-row register
//! format): the register accesses a CPU row does not check itself, under
//! [`RegisterFormat::InRow`](super::register::RegisterFormat::InRow).
//!
//! In that format each CPU row checks its own `rs1`, `rs2` and `rd` accesses on
//! the [`BusId::Reg`] token bus (one token a register access:
//! `(register, timestamp, value_lo, value_hi)`). This table takes the rest:
//!
//! - the register accesses of word (`*W`) instructions (CPU32 sends them), of
//!   the ECALL chips and of HALT, each a MEMW message this table receives as MEMW_R
//!   does today, its step proved by two 16-bit limbs, so any step in
//!   `[1, 2^32]`;
//! - keep-alives: a value-preserving consume and re-produce of a register's
//!   token, inserted where a CPU row's step would pass the CPU's 16-bit range
//!   check. A keep-alive has no requester. `MU_KEEP·(OLD − VAL) = 0` makes it
//!   preserve the value, and the verifier caps the residual table's rows
//!   ([`MAX_ROWS_TOTAL`]), so keep-alives cannot advance a register's clock
//!   past the field.
//!
//! ## Columns (12)
//!
//! `REG` (register index), `TS_0`, `TS_1` (timestamp words), `VAL_0`, `VAL_1`,
//! `OLD_0`, `OLD_1`, `D0`, `D1` (the step less one, in 16-bit limbs:
//! `old_ts = ts − 1 − D0 − 2^16·D1`), `MU_READ`, `MU_WRITE`, `MU_KEEP`.
//!
//! ## Interactions (6), μ = `MU_READ + MU_WRITE + MU_KEEP`
//!
//! - REG send `(REG, old_ts, OLD)` and REG receive `(REG, ts, VAL)` on μ;
//! - IS_HALF[`D0`] and IS_HALF[`D1`] on μ;
//! - MEMW read receiver on `MU_READ`, MEMW write receiver on `MU_WRITE`, the
//!   messages MEMW_R receives today, at address `2·REG`.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::memw_register::RegRow;
use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::emit_is_bit;

pub mod cols {
    /// Register index: x0–x31, x254 (the commit index), x255 (the PC).
    pub const REG: usize = 0;
    pub const TS_0: usize = 1;
    pub const TS_1: usize = 2;
    pub const VAL_0: usize = 3;
    pub const VAL_1: usize = 4;
    pub const OLD_0: usize = 5;
    pub const OLD_1: usize = 6;
    /// `ts − old_ts − 1`, low 16 bits.
    pub const D0: usize = 7;
    /// `ts − old_ts − 1`, high 16 bits.
    pub const D1: usize = 8;
    pub const MU_READ: usize = 9;
    pub const MU_WRITE: usize = 10;
    pub const MU_KEEP: usize = 11;
    pub const NUM_COLUMNS: usize = 12;
}

/// The most residual register rows a block may hold, over all its residual tables: a verifier
/// constant. Every residual register row advances one register's clock by at most `2^32`;
/// with at most `2^28` of them, and each CPU slot advancing by at most `2^16`
/// over at most `3·2^30` slots, a register's timestamps stay below
/// `2^61 ≪ p`, so the field's order is the integers' and no chain of tokens
/// can wrap around to a timestamp it already passed.
pub const MAX_ROWS_TOTAL: u64 = 1 << 28;

/// `2^32` as a linear-term coefficient.
const TWO_32: i64 = 1 << 32;

/// The row's timestamp as one field element, `TS_0 + 2^32·TS_1`.
fn ts_terms() -> Vec<LinearTerm> {
    vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::TS_0,
        },
        LinearTerm::Column {
            coefficient: TWO_32,
            column: cols::TS_1,
        },
    ]
}

fn column(c: usize) -> BusValue {
    BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    }
}

/// The table's interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = Multiplicity::Sum3(cols::MU_READ, cols::MU_WRITE, cols::MU_KEEP);
    let mut old_ts = ts_terms();
    old_ts.extend([
        LinearTerm::Constant(-1),
        LinearTerm::Column {
            coefficient: -1,
            column: cols::D0,
        },
        LinearTerm::Column {
            coefficient: -(1 << 16),
            column: cols::D1,
        },
    ]);
    let address = || {
        BusValue::linear(vec![LinearTerm::Column {
            coefficient: 2,
            column: cols::REG,
        }])
    };
    let ts_words = || [column(cols::TS_0), column(cols::TS_1)];
    let mut read = vec![column(cols::OLD_0), column(cols::OLD_1)];
    read.extend((0..6).map(|_| BusValue::constant(0)));
    let mut write = vec![
        BusValue::constant(1),
        address(),
        BusValue::constant(0),
        column(cols::VAL_0),
        column(cols::VAL_1),
    ];
    write.extend((0..6).map(|_| BusValue::constant(0)));
    write.extend(ts_words());
    write.extend([
        BusValue::constant(1),
        BusValue::constant(0),
        BusValue::constant(0),
    ]);
    // The read message: `old[8]`, then the write message's 16 elements.
    read.extend(write.iter().cloned());
    vec![
        BusInteraction::sender(
            BusId::Reg,
            mu.clone(),
            vec![
                column(cols::REG),
                BusValue::linear(old_ts),
                column(cols::OLD_0),
                column(cols::OLD_1),
            ],
        ),
        BusInteraction::receiver(
            BusId::Reg,
            mu.clone(),
            vec![
                column(cols::REG),
                BusValue::linear(ts_terms()),
                column(cols::VAL_0),
                column(cols::VAL_1),
            ],
        ),
        BusInteraction::sender(BusId::IsHalfword, mu.clone(), vec![column(cols::D0)]),
        BusInteraction::sender(BusId::IsHalfword, mu, vec![column(cols::D1)]),
        BusInteraction::receiver(BusId::Memw, Multiplicity::Column(cols::MU_READ), read),
        BusInteraction::receiver(BusId::Memw, Multiplicity::Column(cols::MU_WRITE), write),
    ]
}

/// The table's constraints: `IS_BIT` on the three flags (idx 0–2) and on their sum
/// (idx 3), and a keep-alive preserves the value, `MU_KEEP·(OLD_i − VAL_i) = 0`
/// (idx 4, 5).
#[derive(Clone, Copy)]
pub struct ResidualConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for ResidualConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU_READ, None);
        emit_is_bit(b, 1, cols::MU_WRITE, None);
        emit_is_bit(b, 2, cols::MU_KEEP, None);
        let one = b.one();
        let mu = b.main(0, cols::MU_READ) + b.main(0, cols::MU_WRITE) + b.main(0, cols::MU_KEEP);
        b.emit_base(3, mu.clone() * (one - mu));
        for (idx, (old, val)) in [(cols::OLD_0, cols::VAL_0), (cols::OLD_1, cols::VAL_1)]
            .into_iter()
            .enumerate()
        {
            let keep = b.main(0, cols::MU_KEEP);
            let diff = b.main(0, old) - b.main(0, val);
            b.emit_base(4 + idx, keep * diff);
        }
    }
}

/// A row's step less one, `ts − old_ts − 1`, as `(D0, D1)`.
fn step_limbs(row: &RegRow) -> (u16, u16) {
    let d = row.step() - 1;
    debug_assert!(
        d < 1 << 32,
        "a residual register step fits 32 bits: {row:?}"
    );
    (d as u16, (d >> 16) as u16)
}

/// The widths residual register traces needed so far in this process (`tables::gpack`).
static WIDTHS: WidthHint = WidthHint::new();

/// The table's trace from its rows (in order, then zero rows to a power of two of
/// at least four), in `form`.
pub(crate) fn generate_residual_trace_segments_as(
    segments: &[&[RegRow]],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let len: usize = segments.iter().map(|s| s.len()).sum();
    let num_rows = len.next_power_of_two().max(4);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, r) in segments.iter().flat_map(|s| s.iter()).enumerate() {
            let (d0, d1) = step_limbs(r);
            table.set_u64(row, cols::REG, u64::from(r.register()));
            table.set_dword_wl(row, cols::TS_0, r.timestamp());
            let (val, old) = r.words();
            table.set_u64(row, cols::VAL_0, u64::from(val[0]));
            table.set_u64(row, cols::VAL_1, u64::from(val[1]));
            table.set_u64(row, cols::OLD_0, u64::from(old[0]));
            table.set_u64(row, cols::OLD_1, u64::from(old[1]));
            table.set_u64(row, cols::D0, u64::from(d0));
            table.set_u64(row, cols::D1, u64::from(d1));
            let keep = r.is_keep();
            table.set_bool(row, cols::MU_READ, !keep && r.is_read());
            table.set_bool(row, cols::MU_WRITE, !keep && !r.is_read());
            table.set_bool(row, cols::MU_KEEP, keep);
        }
    })
}

/// The two IS_HALF lookups each residual register row sends (`D0`, `D1`), counted into
/// `hist`.
pub(crate) fn count_bitwise_from_residual(rows: &[RegRow], hist: &mut BitwiseHistogram) {
    for r in rows {
        let (d0, d1) = step_limbs(r);
        for d in [d0, d1] {
            hist.bump(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                d as u8,
                (d >> 8) as u8,
            ));
        }
    }
}
