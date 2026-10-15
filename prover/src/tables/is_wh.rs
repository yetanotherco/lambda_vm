//! IS_WH table — range check for the top 48 bits of a 64-bit value.
//!
//! Chips that read or write contiguous memory hold one address per access, and
//! those addresses usually differ only in their lowest limb. Instead of range
//! checking all 64 bits locally, a caller keeps the address as a `DWordWHH`
//! (`[half_lo, half_mid, word_hi]`), checks `half_lo` itself and sends the
//! `(word_hi, half_mid)` pair here on the `IsWh` bus (spec `is_wh.toml`). The
//! `IS_WHH` template (spec `is_whh.toml`) is that pair of sends, see [`is_whh`].
//!
//! One row per distinct `(word, half)` pair, with `μ` counting its sends.
//!
//! ## Columns
//! - `value[0]`: Half — the `Half` of the pair (bits 16..32 of the address)
//! - `value[1]`, `value[2]`: Half — the `Word` of the pair as two halves
//!   (bits 32..48 and 48..64)
//! - `μ`: B20 — number of sends of this pair
//!
//! ## Bus Interactions
//! - Sender: IS_B20[μ] (multiplicity 1, bounds μ)
//! - Sender: IS_HALFWORD (×3, one per `value` limb, multiplicity μ)
//! - Receiver: IS_WH[value[1] + 2^16·value[2], value[0]] (multiplicity μ)
//!
//! ## μ bound
//! `μ` lives in the field, so `μ = −1` would turn a row from receiver into
//! sender: it could cancel a forged `IS_WH` send while also receiving the row's
//! own `IS_HALFWORD` checks. The `IS_B20[μ]` send (multiplicity 1, so it is not
//! itself flipped by `μ`) pins `μ ∈ [0, 2^20)`. A pair sent more than [`MU_MAX`]
//! times is split over several rows (see [`dedup_is_wh_rows`]). Padding rows send
//! `IS_B20[0]`, which the BITWISE collector counts (see
//! [`collect_bitwise_from_is_wh`]).
//!
//! The table has no polynomial constraints.

use std::collections::BTreeMap;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable};

/// Column definitions for the IS_WH table.
pub mod cols {
    /// value[0]: Half — the `Half` of the `(Word, Half)` pair.
    pub const VALUE_0: usize = 0;
    /// value[1]: Half — low half of the `Word`.
    pub const VALUE_1: usize = 1;
    /// value[2]: Half — high half of the `Word`.
    pub const VALUE_2: usize = 2;
    /// μ: B20 — number of `IS_WH` sends of this pair.
    pub const MU: usize = 3;

    pub const NUM_COLUMNS: usize = 4;
}

/// Largest multiplicity a single row may carry: `μ` is a `B20`.
pub const MU_MAX: u64 = (1 << 20) - 1;

/// A `(Word, Half)` pair as it travels on the `IsWh` bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IsWhKey {
    /// Bits 32..64 of the checked value.
    pub word: u32,
    /// Bits 16..32 of the checked value.
    pub half: u16,
}

impl IsWhKey {
    /// The pair a `DWordWHH` holding `value` sends to `IS_WH`.
    pub fn of_dword(value: u64) -> Self {
        Self {
            word: (value >> 32) as u32,
            half: ((value >> 16) & 0xFFFF) as u16,
        }
    }
}

/// Deduplicates `IS_WH` sends into trace rows: `(key) -> μ`, splitting a key
/// over several rows when its count exceeds [`MU_MAX`]. Rows come out sorted by
/// key so the trace is deterministic. Shared by trace generation and the
/// BITWISE collector so the per-row lookups they count cannot drift apart.
pub fn dedup_is_wh_rows(keys: &[IsWhKey]) -> Vec<(IsWhKey, u64)> {
    let mut counts: BTreeMap<IsWhKey, u64> = BTreeMap::new();
    for key in keys {
        *counts.entry(*key).or_insert(0) += 1;
    }

    let mut rows = Vec::with_capacity(counts.len());
    for (key, mut left) in counts {
        while left > 0 {
            let mu = left.min(MU_MAX);
            left -= mu;
            rows.push((key, mu));
        }
    }
    rows
}

/// Number of trace rows (real + padding) for the given deduplicated row count.
fn padded_rows(real_rows: usize) -> usize {
    real_rows.next_power_of_two().max(4)
}

/// Generates the IS_WH trace from the `IS_WH` sends of every caller.
///
/// Padding rows are all zero (`value = 0`, `μ = 0`, spec `pad = 0`).
pub fn generate_is_wh_trace(keys: &[IsWhKey]) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let rows = dedup_is_wh_rows(keys);
    let num_rows = padded_rows(rows.len());
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (key, mu)) in rows.iter().enumerate() {
        table.set_half(row_idx, cols::VALUE_0, key.half);
        table.set_half(row_idx, cols::VALUE_1, (key.word & 0xFFFF) as u16);
        table.set_half(row_idx, cols::VALUE_2, (key.word >> 16) as u16);
        table.set_u64(row_idx, cols::MU, *mu);
    }

    trace
}

/// Counts the IS_WH table's BITWISE lookups: per row, `IS_HALFWORD` ×μ on each
/// `value` limb and `IS_B20[μ]` ×1 — padding rows included, which send
/// `IS_B20[0]`.
pub(crate) fn collect_bitwise_from_is_wh(keys: &[IsWhKey], h: &mut BitwiseHistogram) {
    if keys.is_empty() {
        // No sends, no table (`generate_optional`), so nothing to count.
        return;
    }
    let rows = dedup_is_wh_rows(keys);
    let padding = padded_rows(rows.len()) - rows.len();

    let half = |v: u16| {
        BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (v & 0xFF) as u8,
            (v >> 8) as u8,
        )
    };
    let b20 = |v: u64| BitwiseOperation::b20(v as u8, (v >> 8) as u8, (v >> 16) as u8);

    for (key, mu) in &rows {
        h.bump_n(half(key.half), *mu);
        h.bump_n(half((key.word & 0xFFFF) as u16), *mu);
        h.bump_n(half((key.word >> 16) as u16), *mu);
        h.bump(b20(*mu));
    }
    h.bump_n(b20(0), padding as u64);
}

/// Bus interactions of the IS_WH table.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let column = |c: usize| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };

    vec![
        // IS_B20[μ] ×1: bounds μ (spec is_wh.toml, #1017).
        BusInteraction::sender(BusId::IsB20, Multiplicity::One, vec![column(cols::MU)]),
        // IS_HALF[value[i]] ×μ, i = 0..2 (is_wh:c:range_value).
        BusInteraction::sender(BusId::IsHalfword, mu(), vec![column(cols::VALUE_0)]),
        BusInteraction::sender(BusId::IsHalfword, mu(), vec![column(cols::VALUE_1)]),
        BusInteraction::sender(BusId::IsHalfword, mu(), vec![column(cols::VALUE_2)]),
        // IS_WH[value[1] + 2^16·value[2], value[0]] ×(−μ) (is_wh:c:is_wh).
        BusInteraction::receiver(
            BusId::IsWh,
            mu(),
            vec![
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::VALUE_1,
                    },
                    LinearTerm::Column {
                        coefficient: 1 << 16,
                        column: cols::VALUE_2,
                    },
                ]),
                column(cols::VALUE_0),
            ],
        ),
    ]
}

/// The `IS_WHH` template (spec `is_whh.toml`, #1015): range-checks the
/// `DWordWHH` `[half_lo, half_mid, word_hi]` at `start_column` by sending
/// `IS_WH[word_hi, half_mid]` and `IS_HALF[half_lo]`, both with multiplicity
/// `cond`.
///
/// `cond` must be constrained to a bit by the caller (spec `cond: Bit`). These
/// are bus sends, not constraints, so the template cannot enforce it: with
/// `cond = −1` the row would *receive* both checks, cancelling forged sends
/// elsewhere.
pub fn is_whh(start_column: usize, cond: Multiplicity) -> [BusInteraction; 2] {
    let column = |c: usize| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    [
        BusInteraction::sender(
            BusId::IsWh,
            cond.clone(),
            vec![column(start_column + 2), column(start_column + 1)],
        ),
        BusInteraction::sender(BusId::IsHalfword, cond, vec![column(start_column)]),
    ]
}
