//! Fused loads and stores: how a block checks the CPU's memory accesses
//! ([`MemoryFormat`]), and the memory check the two fused tables share.
//!
//! Under [`MemoryFormat::Split`] the CPU's memory dispatch (the `MemoryOp`
//! bus) is received by LOAD or STORE, which send a MEMW message; MEMW_A (or
//! MEMW) receives that message and consumes and produces the accessed bytes'
//! tokens on the Memory bus. Under [`MemoryFormat::Fused`] an access MEMW_A
//! would take (aligned to its width, its bytes of one old timestamp, the step
//! two 16-bit limbs on one high word) is received by MEM_LOAD
//! ([`super::mem_load`]) or MEM_STORE ([`super::mem_store`]) instead. The row
//! carries LOAD's (STORE's) rule and MEMW_A's byte tokens, step limbs and span
//! check, and the MEMW message between them is gone: each equality it carried
//! is the row's own column. LOAD, STORE and MEMW keep the CPU's other accesses;
//! MEMW_A keeps the ECALL chips'.
//!
//! The memory check ([`memory_check_interactions`]) is MEMW_A's in the limbs'
//! form ([`super::memw_aligned::bus_interactions_for`] at
//! [`MemwAlignedOrder::Limbs`](super::memw_aligned::MemwAlignedOrder::Limbs)),
//! with the register flag the constant 0 and one multiplicity column for the
//! row:
//! - `IS_HALF[base_address[0] + write2 + 3·write4 + 7·write8]` (the span);
//! - `IS_HALF[STEP_0]`, `IS_HALF[STEP_1]` (the step `ts − old_ts` is
//!   `1 + STEP_0 + 2^16·STEP_1`, in `1..=2^32`, on the row's own high word);
//! - per byte `k` a consumed token `(0, address + k, old_ts, old[k])` and a
//!   produced one `(0, address + k, ts, new[k])`, byte 0 at `μ`, byte 1 at
//!   `write2 + write4 + write8`, bytes 2 and 3 at `write4 + write8`, bytes 4 to
//!   7 at `write8`.

// No walk routes an access to the fused tables yet: until one does, only the
// tests reach the records, the fills and the lookup counts.
#![cfg_attr(not(test), allow(dead_code))]

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::memw_aligned::step_fits_limbs;
use super::types::BusId;

/// How a block checks the CPU's loads and stores: a constant of the proof
/// format, which the prover and the verifier of one proof take alike and
/// nothing reads from a proof (`block_whir::BlockFormat::memory`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Every CPU load (store) is a LOAD (STORE) row and a MEMW message,
    /// received by MEMW_A or MEMW: the format before fused loads, and the
    /// per-table STARK's.
    #[default]
    Split,
    /// A CPU load or store MEMW_A would take is one MEM_LOAD or MEM_STORE row,
    /// which receives the CPU's dispatch and holds the bytes' tokens itself;
    /// LOAD, STORE and MEMW take the rest (module docs).
    Fused,
}

/// The columns a fused table's memory check reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CheckColumns {
    /// The address as `DWordWHH`: low half, mid half, high word.
    pub(crate) base: [usize; 3],
    /// The timestamp as `DWordWL`.
    pub(crate) ts: [usize; 2],
    /// The width flags: 2, 4 and 8 bytes (none set: 1 byte).
    pub(crate) flags: [usize; 3],
    /// The bytes the consumed tokens carry.
    pub(crate) old: [usize; 8],
    /// The bytes the produced tokens carry.
    pub(crate) new: [usize; 8],
    /// The step less one, `ts − old_ts − 1`, as two 16-bit limbs.
    pub(crate) step: [usize; 2],
    /// The row's multiplicity.
    pub(crate) mu: usize,
}

fn column(c: usize) -> BusValue {
    BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    }
}

fn term(coefficient: i64, column: usize) -> LinearTerm {
    LinearTerm::Column {
        coefficient,
        column,
    }
}

/// The memory check of a fused row (module docs): the span's `IS_HALF`, the
/// two limbs' `IS_HALF`s, then the 16 byte tokens, each byte's consumed token
/// before its produced one. 19 interactions.
pub(crate) fn memory_check_interactions(c: &CheckColumns) -> Vec<BusInteraction> {
    let [f2, f4, f8] = c.flags;
    let mu = Multiplicity::Column(c.mu);
    let mut interactions = Vec::with_capacity(19);

    // The span: the access's last byte stays in the address's low half.
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        mu.clone(),
        vec![BusValue::linear(vec![
            term(1, c.base[0]),
            term(1, f2),
            term(3, f4),
            term(7, f8),
        ])],
    ));
    // The step's limbs: every active row sends both.
    for limb in c.step {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu.clone(),
            vec![BusValue::linear(vec![term(1, limb)])],
        ));
    }

    // The old timestamp: its low word `TS_0 − 1 − STEP_0 − 2^16·STEP_1`, its
    // high word the row's own `TS_1`.
    let old_ts_lo = BusValue::linear(vec![
        term(1, c.ts[0]),
        term(-1, c.step[0]),
        term(-(1 << 16), c.step[1]),
        LinearTerm::Constant(-1),
    ]);
    let address_hi = column(c.base[2]);
    for k in 0..8usize {
        let multiplicity = match k {
            0 => mu.clone(),
            1 => Multiplicity::Sum3(f2, f4, f8),
            2 | 3 => Multiplicity::Sum(f4, f8),
            _ => Multiplicity::Column(f8),
        };
        // An aligned access's byte addresses never carry out of the low word.
        let mut address_lo = vec![term(1, c.base[0]), term(1 << 16, c.base[1])];
        if k > 0 {
            address_lo.push(LinearTerm::Constant(k as i64));
        }
        let address_lo = BusValue::linear(address_lo);
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            multiplicity.clone(),
            vec![
                BusValue::constant(0),
                address_lo.clone(),
                address_hi.clone(),
                old_ts_lo.clone(),
                column(c.ts[1]),
                column(c.old[k]),
            ],
        ));
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            multiplicity,
            vec![
                BusValue::constant(0),
                address_lo,
                address_hi.clone(),
                column(c.ts[0]),
                column(c.ts[1]),
                column(c.new[k]),
            ],
        ));
    }
    interactions
}

/// Whether `width` is a width the CPU accesses memory at.
#[inline]
pub(crate) fn is_width(width: u8) -> bool {
    matches!(width, 1 | 2 | 4 | 8)
}

/// The width flags `(2, 4, 8)` of a `width`-byte access.
#[inline]
pub(crate) fn width_flags(width: u8) -> (bool, bool, bool) {
    (width == 2, width == 4, width == 8)
}

/// The step less one, `timestamp − old_timestamp − 1`, of an access a fused
/// row can take: aligned to its `width` at `address`, from an old timestamp
/// [`step_fits_limbs`] admits. Anything else is refused with the reason, never
/// laid out with wrapped limbs or a span past the low half.
pub(crate) fn fused_step(
    address: u64,
    width: u8,
    old_timestamp: u64,
    timestamp: u64,
) -> Result<u32, String> {
    if !is_width(width) {
        return Err(format!("a fused row takes 1, 2, 4 or 8 bytes, not {width}"));
    }
    if !address.is_multiple_of(u64::from(width)) {
        return Err(format!(
            "a fused row's access is aligned to its width: {width} bytes at {address:#x}"
        ));
    }
    if !step_fits_limbs(old_timestamp, timestamp) {
        return Err(format!(
            "a fused row needs a step in 1..2^32 on one high word, not {old_timestamp:#x} → \
             {timestamp:#x}"
        ));
    }
    Ok((timestamp - old_timestamp - 1) as u32)
}

/// The limbs `(STEP_0, STEP_1)` of a step less one.
#[inline]
pub(crate) fn step_limbs(step: u32) -> (u16, u16) {
    (step as u16, (step >> 16) as u16)
}

/// The three `IS_HALF` lookups of a fused row's memory check
/// ([`memory_check_interactions`]), counted into `histogram`: the span
/// `address[0] + width − 1` and the step's two limbs.
#[inline]
pub(crate) fn count_memory_check_into(
    address: u64,
    width: u8,
    step: u32,
    histogram: &mut BitwiseHistogram,
) {
    let span = (address & 0xFFFF) as u32 + u32::from(width) - 1;
    debug_assert!(span <= 0xFFFF, "an aligned access stays in its low half");
    let (d0, d1) = step_limbs(step);
    for half in [span as u16, d0, d1] {
        histogram.bump(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            half as u8,
            (half >> 8) as u8,
        ));
    }
}
