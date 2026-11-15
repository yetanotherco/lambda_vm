//! MEMW_W: a chip's access to one aligned 8-byte RAM word, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! A row receives one chip message on [`BusId::MemWordMsg`]: a write (the new
//! word) at `MU_WRITE` or an access (the old word and the new one) at
//! `MU_ACCESS` ([`super::word_memory`]'s two shapes, each led by its kind
//! constant). It consumes the word's token at the message's address and an
//! earlier timestamp and produces the next one at the message's timestamp
//! ([`super::word_memory::word_check_interactions`]): it passes the token
//! through and mints none. The address is the message's two elements, taken
//! as they come: the row splits no address and checks no span, so whether an
//! access is aligned is decided by the keys PAGE_W mints alone.
//!
//! Register accesses never reach MEMW_W: they travel on [`BusId::Memw`] with
//! `is_register = 1`, and nothing on [`BusId::MemWordMsg`] is a register.
//!
//! ## Columns (11)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OLD_LO`, `OLD_HI`, `NEW_LO`,
//! `NEW_HI`, `STEP_0`, `STEP_1`, `MU_WRITE`, `MU_ACCESS`.
//!
//! ## Interactions (6)
//!
//! - `IS_HALF[STEP_0]`, `IS_HALF[STEP_1]`, the consumed token and the produced
//!   one, all at `MU_WRITE + MU_ACCESS`;
//! - the write message's receive at `MU_WRITE`, the access message's at
//!   `MU_ACCESS`.
//!
//! ## Constraints (3, degree 2)
//!
//! `MU_WRITE` a bit, `MU_ACCESS` a bit, their sum a bit.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, Multiplicity};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_memory::{
    WORD_BYTES, WordCheck, access_message, column, halves, step_less_one, step_limbs,
    word_check_interactions, write_message,
};
use crate::constraints::templates::emit_is_bit;

/// Column definitions for the MEMW_W table.
pub mod cols {
    /// The word's address: its low 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    /// The word's address: its high 32-bit word.
    pub const ADDRESS_HI: usize = 1;
    /// The access's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The word before the access, low and high halves.
    pub const OLD_LO: usize = 3;
    pub const OLD_HI: usize = 4;
    /// The word after the access, low and high halves.
    pub const NEW_LO: usize = 5;
    pub const NEW_HI: usize = 6;
    /// `timestamp − old_timestamp − 1`, low 16 bits.
    pub const STEP_0: usize = 7;
    /// `timestamp − old_timestamp − 1`, high 16 bits.
    pub const STEP_1: usize = 8;
    /// Whether the row takes a write message.
    pub const MU_WRITE: usize = 9;
    /// Whether the row takes an access message.
    pub const MU_ACCESS: usize = 10;
    pub const NUM_COLUMNS: usize = 11;
}

/// The columns MEMW_W's word check reads.
const CHECK: WordCheck = WordCheck {
    address: [cols::ADDRESS_LO, cols::ADDRESS_HI],
    timestamp: cols::TIMESTAMP,
    old: [cols::OLD_LO, cols::OLD_HI],
    new: [cols::NEW_LO, cols::NEW_HI],
    step: [cols::STEP_0, cols::STEP_1],
};

/// The multiplicity of a row's word check: either message.
fn mu() -> Multiplicity {
    Multiplicity::Sum(cols::MU_WRITE, cols::MU_ACCESS)
}

// =========================================================================
// The record
// =========================================================================

/// Which message a MEMW_W row takes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum MessageKind {
    /// An access: the chip names the old word and the new one.
    #[default]
    Access = 0,
    /// A write: the chip names the new word only.
    Write = 1,
}

/// One MEMW_W row: 40 bytes, `#[repr(C)]` with its layout fixed (asserted
/// below), as a card fill would read it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemwWordRow {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    kind: MessageKind,
    old: u64,
    new: u64,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<MemwWordRow>() == 40);
    assert!(offset_of!(MemwWordRow, address) == 0);
    assert!(offset_of!(MemwWordRow, timestamp) == 8);
    assert!(offset_of!(MemwWordRow, step) == 16);
    assert!(offset_of!(MemwWordRow, kind) == 20);
    assert!(offset_of!(MemwWordRow, old) == 24);
    assert!(offset_of!(MemwWordRow, new) == 32);
};

#[cfg_attr(not(test), allow(dead_code))]
impl MemwWordRow {
    /// The row of a chip's `kind` message on the word at `address`, at
    /// `timestamp`, over the word `old` last touched at `old_timestamp`, leaving
    /// `new`; or the reason no row takes it: an address that is not a multiple
    /// of 8, a read (an access with `old ≠ new` is a read-modify-write, which is
    /// allowed) that is not later than its token, or a step past `2^32`.
    pub(crate) fn new(
        kind: MessageKind,
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        old: u64,
        new: u64,
    ) -> Result<Self, String> {
        if !address.is_multiple_of(WORD_BYTES) {
            return Err(format!("a word access is 8-aligned, not at {address:#x}"));
        }
        let step = step_less_one(old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            kind,
            old,
            new,
        })
    }

    /// The row's BITWISE lookups: the two limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        let (d0, d1) = step_limbs(self.step);
        [d0, d1].into_iter().map(|half| {
            BitwiseOperation::halfword(BitwiseOperationType::IsHalf, half as u8, (half >> 8) as u8)
        })
    }

    /// [`Self::bitwise_ops`] counted into `histogram`.
    #[inline]
    pub(crate) fn count_bitwise_into(&self, histogram: &mut BitwiseHistogram) {
        for op in self.bitwise_ops() {
            histogram.bump(op);
        }
    }

    /// The timestamp of the token the row consumes.
    #[inline]
    pub(crate) fn old_timestamp(&self) -> u64 {
        self.timestamp - 1 - u64::from(self.step)
    }

    #[inline]
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    #[inline]
    pub(crate) fn old(&self) -> u64 {
        self.old
    }

    #[inline]
    pub(crate) fn new_word(&self) -> u64 {
        self.new
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEMW_W traces needed so far in this process (`tables::gpack`).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEMW_W trace of `segments`' rows, one after another, then zero rows to
/// a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_memw_w_trace_segments_as(
    segments: &[&[MemwWordRow]],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let len: usize = segments.iter().map(|s| s.len()).sum();
    let num_rows = len.next_power_of_two().max(4);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, r) in segments.iter().flat_map(|s| s.iter()).enumerate() {
            let (address_lo, address_hi) = halves(r.address);
            table.set_word(row, cols::ADDRESS_LO, address_lo);
            table.set_word(row, cols::ADDRESS_HI, address_hi);
            table.set_u64(row, cols::TIMESTAMP, r.timestamp);
            let (old_lo, old_hi) = halves(r.old);
            table.set_word(row, cols::OLD_LO, old_lo);
            table.set_word(row, cols::OLD_HI, old_hi);
            let (new_lo, new_hi) = halves(r.new);
            table.set_word(row, cols::NEW_LO, new_lo);
            table.set_word(row, cols::NEW_HI, new_hi);
            let (d0, d1) = step_limbs(r.step);
            table.set_half(row, cols::STEP_0, d0);
            table.set_half(row, cols::STEP_1, d1);
            let column = match r.kind {
                MessageKind::Write => cols::MU_WRITE,
                MessageKind::Access => cols::MU_ACCESS,
            };
            table.set_fe(row, column, FE::one());
        }
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

/// MEMW_W's 6 interactions (module docs): the word check, then the write
/// message's receive and the access message's.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = word_check_interactions(&CHECK, mu());
    let address = || [column(cols::ADDRESS_LO), column(cols::ADDRESS_HI)];
    let new = || [column(cols::NEW_LO), column(cols::NEW_HI)];
    interactions.push(BusInteraction::receiver(
        BusId::MemWordMsg,
        Multiplicity::Column(cols::MU_WRITE),
        write_message(address(), column(cols::TIMESTAMP), new()),
    ));
    interactions.push(BusInteraction::receiver(
        BusId::MemWordMsg,
        Multiplicity::Column(cols::MU_ACCESS),
        access_message(
            address(),
            column(cols::TIMESTAMP),
            [column(cols::OLD_LO), column(cols::OLD_HI)],
            new(),
        ),
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEMW_W's 3 constraints:
/// - idx 0, 1: `IS_BIT` on `MU_WRITE`, `MU_ACCESS`;
/// - idx 2: their sum a bit, so a row takes at most one message.
#[derive(Clone, Copy)]
pub struct MemwWordConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemwWordConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU_WRITE, None);
        emit_is_bit(b, 1, cols::MU_ACCESS, None);
        let sum = b.main(0, cols::MU_WRITE) + b.main(0, cols::MU_ACCESS);
        let one = b.one();
        b.emit_base(2, sum.clone() * (one - sum));
    }
}
