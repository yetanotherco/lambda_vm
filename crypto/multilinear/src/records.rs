//! A table's committed columns held as the records its rows come from, a small
//! fraction of the columns' bytes: the card fills the columns from them as it
//! uploads the table (`math_cuda::fill`), and a host reader that asks gets them
//! from the host fill the records also carry.
//!
//! Both fills write the same words, so the store a group's upload makes holds
//! the words the table's packed columns would have widened to, at the same
//! places ([`crate::gpu::upload_tables`]), and no proof byte depends on which
//! one ran.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::narrow::NarrowColumns;

/// Tables filled from their records on the card and on the host since the
/// process started ([`fills`]).
static CARD_FILLS: AtomicU64 = AtomicU64::new(0);
static HOST_FILLS: AtomicU64 = AtomicU64::new(0);

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_card_fills(tables: usize) {
    CARD_FILLS.fetch_add(tables as u64, Ordering::Relaxed);
}

pub(crate) fn note_host_fill() {
    HOST_FILLS.fetch_add(1, Ordering::Relaxed);
}

/// `(on the card, on the host)`: the tables filled from their records so far.
/// A block with a card fills each such table there, once a commit and once an
/// argument; a host fill is a reader that asked for the columns on the host.
pub fn fills() -> (u64, u64) {
    (
        CARD_FILLS.load(Ordering::Relaxed),
        HOST_FILLS.load(Ordering::Relaxed),
    )
}

/// One keccak permutation call as KECCAK_RND's fill reads it: the 24 rows of
/// its rounds are a function of these alone.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeccakRndOp {
    pub timestamp: u64,
    pub input: [u64; 25],
}

const _: () = {
    assert!(std::mem::size_of::<KeccakRndOp>() == 208);
    assert!(std::mem::offset_of!(KeccakRndOp, input) == 8);
};

/// What a card fill reads: the family's records and where its table starts in
/// what they make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordsFill<'a> {
    /// KECCAK_RND's rows `[skip, skip + rows)` of the table `ops` make, 24
    /// rows an op; rows past the last op are zero.
    KeccakRnd { ops: &'a [KeccakRndOp], skip: usize },
}

impl RecordsFill<'_> {
    /// The records' bytes, as they cross the bus.
    pub fn bytes(&self) -> usize {
        match self {
            Self::KeccakRnd { ops, .. } => std::mem::size_of_val(*ops),
        }
    }
}

/// A table's records: its shape, what the card's fill reads, and the host's
/// fill of the same columns.
pub trait TableRecords: Send + Sync + std::fmt::Debug {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// The records as the card's fill reads them.
    fn fill(&self) -> RecordsFill<'_>;
    /// The columns the host fill writes from the records, packed: the words
    /// the card's fill writes. Only a host reader asks. `None` where the host
    /// cannot (the table then reads as no columns, a proof the verifier
    /// refuses).
    fn host_fill(&self) -> Option<NarrowColumns>;
}

/// A table's records, shared: two are equal when they fill the same table
/// from the same records.
#[derive(Clone, Debug)]
pub struct Records(pub Arc<dyn TableRecords>);

impl Records {
    pub fn rows(&self) -> usize {
        self.0.rows()
    }

    pub fn cols(&self) -> usize {
        self.0.cols()
    }

    pub fn fill(&self) -> RecordsFill<'_> {
        self.0.fill()
    }

    /// The bytes the records take.
    pub fn bytes(&self) -> usize {
        self.0.fill().bytes()
    }

    pub fn host_fill(&self) -> Option<NarrowColumns> {
        self.0.host_fill()
    }
}

impl PartialEq for Records {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows() && self.cols() == other.cols() && self.fill() == other.fill()
    }
}

impl Eq for Records {}
