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

    /// The family, where its table starts, and the records' words, in order:
    /// what [`Records::digest`] reads.
    fn words(&self, mut each: impl FnMut(u64)) {
        match self {
            Self::KeccakRnd { ops, skip } => {
                each(0);
                each(*skip as u64);
                for op in ops.iter() {
                    each(op.timestamp);
                    op.input.iter().copied().for_each(&mut each);
                }
            }
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

    /// A 128-bit digest of the table's shape and records: a fast,
    /// non-cryptographic check that records made again are these (the
    /// packed columns' digest, `stark::regen::digest_of`, does the same for a
    /// packed table).
    pub fn digest(&self) -> [u64; 2] {
        const K: [u64; 4] = [
            0x9e37_79b9_7f4a_7c15,
            0xc2b2_ae3d_27d4_eb4f,
            0x1656_67b1_9e37_79f9,
            0x85eb_ca77_c2b2_ae63,
        ];
        let mut lanes = [
            self.rows() as u64,
            self.cols() as u64,
            self.bytes() as u64,
            0x243f_6a88_85a3_08d3,
        ];
        let mut i = 0usize;
        self.fill().words(|w| {
            let l = i % 4;
            lanes[l] = (lanes[l] ^ w).wrapping_mul(K[l]).rotate_left(31);
            i += 1;
        });
        // MurmurHash3's finalizer, so every input bit reaches every output bit.
        let fmix = |mut h: u64| {
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^ (h >> 33)
        };
        [
            fmix(lanes[0] ^ lanes[2].rotate_left(23)),
            fmix(lanes[1] ^ lanes[3].rotate_left(41)),
        ]
    }
}

impl PartialEq for Records {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows() && self.cols() == other.cols() && self.fill() == other.fill()
    }
}

impl Eq for Records {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Ops(Vec<KeccakRndOp>, usize, usize);

    impl TableRecords for Ops {
        fn rows(&self) -> usize {
            self.2
        }

        fn cols(&self) -> usize {
            1480
        }

        fn fill(&self) -> RecordsFill<'_> {
            RecordsFill::KeccakRnd {
                ops: &self.0,
                skip: self.1,
            }
        }

        fn host_fill(&self) -> Option<NarrowColumns> {
            None
        }
    }

    fn ops(n: u64) -> Vec<KeccakRndOp> {
        (0..n)
            .map(|k| KeccakRndOp {
                timestamp: 4 * k + 8,
                input: std::array::from_fn(|i| k * 100 + i as u64),
            })
            .collect()
    }

    fn records(ops: Vec<KeccakRndOp>, skip: usize, rows: usize) -> Records {
        Records(Arc::new(Ops(ops, skip, rows)))
    }

    /// Records are equal, and digest alike, when they fill the same table from
    /// the same ops; one word, the cut or the height apart, they are not.
    #[test]
    fn records_digest_their_table_and_every_word() {
        let base = records(ops(3), 8, 32);
        assert_eq!(base, records(ops(3), 8, 32));
        assert_eq!(base.digest(), records(ops(3), 8, 32).digest());
        assert_eq!(base.bytes(), 3 * 208);
        let mut lane = ops(3);
        lane[2].input[24] ^= 1;
        let mut stamp = ops(3);
        stamp[0].timestamp += 4;
        for (what, other) in [
            ("a lane", records(lane, 8, 32)),
            ("a timestamp", records(stamp, 8, 32)),
            ("the cut", records(ops(3), 9, 32)),
            ("the height", records(ops(3), 8, 64)),
            ("an op fewer", records(ops(2), 8, 32)),
        ] {
            assert_ne!(base, other, "{what}");
            assert_ne!(base.digest(), other.digest(), "{what}");
        }
    }
}
