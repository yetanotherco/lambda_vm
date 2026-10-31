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

/// Under `LAMBDA_VM_CARD_UPLOAD_TIMES=1`, the group uploads' tables timed on
/// the card since the process started:
/// `((packed tables, copy and widen s), (records tables, copy and fill s))`
/// (`math_cuda::columns::upload_kernel_times`). Zeros otherwise, and in a
/// build without a card.
pub fn upload_times() -> ((u64, f64), (u64, f64)) {
    #[cfg(feature = "cuda")]
    {
        math_cuda::columns::upload_kernel_times()
    }
    #[cfg(not(feature = "cuda"))]
    {
        ((0, 0.0), (0, 0.0))
    }
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

/// One cycle's executor log as CPU's fill reads it (the executor's `Log`,
/// field for field): the pcs before and after, and the values it read and
/// wrote.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuLog {
    pub current_pc: u64,
    pub next_pc: u64,
    pub src1_val: u64,
    pub src2_val: u64,
    pub dst_val: u64,
}

/// One MEMW_R row as its fill reads it (the prover's register row, field for
/// field): `address` the register index, `old_ts_lo` the old timestamp's low
/// word, `is_read` 0 or 1, `pad` 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegisterRow {
    pub timestamp: u64,
    pub val0: u32,
    pub val1: u32,
    pub old0: u32,
    pub old1: u32,
    pub old_ts_lo: u32,
    pub address: u16,
    pub is_read: u8,
    pub pad: u8,
}

/// One MEMW_A row as its fill reads it (the prover's aligned row, field for
/// field): the base address, timestamp and shared old timestamp, the value
/// and old value packed a word each (eight bytes for a memory access, the two
/// 32-bit halves for a register access), the width, and flags 0 or 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AlignedRow {
    pub base_address: u64,
    pub timestamp: u64,
    pub old_timestamp: u64,
    pub value: u64,
    pub old: u64,
    pub width: u8,
    pub is_read: u8,
    pub is_register: u8,
}

/// One LOAD op as its fill reads it (the prover's load op, field for field):
/// the base address, the timestamp, the width, whether it sign-extends (0 or
/// 1) and the result's bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOp {
    pub base_address: u64,
    pub timestamp: u64,
    pub width: u8,
    pub signed: u8,
    pub res: [u8; 8],
}

/// One STORE op as its fill reads it (the prover's store op, field for field):
/// the base address, the timestamp, the value, and the width flags (0 or 1).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreOp {
    pub base_address: u64,
    pub timestamp: u64,
    pub value: u64,
    pub write2: u8,
    pub write4: u8,
    pub write8: u8,
}

/// One SHIFT op as its fill reads it (the prover's shift op, field for
/// field): the full shift amount, the input as four 16-bit halves, the
/// amount's low byte, and flags 0 or 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShiftOp {
    pub shift_amount: u64,
    pub in_halves: [u16; 4],
    pub shift: u8,
    pub direction: u8,
    pub signed: u8,
    pub word_instr: u8,
}

/// One MEMW op as its fill reads it (the prover's MEMW op, field for field):
/// the base address, the timestamp, the eight old timestamps, the eight value
/// and old elements, and the flags and width.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemwOp {
    pub base_address: u64,
    pub timestamp: u64,
    pub old_timestamp: [u64; 8],
    pub value: [u32; 8],
    pub old: [u32; 8],
    pub is_register: u8,
    pub width: u8,
    pub is_read: u8,
}

/// One DECODE row as CPU's fill reads it: the pc, the extended immediate,
/// then the decoded fields as bytes (flags 0 or 1).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeRow {
    pub pc: u64,
    pub imm: u64,
    pub read_register1: u8,
    pub read_register2: u8,
    pub write_register: u8,
    pub word_instr: u8,
    pub alu: u8,
    pub add: u8,
    pub sub: u8,
    pub memory: u8,
    pub branch: u8,
    pub ecall: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub rd: u8,
    pub half_instruction_length: u8,
    pub alu_flags: u8,
    pub mem_flags: u8,
}

/// A run of pcs 4 apart: pc `first_pc + 4k` is DECODE row `row + k` for
/// `k < len`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeRun {
    pub first_pc: u64,
    pub row: u32,
    pub len: u32,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<CpuLog>() == 40);
    assert!(offset_of!(CpuLog, dst_val) == 32);
    assert!(size_of::<RegisterRow>() == 32);
    assert!(offset_of!(RegisterRow, old_ts_lo) == 24);
    assert!(offset_of!(RegisterRow, address) == 28);
    assert!(offset_of!(RegisterRow, is_read) == 30);
    assert!(offset_of!(RegisterRow, pad) == 31);
    assert!(size_of::<DecodeRow>() == 32);
    assert!(offset_of!(DecodeRow, read_register1) == 16);
    assert!(offset_of!(DecodeRow, mem_flags) == 31);
    assert!(size_of::<DecodeRun>() == 16);
    assert!(offset_of!(DecodeRun, len) == 12);
    assert!(size_of::<AlignedRow>() == 48);
    assert!(offset_of!(AlignedRow, width) == 40);
    assert!(offset_of!(AlignedRow, is_register) == 42);
    assert!(size_of::<LoadOp>() == 32);
    assert!(offset_of!(LoadOp, signed) == 17);
    assert!(offset_of!(LoadOp, res) == 18);
    assert!(size_of::<StoreOp>() == 32);
    assert!(offset_of!(StoreOp, write8) == 26);
    assert!(size_of::<ShiftOp>() == 24);
    assert!(offset_of!(ShiftOp, word_instr) == 19);
    assert!(size_of::<MemwOp>() == 152);
    assert!(offset_of!(MemwOp, is_read) == 146);
};

/// A program's DECODE rows as CPU's fill reads them, shared by every CPU
/// table of a prove: the entries in pc order and the runs of pcs 4 apart. On
/// a card it is uploaded once, at the first fill that reads it, and freed with
/// it.
pub struct DecodeRecords {
    entries: Vec<DecodeRow>,
    runs: Vec<DecodeRun>,
    #[cfg(feature = "cuda")]
    card: std::sync::Mutex<Option<Arc<math_cuda::fill::CardDecode>>>,
}

impl DecodeRecords {
    /// The rows and runs of a program. Refused (`None`) unless there is a row,
    /// the runs ascend without overlapping and every row they name is one.
    pub fn new(entries: Vec<DecodeRow>, runs: Vec<DecodeRun>) -> Option<Self> {
        if entries.is_empty() || runs.is_empty() {
            return None;
        }
        let mut end_pc = 0u64;
        for (k, run) in runs.iter().enumerate() {
            let last = run
                .first_pc
                .checked_add(4 * u64::from(run.len.checked_sub(1)?))?;
            if (k > 0 && run.first_pc <= end_pc)
                || run.row as usize + run.len as usize > entries.len()
            {
                return None;
            }
            end_pc = last;
        }
        Some(Self {
            entries,
            runs,
            #[cfg(feature = "cuda")]
            card: std::sync::Mutex::new(None),
        })
    }

    pub fn entries(&self) -> &[DecodeRow] {
        &self.entries
    }

    pub fn runs(&self) -> &[DecodeRun] {
        &self.runs
    }

    /// The DECODE row of `pc`, as CPU's fill finds it: the last run starting
    /// at or below `pc`, if it holds it.
    pub fn row(&self, pc: u64) -> Option<u32> {
        let at = self.runs.partition_point(|run| run.first_pc <= pc);
        let run = self.runs.get(at.checked_sub(1)?)?;
        let offset = pc - run.first_pc;
        (offset.is_multiple_of(4) && offset / 4 < u64::from(run.len))
            .then(|| run.row + (offset / 4) as u32)
    }

    /// The rows on the card: uploaded at the first call, under a promise of
    /// their bytes, and kept until this goes. `None` when the card declines.
    #[cfg(feature = "cuda")]
    pub(crate) fn on_card(&self) -> Option<Arc<math_cuda::fill::CardDecode>> {
        use math_cuda::fill::{CardDecode, DecodeEntryC, DecodeRunC};
        const _: () = {
            use std::mem::{align_of, size_of};
            assert!(size_of::<DecodeRow>() == size_of::<DecodeEntryC>());
            assert!(align_of::<DecodeRow>() == align_of::<DecodeEntryC>());
            assert!(
                std::mem::offset_of!(DecodeRow, mem_flags)
                    == std::mem::offset_of!(DecodeEntryC, mem_flags)
            );
            assert!(size_of::<DecodeRun>() == size_of::<DecodeRunC>());
            assert!(align_of::<DecodeRun>() == align_of::<DecodeRunC>());
            assert!(std::mem::offset_of!(DecodeRun, len) == std::mem::offset_of!(DecodeRunC, len));
        };
        let mut held = self.card.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(card) = held.as_ref() {
            return Some(Arc::clone(card));
        }
        // SAFETY: each pair is `#[repr(C)]` with the same fields in the same
        // order, so the same size, alignment and offsets (asserted above).
        let (entries, runs) = unsafe {
            (
                std::slice::from_raw_parts(
                    self.entries.as_ptr() as *const DecodeEntryC,
                    self.entries.len(),
                ),
                std::slice::from_raw_parts(
                    self.runs.as_ptr() as *const DecodeRunC,
                    self.runs.len(),
                ),
            )
        };
        let card = Arc::new(CardDecode::resident(entries, runs).ok()?);
        *held = Some(Arc::clone(&card));
        Some(card)
    }
}

impl std::fmt::Debug for DecodeRecords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DecodeRecords {{ {} rows, {} runs }}",
            self.entries.len(),
            self.runs.len()
        )
    }
}

impl PartialEq for DecodeRecords {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries && self.runs == other.runs
    }
}

impl Eq for DecodeRecords {}

/// What a card fill reads: the family's records and where its table starts in
/// what they make. A table's records may lie in pieces (the window parts a
/// chunk was cut from), read one after another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordsFill<'a> {
    /// KECCAK_RND's rows `[skip, skip + rows)` of the table `ops` make, 24
    /// rows an op; rows past the last op are zero.
    KeccakRnd { ops: &'a [KeccakRndOp], skip: usize },
    /// CPU's rows: one a log of the pieces in order, log `i` the cycle
    /// `first_cycle + i`, read with its row of `decode`; then the padding rows.
    Cpu {
        logs: Vec<&'a [CpuLog]>,
        first_cycle: u64,
        decode: &'a DecodeRecords,
    },
    /// MEMW_R's rows: the pieces in order, then zero rows.
    MemwR { rows: Vec<&'a [RegisterRow]> },
    /// MEMW_A's rows: the pieces in order, then zero rows.
    MemwA { rows: Vec<&'a [AlignedRow]> },
    /// LOAD's rows: one an op of the pieces in order, then zero rows.
    Load { ops: Vec<&'a [LoadOp]> },
    /// STORE's rows: one an op of the pieces in order, then zero rows.
    Store { ops: Vec<&'a [StoreOp]> },
    /// SHIFT's rows: one an op of the pieces in order, then padding rows.
    Shift { ops: Vec<&'a [ShiftOp]> },
    /// MEMW's rows: one an op of the pieces in order, then zero rows.
    Memw { ops: Vec<&'a [MemwOp]> },
}

impl RecordsFill<'_> {
    /// The records' bytes, as they cross the bus (a CPU table's DECODE rows
    /// aside: one copy serves every CPU table).
    pub fn bytes(&self) -> usize {
        match self {
            Self::KeccakRnd { ops, .. } => std::mem::size_of_val(*ops),
            Self::Cpu { logs, .. } => logs.iter().map(|p| std::mem::size_of_val(*p)).sum(),
            Self::MemwR { rows } => rows.iter().map(|p| std::mem::size_of_val(*p)).sum(),
            Self::MemwA { rows } => rows.iter().map(|p| std::mem::size_of_val(*p)).sum(),
            Self::Load { ops } => ops.iter().map(|p| std::mem::size_of_val(*p)).sum(),
            Self::Store { ops } => ops.iter().map(|p| std::mem::size_of_val(*p)).sum(),
            Self::Shift { ops } => ops.iter().map(|p| std::mem::size_of_val(*p)).sum(),
            Self::Memw { ops } => ops.iter().map(|p| std::mem::size_of_val(*p)).sum(),
        }
    }

    /// The records' count: ops, logs or rows.
    pub fn len(&self) -> usize {
        match self {
            Self::KeccakRnd { ops, .. } => ops.len(),
            Self::Cpu { logs, .. } => logs.iter().map(|p| p.len()).sum(),
            Self::MemwR { rows } => rows.iter().map(|p| p.len()).sum(),
            Self::MemwA { rows } => rows.iter().map(|p| p.len()).sum(),
            Self::Load { ops } => ops.iter().map(|p| p.len()).sum(),
            Self::Store { ops } => ops.iter().map(|p| p.len()).sum(),
            Self::Shift { ops } => ops.iter().map(|p| p.len()).sum(),
            Self::Memw { ops } => ops.iter().map(|p| p.len()).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The family, where its table starts, and the records' words, in order
    /// (whatever pieces they lie in): what [`Records::digest`] reads. A CPU
    /// table's DECODE rows are the program's, the same for every table: left
    /// out.
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
            Self::Cpu {
                logs, first_cycle, ..
            } => {
                each(1);
                each(*first_cycle);
                for log in logs.iter().flat_map(|piece| piece.iter()) {
                    each(log.current_pc);
                    each(log.next_pc);
                    each(log.src1_val);
                    each(log.src2_val);
                    each(log.dst_val);
                }
            }
            Self::MemwR { rows } => {
                each(2);
                for row in rows.iter().flat_map(|piece| piece.iter()) {
                    each(row.timestamp);
                    each(u64::from(row.val0) | u64::from(row.val1) << 32);
                    each(u64::from(row.old0) | u64::from(row.old1) << 32);
                    each(
                        u64::from(row.old_ts_lo)
                            | u64::from(row.address) << 32
                            | u64::from(row.is_read) << 48
                            | u64::from(row.pad) << 56,
                    );
                }
            }
            // The fields alone: these records end in padding.
            Self::MemwA { rows } => {
                each(3);
                for row in rows.iter().flat_map(|piece| piece.iter()) {
                    each(row.base_address);
                    each(row.timestamp);
                    each(row.old_timestamp);
                    each(row.value);
                    each(row.old);
                    each(
                        u64::from(row.width)
                            | u64::from(row.is_read) << 8
                            | u64::from(row.is_register) << 16,
                    );
                }
            }
            Self::Load { ops } => {
                each(4);
                for op in ops.iter().flat_map(|piece| piece.iter()) {
                    each(op.base_address);
                    each(op.timestamp);
                    each(u64::from(op.width) | u64::from(op.signed) << 8);
                    each(u64::from_le_bytes(op.res));
                }
            }
            Self::Store { ops } => {
                each(5);
                for op in ops.iter().flat_map(|piece| piece.iter()) {
                    each(op.base_address);
                    each(op.timestamp);
                    each(op.value);
                    each(
                        u64::from(op.write2)
                            | u64::from(op.write4) << 8
                            | u64::from(op.write8) << 16,
                    );
                }
            }
            Self::Shift { ops } => {
                each(6);
                for op in ops.iter().flat_map(|piece| piece.iter()) {
                    each(op.shift_amount);
                    each(
                        op.in_halves
                            .iter()
                            .enumerate()
                            .fold(0, |w, (i, &h)| w | u64::from(h) << (16 * i)),
                    );
                    each(
                        u64::from(op.shift)
                            | u64::from(op.direction) << 8
                            | u64::from(op.signed) << 16
                            | u64::from(op.word_instr) << 24,
                    );
                }
            }
            Self::Memw { ops } => {
                each(7);
                for op in ops.iter().flat_map(|piece| piece.iter()) {
                    each(op.base_address);
                    each(op.timestamp);
                    op.old_timestamp.iter().for_each(|&t| each(t));
                    for pair in [&op.value, &op.old] {
                        for two in pair.chunks_exact(2) {
                            each(u64::from(two[0]) | u64::from(two[1]) << 32);
                        }
                    }
                    each(
                        u64::from(op.is_register)
                            | u64::from(op.width) << 8
                            | u64::from(op.is_read) << 16,
                    );
                }
            }
        }
    }
}

/// What a records holder handed back to the OS as it let its records go
/// ([`TableRecords::return_pages`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagesReturned {
    /// Buffers this holder was the last to hold: their pages handed back (or
    /// tried, see `failed`).
    pub buffers: usize,
    /// Buffers another holder still holds: left as they are.
    pub shared: usize,
    /// Bytes of whole pages handed back.
    pub bytes: u64,
    /// Buffers whose pages the OS would not take back (logged once).
    pub failed: usize,
}

impl std::ops::AddAssign for PagesReturned {
    fn add_assign(&mut self, other: Self) {
        self.buffers += other.buffers;
        self.shared += other.shared;
        self.bytes += other.bytes;
        self.failed += other.failed;
    }
}

/// A table's records: its shape, what the card's fill reads, and the host's
/// fill of the same columns.
pub trait TableRecords: Send + Sync + std::fmt::Debug {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// The records as the card's fill reads them.
    fn fill(&self) -> RecordsFill<'_>;
    /// The records' bytes ([`RecordsFill::bytes`]).
    fn bytes(&self) -> usize {
        self.fill().bytes()
    }

    /// The leading columns the fill writes from the row index alone, as a
    /// table's preprocessed columns are: a table held as records is laid out
    /// only for an AIR with that many preprocessed columns.
    fn row_index_columns(&self) -> usize {
        0
    }
    /// The columns the host fill writes from the records, packed: the words
    /// the card's fill writes. Only a host reader asks. `None` where the host
    /// cannot (the table then reads as no columns, a proof the verifier
    /// refuses).
    fn host_fill(&self) -> Option<NarrowColumns>;

    /// Hands the pages of the buffers this holder is the last to hold back to
    /// the OS, as its records are let go: nothing reads them after, their
    /// holder drops them next ([`Records::return_pages`]). Buffers another
    /// holder still holds stay as they are. The default returns nothing.
    fn return_pages(&mut self) -> PagesReturned {
        PagesReturned::default()
    }
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
        self.0.bytes()
    }

    /// [`TableRecords::row_index_columns`].
    pub fn row_index_columns(&self) -> usize {
        self.0.row_index_columns()
    }

    /// A 128-bit digest of the table's shape and records: a fast,
    /// non-cryptographic check that records made again are these (the
    /// packed columns' digest, `stark::regen::digest_of`, does the same for a
    /// packed table).
    pub fn digest(&self) -> [u64; 2] {
        digest_of(self.0.as_ref())
    }

    pub fn host_fill(&self) -> Option<NarrowColumns> {
        self.0.host_fill()
    }

    /// Lets the records go, first handing their buffers' pages back to the
    /// OS where this is the records' last handle
    /// ([`TableRecords::return_pages`]); another handle still holding them
    /// returns nothing.
    pub fn return_pages(mut self) -> PagesReturned {
        match Arc::get_mut(&mut self.0) {
            Some(records) => records.return_pages(),
            None => PagesReturned::default(),
        }
    }
}

/// [`Records::digest`] of records not (yet) shared: a regenerator that
/// rebuilt a table's records digests them before it fills the table from
/// them.
pub fn digest_of(records: &dyn TableRecords) -> [u64; 2] {
    const K: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0x85eb_ca77_c2b2_ae63,
    ];
    let mut lanes = [
        records.rows() as u64,
        records.cols() as u64,
        records.bytes() as u64,
        0x243f_6a88_85a3_08d3,
    ];
    let mut i = 0usize;
    records.fill().words(|w| {
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

    fn decode() -> DecodeRecords {
        let row = |pc| DecodeRow {
            pc,
            ..DecodeRow::default()
        };
        DecodeRecords::new(
            vec![row(0x1000), row(0x1004), row(0x2000), row(0x2008)],
            vec![
                DecodeRun {
                    first_pc: 0x1000,
                    row: 0,
                    len: 2,
                },
                DecodeRun {
                    first_pc: 0x2000,
                    row: 2,
                    len: 1,
                },
                DecodeRun {
                    first_pc: 0x2008,
                    row: 3,
                    len: 1,
                },
            ],
        )
        .expect("a decode table")
    }

    fn logs(n: u64) -> Vec<CpuLog> {
        (0..n)
            .map(|i| CpuLog {
                current_pc: 0x1000 + 4 * (i % 2),
                next_pc: 0x1004,
                src1_val: i,
                src2_val: i.wrapping_mul(0x9e37_79b9),
                dst_val: !i,
            })
            .collect()
    }

    /// Every pc a run holds finds its row; the pcs between and around the
    /// runs find none.
    #[test]
    fn a_pc_finds_the_row_of_its_run() {
        let decode = decode();
        for (pc, row) in [(0x1000, 0), (0x1004, 1), (0x2000, 2), (0x2008, 3)] {
            assert_eq!(decode.row(pc), Some(row), "pc {pc:#x}");
        }
        for pc in [0, 0xffc, 0x1002, 0x1008, 0x2004, 0x200c, u64::MAX] {
            assert_eq!(decode.row(pc), None, "pc {pc:#x}");
        }
    }

    /// DECODE rows are held only as a table the fill can search: a row, runs
    /// that ascend without overlapping, each holding a pc, every row named one
    /// of the table's.
    #[test]
    fn decode_rows_that_are_no_table_are_refused() {
        let ok = decode();
        let entries = ok.entries().to_vec();
        let runs = ok.runs().to_vec();
        assert!(DecodeRecords::new(entries.clone(), runs.clone()).is_some());
        assert!(DecodeRecords::new(Vec::new(), runs.clone()).is_none());
        assert!(DecodeRecords::new(entries.clone(), Vec::new()).is_none());
        let mut empty_run = runs.clone();
        empty_run[1].len = 0;
        assert!(DecodeRecords::new(entries.clone(), empty_run).is_none());
        let mut overlapping = runs.clone();
        overlapping[1].first_pc = 0x1004;
        assert!(DecodeRecords::new(entries.clone(), overlapping).is_none());
        let mut descending = runs.clone();
        descending.swap(0, 1);
        assert!(DecodeRecords::new(entries.clone(), descending).is_none());
        let mut past = runs;
        past[2].row = 4;
        assert!(DecodeRecords::new(entries, past).is_none());
    }

    /// CPU, MEMW_R or a row family's records in pieces.
    #[derive(Debug)]
    enum Pieces {
        Cpu(Vec<Vec<CpuLog>>, u64, Arc<DecodeRecords>),
        MemwR(Vec<Vec<RegisterRow>>),
        MemwA(Vec<Vec<AlignedRow>>),
        Load(Vec<Vec<LoadOp>>),
        Store(Vec<Vec<StoreOp>>),
        Shift(Vec<Vec<ShiftOp>>),
        Memw(Vec<Vec<MemwOp>>),
    }

    impl TableRecords for Pieces {
        fn rows(&self) -> usize {
            16
        }

        fn cols(&self) -> usize {
            match self {
                Self::Cpu(..) => 38,
                Self::MemwR(_) => 10,
                Self::MemwA(_) => 29,
                Self::Load(_) => 18,
                Self::Store(_) => 16,
                Self::Shift(_) => 29,
                Self::Memw(_) => 49,
            }
        }

        fn fill(&self) -> RecordsFill<'_> {
            match self {
                Self::Cpu(pieces, first_cycle, decode) => RecordsFill::Cpu {
                    logs: pieces.iter().map(Vec::as_slice).collect(),
                    first_cycle: *first_cycle,
                    decode,
                },
                Self::MemwR(pieces) => RecordsFill::MemwR {
                    rows: pieces.iter().map(Vec::as_slice).collect(),
                },
                Self::MemwA(pieces) => RecordsFill::MemwA {
                    rows: pieces.iter().map(Vec::as_slice).collect(),
                },
                Self::Load(pieces) => RecordsFill::Load {
                    ops: pieces.iter().map(Vec::as_slice).collect(),
                },
                Self::Store(pieces) => RecordsFill::Store {
                    ops: pieces.iter().map(Vec::as_slice).collect(),
                },
                Self::Shift(pieces) => RecordsFill::Shift {
                    ops: pieces.iter().map(Vec::as_slice).collect(),
                },
                Self::Memw(pieces) => RecordsFill::Memw {
                    ops: pieces.iter().map(Vec::as_slice).collect(),
                },
            }
        }

        fn host_fill(&self) -> Option<NarrowColumns> {
            None
        }
    }

    /// `records` cut into pieces at `cuts`.
    fn cut<T: Clone>(records: &[T], cuts: &[usize]) -> Vec<Vec<T>> {
        let mut pieces = Vec::new();
        let mut from = 0;
        for &to in cuts.iter().chain([&records.len()]) {
            pieces.push(records[from..to].to_vec());
            from = to;
        }
        pieces
    }

    /// ★ A CPU or MEMW_R table's digest reads its records' words in order,
    /// whatever pieces they lie in, and where the table starts: one bit moved
    /// in any field of any record, another first cycle, or a record fewer,
    /// moves it.
    #[test]
    fn the_records_digest_reads_every_field_and_not_the_pieces() {
        let decode = Arc::new(decode());
        let logs = logs(9);
        let cpu = |pieces: Vec<Vec<CpuLog>>, first_cycle: u64| {
            Records(Arc::new(Pieces::Cpu(
                pieces,
                first_cycle,
                Arc::clone(&decode),
            )))
            .digest()
        };
        let whole = cpu(cut(&logs, &[]), 7);
        assert_eq!(cpu(cut(&logs, &[4]), 7), whole);
        assert_eq!(cpu(cut(&logs, &[1, 1, 8]), 7), whole);
        assert_ne!(cpu(cut(&logs, &[]), 8), whole, "the first cycle");
        assert_ne!(cpu(cut(&logs[..8], &[]), 7), whole, "a record fewer");
        for at in 0..logs.len() {
            for field in 0..5 {
                let mut moved = logs.clone();
                let log = &mut moved[at];
                let word = match field {
                    0 => &mut log.current_pc,
                    1 => &mut log.next_pc,
                    2 => &mut log.src1_val,
                    3 => &mut log.src2_val,
                    _ => &mut log.dst_val,
                };
                *word ^= 1 << (at % 64);
                assert_ne!(cpu(cut(&moved, &[3]), 7), whole, "log {at}, field {field}");
            }
        }
        let rows: Vec<RegisterRow> = (0..5u32)
            .map(|i| RegisterRow {
                timestamp: 4 * u64::from(i) + 8,
                val0: i,
                val1: i + 1,
                old0: i + 2,
                old1: i + 3,
                old_ts_lo: i + 4,
                address: i as u16,
                is_read: (i % 2) as u8,
                pad: 0,
            })
            .collect();
        let memw_r = |pieces| Records(Arc::new(Pieces::MemwR(pieces))).digest();
        let whole = memw_r(cut(&rows, &[]));
        assert_eq!(memw_r(cut(&rows, &[2])), whole);
        for at in 0..rows.len() {
            for field in 0..9 {
                let mut moved = rows.clone();
                let row = &mut moved[at];
                match field {
                    0 => row.timestamp ^= 1 << 40,
                    1 => row.val0 ^= 1,
                    2 => row.val1 ^= 1 << 31,
                    3 => row.old0 ^= 1,
                    4 => row.old1 ^= 1 << 31,
                    5 => row.old_ts_lo ^= 1,
                    6 => row.address ^= 1 << 15,
                    7 => row.is_read ^= 1,
                    _ => row.pad ^= 1,
                }
                assert_ne!(memw_r(cut(&moved, &[])), whole, "row {at}, field {field}");
            }
        }
    }

    /// ★ A MEMW_A or LOAD table's digest reads every field of every row, in
    /// order, whatever pieces they lie in: one bit moved in any field (each
    /// result byte apart), or a row fewer, moves it.
    #[test]
    fn the_row_records_digest_reads_every_field_and_not_the_pieces() {
        let aligned: Vec<AlignedRow> = (0..5u64)
            .map(|i| AlignedRow {
                base_address: 0x8000 + 8 * i,
                timestamp: 4 * i + 8,
                old_timestamp: 4 * i + 1,
                value: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                old: !i,
                width: 8,
                is_read: (i % 2) as u8,
                is_register: (i % 3 == 0) as u8,
            })
            .collect();
        let memw_a = |pieces| Records(Arc::new(Pieces::MemwA(pieces))).digest();
        let whole = memw_a(cut(&aligned, &[]));
        assert_eq!(memw_a(cut(&aligned, &[2])), whole);
        assert_eq!(memw_a(cut(&aligned, &[0, 1, 5])), whole);
        assert_ne!(memw_a(cut(&aligned[..4], &[])), whole, "a row fewer");
        for at in 0..aligned.len() {
            for field in 0..8 {
                let mut moved = aligned.clone();
                let row = &mut moved[at];
                match field {
                    0 => row.base_address ^= 1 << 63,
                    1 => row.timestamp ^= 1,
                    2 => row.old_timestamp ^= 1 << 32,
                    3 => row.value ^= 1 << (at % 64),
                    4 => row.old ^= 1 << 63,
                    5 => row.width ^= 0x80,
                    6 => row.is_read ^= 1,
                    _ => row.is_register ^= 1,
                }
                assert_ne!(memw_a(cut(&moved, &[3])), whole, "row {at}, field {field}");
            }
        }
        let ops: Vec<LoadOp> = (0..5u64)
            .map(|i| LoadOp {
                base_address: 0x9000 + i,
                timestamp: 4 * i + 8,
                width: 1 << (i % 4),
                signed: (i % 2) as u8,
                res: (i * 0x0102_0304_0506_0708).to_le_bytes(),
            })
            .collect();
        let load = |pieces| Records(Arc::new(Pieces::Load(pieces))).digest();
        let whole = load(cut(&ops, &[]));
        assert_eq!(load(cut(&ops, &[4])), whole);
        assert_ne!(load(cut(&ops[1..], &[])), whole, "an op fewer");
        for at in 0..ops.len() {
            for field in 0..12 {
                let mut moved = ops.clone();
                let op = &mut moved[at];
                match field {
                    0 => op.base_address ^= 1 << 40,
                    1 => op.timestamp ^= 1 << 63,
                    2 => op.width ^= 0x80,
                    3 => op.signed ^= 1,
                    byte => op.res[byte - 4] ^= 1 << (byte % 8),
                }
                assert_ne!(load(cut(&moved, &[1])), whole, "op {at}, field {field}");
            }
        }
    }

    /// ★ A STORE, SHIFT or MEMW table's digest reads every field of every op, in
    /// order, whatever pieces they lie in: one bit moved in any field (each
    /// flag, half, element and old timestamp apart), or an op fewer, moves it.
    #[test]
    fn the_op_records_digest_reads_every_field_and_not_the_pieces() {
        let mut rng = 0x5EED_u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng ^ rng >> 29
        };
        // The digest of `ops` is the same however they are cut, and moves with
        // an op fewer or any one field of any op flipped (`flip` flips field
        // `k`, and says `false` past the last).
        fn each_field<T: Clone>(
            ops: &[T],
            digest: impl Fn(Vec<Vec<T>>) -> [u64; 2],
            flip: impl Fn(&mut T, usize) -> bool,
            what: &str,
        ) {
            let whole = digest(cut(ops, &[]));
            assert_eq!(digest(cut(ops, &[1])), whole, "{what}: the pieces");
            assert_eq!(
                digest(cut(ops, &[0, 2, ops.len()])),
                whole,
                "{what}: the pieces"
            );
            assert_ne!(digest(cut(&ops[1..], &[])), whole, "{what}: an op fewer");
            for at in 0..ops.len() {
                let mut k = 0;
                loop {
                    let mut moved = ops.to_vec();
                    if !flip(&mut moved[at], k) {
                        break;
                    }
                    assert_ne!(
                        digest(cut(&moved, &[2])),
                        whole,
                        "{what}: op {at}, field {k}"
                    );
                    k += 1;
                }
                assert!(k > 3, "{what}: {k} fields");
            }
        }
        let stores: Vec<StoreOp> = (0..4u8)
            .map(|i| StoreOp {
                base_address: next(),
                timestamp: next(),
                value: next(),
                write2: u8::from(i == 1),
                write4: u8::from(i == 2),
                write8: u8::from(i == 3),
            })
            .collect();
        each_field(
            &stores,
            |p| Records(Arc::new(Pieces::Store(p))).digest(),
            |op, k| {
                match k {
                    0 => op.base_address ^= 1 << 63,
                    1 => op.timestamp ^= 1,
                    2 => op.value ^= 1 << 40,
                    3 => op.write2 ^= 1,
                    4 => op.write4 ^= 1,
                    5 => op.write8 ^= 1,
                    _ => return false,
                }
                true
            },
            "STORE",
        );
        let shifts: Vec<ShiftOp> = (0..4u8)
            .map(|i| ShiftOp {
                shift_amount: next(),
                in_halves: [next() as u16, next() as u16, next() as u16, next() as u16],
                shift: next() as u8,
                direction: i & 1,
                signed: (i >> 1) & 1,
                word_instr: u8::from(i == 3),
            })
            .collect();
        each_field(
            &shifts,
            |p| Records(Arc::new(Pieces::Shift(p))).digest(),
            |op, k| {
                match k {
                    0 => op.shift_amount ^= 1 << 33,
                    1..=4 => op.in_halves[k - 1] ^= 1 << 15,
                    5 => op.shift ^= 0x80,
                    6 => op.direction ^= 1,
                    7 => op.signed ^= 1,
                    8 => op.word_instr ^= 1,
                    _ => return false,
                }
                true
            },
            "SHIFT",
        );
        let memws: Vec<MemwOp> = (0..4u8)
            .map(|i| MemwOp {
                base_address: next(),
                timestamp: next(),
                old_timestamp: std::array::from_fn(|_| next()),
                value: std::array::from_fn(|_| next() as u32),
                old: std::array::from_fn(|_| next() as u32),
                is_register: i & 1,
                width: 1 << i,
                is_read: (i >> 1) & 1,
            })
            .collect();
        each_field(
            &memws,
            |p| Records(Arc::new(Pieces::Memw(p))).digest(),
            |op, k| {
                match k {
                    0 => op.base_address ^= 1,
                    1 => op.timestamp ^= 1 << 63,
                    2..=9 => op.old_timestamp[k - 2] ^= 1 << (k * 7),
                    10..=17 => op.value[k - 10] ^= 1 << 31,
                    18..=25 => op.old[k - 18] ^= 1,
                    26 => op.is_register ^= 1,
                    27 => op.width ^= 0x80,
                    28 => op.is_read ^= 1,
                    _ => return false,
                }
                true
            },
            "MEMW",
        );
    }

    /// A fill's bytes and count are its pieces' summed. (Two fills compare
    /// equal only when their pieces lie alike; their digests match however
    /// the pieces lie.)
    #[test]
    fn a_fill_is_its_pieces_in_order() {
        let decode = decode();
        let logs = logs(6);
        let split = RecordsFill::Cpu {
            logs: vec![&logs[..2], &logs[2..]],
            first_cycle: 0,
            decode: &decode,
        };
        assert_eq!((split.len(), split.bytes()), (6, 240));
        let rows = [RegisterRow::default(); 3];
        let memw_r = RecordsFill::MemwR {
            rows: vec![&rows[..1], &rows[1..]],
        };
        assert_eq!((memw_r.len(), memw_r.bytes()), (3, 96));
        assert!(!memw_r.is_empty());
        assert!(RecordsFill::MemwR { rows: Vec::new() }.is_empty());
        let aligned = [AlignedRow::default(); 3];
        let memw_a = RecordsFill::MemwA {
            rows: vec![&aligned[..2], &aligned[2..]],
        };
        assert_eq!((memw_a.len(), memw_a.bytes()), (3, 144));
        let ops = [LoadOp::default(); 5];
        let load = RecordsFill::Load {
            ops: vec![&ops[..1], &ops[1..3], &ops[3..]],
        };
        assert_eq!((load.len(), load.bytes()), (5, 160));
    }
}
