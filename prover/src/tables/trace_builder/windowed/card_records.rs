//! A streamed CPU or MEMW_R chunk held as the records its rows come from
//! (`multilinear::records`), for the card to fill its columns as its group
//! uploads it: a CPU chunk as the executor's logs of its cycles, a MEMW_R
//! chunk as its register rows, a MEMW_A, LOAD, STORE, MEMW or SHIFT chunk as
//! its rows or ops ([`RowRecords`]), each in the window parts the chunk was cut from, shared
//! and never copied. A host reader gets the host fill of the same
//! rows, which writes the same words.

use std::sync::Arc;

use executor::vm::logs::Log;
use multilinear::narrow::NarrowColumns;
use multilinear::records::{
    ALIGNED_ECALL, AlignedRef, AlignedRefPart, AlignedRefs, DecodeRecords, PagesReturned, Records,
    RecordsFill, TableRecords,
};

use super::history_rows::RefWindow;
use super::history_sides::{HistoryParts, HistorySides};
use super::{Segments, StreamTable, Table};
use crate::tables::decode::DecodeTable;
use crate::tables::gpack::TraceForm;
use crate::tables::load::LoadOperation;
use crate::tables::memw::MemwOperation;
use crate::tables::memw_aligned::AlignedRow;
use crate::tables::memw_register::RegRow;
use crate::tables::shift::ShiftOperation;
use crate::tables::store::StoreOperation;
use crate::tables::{cpu, load, memw, memw_aligned, memw_register, shift, store};

/// The program's DECODE table for the CPU chunks held as logs: the host's,
/// which a host reader fills with, and its rows as the card reads them.
pub(crate) struct CardDecode {
    pub(crate) table: Arc<DecodeTable>,
    pub(crate) records: DecodeRecords,
}

impl CardDecode {
    /// `table` and its rows as the card reads them; `None` for a program with
    /// no instruction.
    pub(crate) fn new(table: Arc<DecodeTable>) -> Option<Self> {
        let records = table.card_records()?;
        Some(Self { table, records })
    }
}

impl std::fmt::Debug for CardDecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CardDecode({:?})", self.records)
    }
}

/// A trace held as `records`. A trace refuses records only over another field
/// than Goldilocks or for a table of no columns, neither of which a streamed
/// table is; were it refused, the trace is the host fill's columns, packed.
fn held(records: Records) -> Table {
    Table::try_from_records_main(records, 1).unwrap_or_else(|records| {
        let (rows, cols) = (records.rows(), records.cols());
        records
            .host_fill()
            .and_then(|packed| Table::try_from_narrow_main(packed, 1).ok())
            .unwrap_or_else(|| Table::new_main(vec![Default::default(); rows * cols], cols, 1))
    })
}

/// Whether `rows` rows are a whole streamed table: a power of two of at least
/// four, the height its host fill pads them to.
pub(super) fn whole_table(rows: usize) -> bool {
    rows.is_power_of_two() && rows >= 4
}

/// A full CPU chunk held as the logs of its cycles, from `first_cycle` on.
pub(crate) struct CpuLogRecords {
    logs: Segments<Log>,
    /// Under the in-row register format, one side record a log, in the same
    /// order (in pieces of their own); empty otherwise.
    sides: Segments<cpu::CpuSide>,
    /// Under the in-row register format with the side records derived on the
    /// card: what they are derived from (`sides` is then empty).
    history: Option<HistorySides>,
    first_cycle: u64,
    decode: Arc<CardDecode>,
}

impl CpuLogRecords {
    /// The chunk the cycles of `logs` from `first_cycle` on make. Refused
    /// (`None`) unless they are a whole table: a chunk of another height goes
    /// to the host's fill, which pads it.
    pub(crate) fn new(
        logs: Segments<Log>,
        first_cycle: u64,
        decode: Arc<CardDecode>,
    ) -> Option<Self> {
        whole_table(logs.len()).then_some(Self {
            logs,
            sides: Segments { parts: Vec::new() },
            history: None,
            first_cycle,
            decode,
        })
    }

    /// [`Self::new`] under the in-row register format: the chunk's logs and
    /// their side records, one a log (refused otherwise), the table's five
    /// more columns filled from them.
    pub(crate) fn in_row(
        logs: Segments<Log>,
        sides: Segments<cpu::CpuSide>,
        first_cycle: u64,
        decode: Arc<CardDecode>,
    ) -> Option<Self> {
        (sides.len() == logs.len() && sides.len() > 0)
            .then(|| Self::new(logs, first_cycle, decode))
            .flatten()
            .map(|records| Self { sides, ..records })
    }

    /// [`Self::in_row`] with the side records derived on the card from the
    /// executor's history: `extras` (one a log) and `exceptions` (ascending
    /// by cycle, every row whose `rd` old value the logs do not give;
    /// [`super::history_sides`]). Refused unless there is one record a log.
    pub(crate) fn in_row_from_history(
        logs: Segments<Log>,
        extras: HistoryParts,
        exceptions: Vec<multilinear::records::SideException>,
        first_cycle: u64,
        decode: Arc<CardDecode>,
    ) -> Option<Self> {
        (extras.len() == logs.len() && extras.len() > 0)
            .then(|| Self::new(logs, first_cycle, decode))
            .flatten()
            .map(|records| Self {
                history: Some(HistorySides::new(extras, exceptions)),
                ..records
            })
    }

    /// The table held as these records.
    pub(crate) fn into_trace(self) -> Table {
        held(Records(Arc::new(self)))
    }

    /// The host's DECODE table its CPU ops are read with.
    pub(crate) fn decode_table(&self) -> &DecodeTable {
        &self.decode.table
    }

    /// Its side records, under the in-row register format (those the host
    /// derives, where the card derives them); `None` otherwise, and where the
    /// derivation refused a row.
    pub(crate) fn sides(&self) -> Option<&Segments<cpu::CpuSide>> {
        match &self.history {
            Some(h) => h.derived(&self.logs, self.first_cycle, &self.decode.table),
            None => (self.sides.len() > 0).then_some(&self.sides),
        }
    }

    /// A test's fault: the same chunk from a copy of its logs, each log's
    /// first source value (`src1_val`, the `rv1` of nearly every row) and
    /// written value (`dst_val`) XORed with `by`, in one piece: a CPU table
    /// the run did not make, the same however it is filled.
    #[cfg(test)]
    pub(crate) fn with_register_values_xored(self, by: u64) -> Self {
        let logs: Vec<Log> = self
            .logs
            .slices()
            .iter()
            .flat_map(|piece| piece.iter())
            .map(|log| Log {
                src1_val: log.src1_val ^ by,
                dst_val: log.dst_val ^ by,
                ..log.clone()
            })
            .collect();
        Self {
            logs: Segments::one(logs),
            ..self
        }
    }
}

impl std::fmt::Debug for CpuLogRecords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CPU logs: {} cycles from {} in {} pieces",
            self.logs.len(),
            self.first_cycle,
            self.logs.parts.len()
        )
    }
}

impl TableRecords for CpuLogRecords {
    fn rows(&self) -> usize {
        self.logs.len()
    }

    fn cols(&self) -> usize {
        if self.sides.len() == 0 && self.history.is_none() {
            cpu::cols::NUM_COLUMNS
        } else {
            cpu::inrow_cols::NUM_COLUMNS_INROW
        }
    }

    fn fill(&self) -> RecordsFill<'_> {
        RecordsFill::Cpu {
            logs: self.logs.slices().into_iter().map(cpu::card_logs).collect(),
            sides: self.sides.slices(),
            first_cycle: self.first_cycle,
            decode: &self.decode.records,
            history: self
                .history
                .as_ref()
                .map(|h| multilinear::records::SidesFromHistory {
                    stamps: h
                        .extras
                        .slices()
                        .into_iter()
                        .map(super::history_sides::card_stamps)
                        .collect(),
                    exceptions: vec![&h.exceptions[..]],
                }),
        }
    }

    fn bytes(&self) -> usize {
        self.logs.op_bytes()
            + self.sides.op_bytes()
            + self.history.as_ref().map_or(0, |h| {
                h.extras.bytes() + std::mem::size_of_val(&h.exceptions[..])
            })
    }

    /// Where the card derives the side records, the records as if they had
    /// been held: the logs and the side records the host derives, so a chunk
    /// rebuilt either way digests alike.
    fn digest_fill(&self) -> Option<RecordsFill<'_>> {
        self.history.as_ref()?;
        let sides = self.sides()?;
        Some(RecordsFill::Cpu {
            logs: self.logs.slices().into_iter().map(cpu::card_logs).collect(),
            sides: sides.slices(),
            first_cycle: self.first_cycle,
            decode: &self.decode.records,
            history: None,
        })
    }

    /// TIMESTAMP, `4·(row + 1)`: the fill writes it from the row index, as the
    /// CPU table's preprocessed column.
    fn row_index_columns(&self) -> usize {
        cpu::NUM_PRECOMPUTED_COLS
    }

    /// The CPU ops the walk made of these logs, filled as the host fills a
    /// streamed CPU chunk. `None` when a log's pc has no DECODE row (the walk
    /// refused it already) or a cycle is past the clock's range.
    fn host_fill(&self) -> Option<NarrowColumns> {
        let mut ops = Vec::with_capacity(self.logs.len());
        let mut first = usize::try_from(self.first_cycle).ok()?;
        for piece in self.logs.slices() {
            super::super::collect_cpu_ops_into(piece, &self.decode.table, first, &mut ops).ok()?;
            first += piece.len();
        }
        let mut trace = match (self.sides.len(), &self.history) {
            (0, None) => cpu::generate_cpu_trace_as(&ops, &self.decode.table, TraceForm::Narrow),
            _ => cpu::generate_cpu_trace_inrow_segments_as(
                &[&ops],
                &self.sides()?.slices(),
                &self.decode.table,
                TraceForm::Narrow,
            ),
        };
        // Written packed, except in a build that holds every trace wide.
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }

    fn return_pages(&mut self) -> PagesReturned {
        let mut got = self.logs.return_pages();
        got += self.sides.return_pages();
        got
    }
}

/// A full MEMW_R chunk held as its register rows.
pub(crate) struct RegisterRowRecords {
    /// Its rows: as given, or decoded from `words` the first time a host
    /// reader asks (an error when they cannot be: the run is then refused at
    /// its finish, and the readers here get no rows, never a panic).
    rows: std::sync::OnceLock<Result<Segments<RegRow>, String>>,
    /// Words mode: the chunk as the walk wrote it, and the program's DECODE
    /// rows its rows decode with.
    words: Option<(super::WordChunk, Arc<DecodeTable>)>,
}

impl RegisterRowRecords {
    /// The chunk `rows` make. Refused (the rows given back) unless they are a
    /// whole table: a chunk of another height goes to the host's fill, which
    /// pads it.
    pub(crate) fn new(rows: Segments<RegRow>) -> Result<Self, Segments<RegRow>> {
        if whole_table(rows.len()) {
            Ok(Self {
                rows: std::sync::OnceLock::from(Ok(rows)),
                words: None,
            })
        } else {
            Err(rows)
        }
    }

    /// A full chunk in words mode: the card builds its rows from its words
    /// and the logs it holds; every host reader gets them decoded.
    pub(crate) fn from_words(chunk: super::WordChunk, decode: Arc<DecodeTable>) -> Self {
        Self {
            rows: std::sync::OnceLock::new(),
            words: Some((chunk, decode)),
        }
    }

    /// Its rows (decoded once, in words mode), or why they cannot be.
    fn rows_held(&self) -> Result<&Segments<RegRow>, &str> {
        self.rows
            .get_or_init(|| match self.words.as_ref() {
                Some((chunk, decode)) => chunk
                    .decode(decode, 0..chunk.len())
                    .map(Segments::one)
                    .map_err(|e| format!("a MEMW_R chunk's words: {e}")),
                None => Err("a MEMW_R chunk with neither rows nor words".to_string()),
            })
            .as_ref()
            .map_err(String::as_str)
    }

    /// Its row count, without decoding.
    fn len(&self) -> usize {
        match (&self.words, self.rows.get()) {
            (Some((chunk, _)), _) => chunk.len(),
            (None, Some(Ok(rows))) => rows.len(),
            (None, _) => 0,
        }
    }

    /// The table held as these records.
    pub(crate) fn into_trace(self) -> Table {
        held(Records(Arc::new(self)))
    }

    /// The rows back.
    pub(crate) fn into_rows(self) -> Segments<RegRow> {
        // None when they cannot be decoded: the regenerated chunk's digest
        // then differs and its slot refuses it.
        let _ = self.rows_held();
        match self.rows.into_inner() {
            Some(Ok(rows)) => rows,
            _ => Segments { parts: Vec::new() },
        }
    }
}

impl std::fmt::Debug for RegisterRowRecords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MEMW_R rows: {} ({})",
            self.len(),
            if self.words.is_some() {
                "words"
            } else {
                "rows"
            }
        )
    }
}

impl TableRecords for RegisterRowRecords {
    fn rows(&self) -> usize {
        self.len()
    }

    fn cols(&self) -> usize {
        memw_register::cols::NUM_COLUMNS
    }

    fn fill(&self) -> RecordsFill<'_> {
        RecordsFill::MemwR {
            rows: self
                .rows_held()
                .map(Segments::slices)
                .unwrap_or_default()
                .into_iter()
                .map(memw_register::card_rows)
                .collect(),
        }
    }

    /// The rows' bytes in either form (words mode reports the rows it decodes
    /// to, so its digest is the rows' digest).
    fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<RegRow>()
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let mut trace = memw_register::generate_memw_register_trace_from_rows_segments_as(
            &self.rows_held().ok()?.slices(),
            TraceForm::Narrow,
        );
        // Written packed, except in a build that holds every trace wide.
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }

    fn return_pages(&mut self) -> PagesReturned {
        match (self.rows.get_mut(), self.words.as_mut()) {
            (Some(Ok(rows)), _) => rows.return_pages(),
            (None, Some((chunk, _))) => chunk.words.return_pages(),
            _ => PagesReturned::default(),
        }
    }

    fn register_words(&self) -> Option<multilinear::records::RegisterWords<'_>> {
        let (chunk, _) = self.words.as_ref()?;
        Some(multilinear::records::RegisterWords {
            words: chunk.words.slices(),
            whole: chunk
                .whole
                .slices()
                .into_iter()
                .map(memw_register::card_rows)
                .collect(),
            whole_base: chunk.whole_base,
        })
    }

    fn decode_register_rows(
        &self,
        from: usize,
        to: usize,
    ) -> Vec<multilinear::records::RegisterRow> {
        // Rows that cannot be decoded come back short: the card's fill then
        // refuses the table, and the run is refused at its finish.
        let Some((chunk, decode)) = self.words.as_ref() else {
            return self
                .rows_held()
                .map(|rows| {
                    rows.slices()
                        .into_iter()
                        .flat_map(memw_register::card_rows)
                        .skip(from)
                        .take(to - from)
                        .copied()
                        .collect()
                })
                .unwrap_or_default();
        };
        chunk
            .decode(decode, from..to)
            .map(|rows| memw_register::card_rows(&rows).to_vec())
            .unwrap_or_default()
    }
}

/// A streamed family whose full chunks may be held as their rows, one record a
/// row, the walk's own structs read in place by the card (MEMW_A's aligned
/// rows, LOAD's, STORE's, MEMW's and SHIFT's ops): the records as the card
/// reads them, and the host fill of the same rows.
pub(crate) trait RowFamily: Sized + Send + Sync + 'static {
    const TABLE: StreamTable;
    const COLUMNS: usize;
    /// The family's records as the card's fill reads them.
    fn fill(pieces: Vec<&[Self]>) -> RecordsFill<'_>;
    /// The host fill of the pieces' rows, in `form`: the table the chunk is
    /// without records.
    fn host_fill(pieces: &[&[Self]], form: TraceForm) -> Table;
    /// A test's fault: the same row, its timestamp `cycles` later.
    #[cfg(test)]
    fn later(&self, cycles: u64) -> Self;
}

impl RowFamily for AlignedRow {
    const TABLE: StreamTable = StreamTable::MemwAligned;
    const COLUMNS: usize = memw_aligned::cols::NUM_COLUMNS;

    fn fill(pieces: Vec<&[Self]>) -> RecordsFill<'_> {
        RecordsFill::MemwA {
            rows: pieces.into_iter().map(memw_aligned::card_rows).collect(),
        }
    }

    fn host_fill(pieces: &[&[Self]], form: TraceForm) -> Table {
        memw_aligned::generate_memw_aligned_trace_segments_as(pieces, form)
    }

    #[cfg(test)]
    fn later(&self, cycles: u64) -> Self {
        AlignedRow::later(*self, cycles)
    }
}

impl RowFamily for LoadOperation {
    const TABLE: StreamTable = StreamTable::Load;
    const COLUMNS: usize = load::cols::NUM_COLUMNS;

    fn fill(pieces: Vec<&[Self]>) -> RecordsFill<'_> {
        RecordsFill::Load {
            ops: pieces.into_iter().map(load::card_ops).collect(),
        }
    }

    fn host_fill(pieces: &[&[Self]], form: TraceForm) -> Table {
        load::generate_load_trace_segments_as(pieces, form)
    }

    #[cfg(test)]
    fn later(&self, cycles: u64) -> Self {
        Self {
            timestamp: self.timestamp + cycles,
            ..self.clone()
        }
    }
}

impl RowFamily for StoreOperation {
    const TABLE: StreamTable = StreamTable::Store;
    const COLUMNS: usize = store::cols::NUM_COLUMNS;

    fn fill(pieces: Vec<&[Self]>) -> RecordsFill<'_> {
        RecordsFill::Store {
            ops: pieces.into_iter().map(store::card_ops).collect(),
        }
    }

    fn host_fill(pieces: &[&[Self]], form: TraceForm) -> Table {
        store::generate_store_trace_segments_as(pieces, form)
    }

    #[cfg(test)]
    fn later(&self, cycles: u64) -> Self {
        Self {
            timestamp: self.timestamp + cycles,
            ..self.clone()
        }
    }
}

impl RowFamily for MemwOperation {
    const TABLE: StreamTable = StreamTable::Memw;
    const COLUMNS: usize = memw::cols::NUM_COLUMNS;

    fn fill(pieces: Vec<&[Self]>) -> RecordsFill<'_> {
        RecordsFill::Memw {
            ops: pieces.into_iter().map(memw::card_ops).collect(),
        }
    }

    fn host_fill(pieces: &[&[Self]], form: TraceForm) -> Table {
        memw::generate_memw_trace_segments_as(pieces, form)
    }

    #[cfg(test)]
    fn later(&self, cycles: u64) -> Self {
        Self {
            timestamp: self.timestamp + cycles,
            ..self.clone()
        }
    }
}

impl RowFamily for ShiftOperation {
    const TABLE: StreamTable = StreamTable::Shift;
    const COLUMNS: usize = shift::cols::NUM_COLUMNS;

    fn fill(pieces: Vec<&[Self]>) -> RecordsFill<'_> {
        RecordsFill::Shift {
            ops: pieces.into_iter().map(shift::card_ops).collect(),
        }
    }

    fn host_fill(pieces: &[&[Self]], form: TraceForm) -> Table {
        shift::generate_shift_trace_segments_as(pieces, form)
    }

    /// SHIFT has no timestamp: the same op with its input's low half moved by
    /// `cycles`, every column computed from that input, a row the CPU never
    /// sent.
    #[cfg(test)]
    fn later(&self, cycles: u64) -> Self {
        let mut op = self.clone();
        op.in_halves[0] = op.in_halves[0].wrapping_add(cycles as u16);
        op
    }
}

/// A full chunk of a [`RowFamily`] held as its rows.
pub(crate) struct RowRecords<T: RowFamily> {
    rows: Segments<T>,
}

impl<T: RowFamily> RowRecords<T> {
    /// The chunk `rows` make. Refused (the rows given back) unless they are a
    /// whole table: a chunk of another height goes to the host's fill, which
    /// pads it.
    pub(crate) fn new(rows: Segments<T>) -> Result<Self, Segments<T>> {
        if whole_table(rows.len()) {
            Ok(Self { rows })
        } else {
            Err(rows)
        }
    }

    /// A test's fault: the same rows in one piece, each one's timestamp
    /// `cycles` later.
    #[cfg(test)]
    pub(crate) fn later(self, cycles: u64) -> Self {
        let pieces = self.rows.slices();
        let rows = pieces.iter().flat_map(|piece| piece.iter());
        Self {
            rows: Segments::one(rows.map(|row| row.later(cycles)).collect()),
        }
    }
}

impl<T: RowFamily> std::fmt::Debug for RowRecords<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} rows: {} in {} pieces",
            T::TABLE,
            self.rows.len(),
            self.rows.parts.len()
        )
    }
}

impl<T: RowFamily> TableRecords for RowRecords<T> {
    fn rows(&self) -> usize {
        self.rows.len()
    }

    fn cols(&self) -> usize {
        T::COLUMNS
    }

    fn fill(&self) -> RecordsFill<'_> {
        T::fill(self.rows.slices())
    }

    fn bytes(&self) -> usize {
        self.rows.op_bytes()
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let mut trace = T::host_fill(&self.rows.slices(), TraceForm::Narrow);
        // Written packed, except in a build that holds every trace wide.
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }

    fn return_pages(&mut self) -> PagesReturned {
        self.rows.return_pages()
    }
}

/// A chunk held as the rows of a [`RowFamily`], as the hand-out carries it:
/// what it weighs, the table it becomes, and the table a regenerator rebuilds
/// with the digest of the records it would have been held as.
pub(crate) trait RowChunk: Send + std::fmt::Debug {
    fn op_bytes(&self) -> usize;
    fn op_count(&self) -> usize;
    /// The table held as these records.
    fn into_trace(self: Box<Self>) -> Table;
    /// With `digest`, the records' digest (`multilinear::records::digest_of`);
    /// and the table's host fill in `form`.
    fn regenerate(self: Box<Self>, form: TraceForm, digest: bool) -> (Option<[u64; 2]>, Table);
}

impl<T: RowFamily> RowChunk for RowRecords<T> {
    fn op_bytes(&self) -> usize {
        self.rows.op_bytes()
    }

    fn op_count(&self) -> usize {
        self.rows.len()
    }

    fn into_trace(self: Box<Self>) -> Table {
        held(Records(Arc::new(*self)))
    }

    fn regenerate(self: Box<Self>, form: TraceForm, digest: bool) -> (Option<[u64; 2]>, Table) {
        let digest = digest.then(|| multilinear::records::digest_of(&*self));
        (digest, T::host_fill(&self.rows.slices(), form))
    }
}

/// A full MEMW_A chunk held as its rows' refs, for the card to derive its
/// rows from the executor's history ([`super::history_rows`]): the refs in
/// the window parts the chunk was cut from, each part beside the window it
/// lies in. Every host reader gets the rows the host derives from them, once,
/// on the first ask; the chunk digests and fills as the same rows held as
/// rows.
// Handed out by the windowed builder under the card-built trace.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct AlignedRefRecords {
    refs: Segments<AlignedRef>,
    /// The window of each part of `refs`, one a part.
    windows: Vec<Arc<RefWindow>>,
    /// Each part's regular rows' access indices and ECALL rows' whole-row
    /// indices, `lo..hi`: what of its window the card reads.
    spans: Vec<(std::ops::Range<u32>, std::ops::Range<u32>)>,
    decode: Arc<CardDecode>,
    order: memw_aligned::MemwAlignedOrder,
    derived: std::sync::OnceLock<Result<Vec<AlignedRow>, String>>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl AlignedRefRecords {
    /// The chunk `refs` (one window of `windows` a part, in order) make, in
    /// the run's `order`. Refused (the refs given back) unless they are a
    /// whole table with one window a part.
    pub(crate) fn new(
        refs: Segments<AlignedRef>,
        windows: Vec<Arc<RefWindow>>,
        decode: Arc<CardDecode>,
        order: memw_aligned::MemwAlignedOrder,
    ) -> Result<Self, Segments<AlignedRef>> {
        if !whole_table(refs.len()) || windows.len() != refs.parts.len() {
            return Err(refs);
        }
        let spans = refs
            .slices()
            .iter()
            .map(|part| {
                let span = |ecall: bool| {
                    let ats = part
                        .iter()
                        .filter(|r| (r.cycle & ALIGNED_ECALL != 0) == ecall)
                        .map(|r| r.at);
                    let (lo, hi) = ats.fold((u32::MAX, 0u32), |(lo, hi), at| {
                        (lo.min(at), hi.max(at.saturating_add(1)))
                    });
                    if lo < hi { lo..hi } else { 0..0 }
                };
                (span(false), span(true))
            })
            .collect();
        Ok(Self {
            refs,
            windows,
            spans,
            decode,
            order,
            derived: std::sync::OnceLock::new(),
        })
    }

    /// The rows as the host derives them, or why they cannot be.
    pub(crate) fn rows(&self) -> Result<&[AlignedRow], &str> {
        self.derived
            .get_or_init(|| {
                let slices = self.refs.slices();
                let parts: Vec<(&[AlignedRef], &RefWindow)> = slices
                    .iter()
                    .zip(&self.windows)
                    .map(|(refs, window)| (*refs, &**window))
                    .collect();
                super::history_rows::aligned_rows_from_refs(&parts, &self.decode.table, self.order)
            })
            .as_deref()
            .map_err(String::as_str)
    }

    /// The table held as these records.
    pub(crate) fn into_trace(self) -> Table {
        held(Records(Arc::new(self)))
    }

    /// The rows back, as the host derives them, or why they cannot be.
    pub(crate) fn into_rows(self) -> Result<Segments<AlignedRow>, String> {
        let _ = self.rows();
        match self.derived.into_inner() {
            Some(Ok(rows)) => Ok(Segments::one(rows)),
            Some(Err(e)) => Err(e),
            None => Err("a MEMW_A chunk's rows were not derived".to_string()),
        }
    }
}

impl std::fmt::Debug for AlignedRefRecords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MEMW_A refs: {} in {} pieces",
            self.refs.len(),
            self.refs.parts.len()
        )
    }
}

impl TableRecords for AlignedRefRecords {
    fn rows(&self) -> usize {
        self.refs.len()
    }

    fn cols(&self) -> usize {
        memw_aligned::cols::NUM_COLUMNS
    }

    /// The rows the host derives (none where it cannot: the table then
    /// digests and fills as no table its AIR takes).
    fn fill(&self) -> RecordsFill<'_> {
        RecordsFill::MemwA {
            rows: self
                .rows()
                .map(|rows| vec![memw_aligned::card_rows(rows)])
                .unwrap_or_default(),
        }
    }

    /// The rows' bytes, as held as rows: the same chunk held either way
    /// digests alike.
    fn bytes(&self) -> usize {
        self.refs.len() * std::mem::size_of::<AlignedRow>()
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let rows = self.rows().ok()?;
        let mut trace =
            memw_aligned::generate_memw_aligned_trace_segments_as(&[rows], TraceForm::Narrow);
        // Written packed, except in a build that holds every trace wide.
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }

    fn return_pages(&mut self) -> PagesReturned {
        self.refs.return_pages()
    }

    fn aligned_refs(&self) -> Option<AlignedRefs<'_>> {
        let parts = self
            .refs
            .slices()
            .into_iter()
            .zip(&self.windows)
            .zip(&self.spans)
            .map(|((refs, window), (accesses, whole))| {
                let h = &window.history;
                AlignedRefPart {
                    refs,
                    accesses: super::history_rows::card_accesses(
                        h.accesses
                            .get(accesses.start as usize..accesses.end as usize)
                            .unwrap_or_default(),
                    ),
                    access_base: accesses.start,
                    mixed: &h.mixed,
                    whole: memw_aligned::card_rows(
                        window
                            .whole
                            .get(whole.start as usize..whole.end as usize)
                            .unwrap_or_default(),
                    ),
                    whole_base: whole.start,
                }
            })
            .collect();
        Some(AlignedRefs {
            parts,
            decode: &self.decode.records,
            limbs: self.order == memw_aligned::MemwAlignedOrder::Limbs,
        })
    }
}

impl RowChunk for AlignedRefRecords {
    fn op_bytes(&self) -> usize {
        self.refs.op_bytes()
    }

    fn op_count(&self) -> usize {
        self.refs.len()
    }

    fn into_trace(self: Box<Self>) -> Table {
        held(Records(Arc::new(*self)))
    }

    fn regenerate(self: Box<Self>, form: TraceForm, digest: bool) -> (Option<[u64; 2]>, Table) {
        let digest = digest.then(|| multilinear::records::digest_of(&*self));
        let rows = self.rows().unwrap_or_default();
        (
            digest,
            memw_aligned::generate_memw_aligned_trace_segments_as(&[rows], form),
        )
    }
}
