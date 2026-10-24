//! A streamed CPU or MEMW_R chunk held as the records its rows come from
//! (`multilinear::records`), for the card to fill its columns as its group
//! uploads it: a CPU chunk as the executor's logs of its cycles, a MEMW_R
//! chunk as its register rows, each in the window parts the chunk was cut
//! from, shared and never copied. A host reader gets the host fill of the same
//! rows, which writes the same words.

use std::sync::Arc;

use executor::vm::logs::Log;
use multilinear::narrow::NarrowColumns;
use multilinear::records::{DecodeRecords, Records, RecordsFill, TableRecords};

use super::{Segments, Table};
use crate::tables::decode::DecodeTable;
use crate::tables::gpack::TraceForm;
use crate::tables::memw_register::RegRow;
use crate::tables::{cpu, memw_register};

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
fn whole_table(rows: usize) -> bool {
    rows.is_power_of_two() && rows >= 4
}

/// A full CPU chunk held as the logs of its cycles, from `first_cycle` on.
pub(crate) struct CpuLogRecords {
    logs: Segments<Log>,
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
            first_cycle,
            decode,
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
        cpu::cols::NUM_COLUMNS
    }

    fn fill(&self) -> RecordsFill<'_> {
        RecordsFill::Cpu {
            logs: self.logs.slices().into_iter().map(cpu::card_logs).collect(),
            first_cycle: self.first_cycle,
            decode: &self.decode.records,
        }
    }

    fn bytes(&self) -> usize {
        self.logs.op_bytes()
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
        let mut trace = cpu::generate_cpu_trace_as(&ops, &self.decode.table, TraceForm::Narrow);
        // Written packed, except in a build that holds every trace wide.
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }
}

/// A full MEMW_R chunk held as its register rows.
pub(crate) struct RegisterRowRecords {
    rows: Segments<RegRow>,
}

impl RegisterRowRecords {
    /// The chunk `rows` make. Refused (the rows given back) unless they are a
    /// whole table: a chunk of another height goes to the host's fill, which
    /// pads it.
    pub(crate) fn new(rows: Segments<RegRow>) -> Result<Self, Segments<RegRow>> {
        if whole_table(rows.len()) {
            Ok(Self { rows })
        } else {
            Err(rows)
        }
    }

    /// The table held as these records.
    pub(crate) fn into_trace(self) -> Table {
        held(Records(Arc::new(self)))
    }

    /// The rows back.
    pub(crate) fn into_rows(self) -> Segments<RegRow> {
        self.rows
    }
}

impl std::fmt::Debug for RegisterRowRecords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MEMW_R rows: {} in {} pieces",
            self.rows.len(),
            self.rows.parts.len()
        )
    }
}

impl TableRecords for RegisterRowRecords {
    fn rows(&self) -> usize {
        self.rows.len()
    }

    fn cols(&self) -> usize {
        memw_register::cols::NUM_COLUMNS
    }

    fn fill(&self) -> RecordsFill<'_> {
        RecordsFill::MemwR {
            rows: self
                .rows
                .slices()
                .into_iter()
                .map(memw_register::card_rows)
                .collect(),
        }
    }

    fn bytes(&self) -> usize {
        self.rows.op_bytes()
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let mut trace = memw_register::generate_memw_register_trace_from_rows_segments_as(
            &self.rows.slices(),
            TraceForm::Narrow,
        );
        // Written packed, except in a build that holds every trace wide.
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }
}
