//! A table's columns filled on the card from the records its rows come from
//! (`kernels/fill.cu`), instead of filled packed on the host and widened on the
//! card ([`crate::narrow`]): MEMW_R from its register rows, CPU from the
//! executor's logs and the program's decode table, KECCAK_RND from its keccak
//! operations.
//!
//! Each kernel writes the words the prover's host fill writes for the same
//! rows, padding rows included, column-major (`out[col · height + row]`): a
//! table's run of columns in the resident store ([`crate::columns`]), the
//! layout [`crate::narrow::widen_col_major_to_host`] writes. The records cross
//! the bus as the `#[repr(C)]` structs below, which are the kernels' structs
//! field for field (their sizes and offsets are asserted on both sides).
//!
//! Each wrapper uploads its records from host slices the way the widen path
//! uploads packed bytes (a pageable copy on the destination's stream), launches
//! its kernel into the destination and returns the upload's and the kernel's
//! event-timed milliseconds ([`FillTimes`]). [`widen_col_major_timed`] is
//! today's path timed the same way, for the comparison.
//!
//! The module is loaded on first use from its own cubin, never by
//! [`crate::device::Backend::init`].

use std::mem::{offset_of, size_of};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaViewMut, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::Result;
use crate::device::backend;
use crate::narrow::NarrowInput;

const FILL_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fill.cubin"));

/// MEMW_R's columns (`memw_register::cols::NUM_COLUMNS`).
pub const MEMW_R_COLUMNS: usize = 10;
/// CPU's columns (`cpu::cols::NUM_COLUMNS`).
pub const CPU_COLUMNS: usize = 37;
/// KECCAK_RND's columns (`keccak_rnd::cols::NUM_COLUMNS`).
pub const KECCAK_RND_COLUMNS: usize = 1480;
/// Rows a keccak operation fills.
pub const KECCAK_RND_ROWS_PER_OP: usize = 24;

/// The column sections the KECCAK_RND kernel's grid splits a row into
/// (`FILL_KECCAK_SECTIONS`).
const KECCAK_SECTIONS: u32 = 3;
/// Threads a block for the row kernels.
const MEMW_R_BLOCK: u32 = 256;
const CPU_BLOCK: u32 = 256;
const KECCAK_BLOCK: u32 = 128;

/// The executor's log of one cycle (`executor::vm::logs::Log`), field for
/// field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogC {
    pub current_pc: u64,
    pub next_pc: u64,
    pub src1_val: u64,
    pub src2_val: u64,
    pub dst_val: u64,
}

/// One MEMW_R row (the prover's `RegRow`): `address` the register index,
/// `old_ts_lo` the old timestamp's low word, `is_read` 0 or 1, `pad` 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegRowC {
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

/// One DECODE row (the prover's `DecodeEntry`): the pc, the extended
/// immediate, then `ShrunkDecode`'s fields as bytes (flags 0 or 1).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeEntryC {
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

/// A run of pcs 4 apart (the prover's `DecodeTable` runs): pc
/// `first_pc + 4k` is DECODE row `row + k` for `k < len`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeRunC {
    pub first_pc: u64,
    pub row: u32,
    pub len: u32,
}

/// One keccak permutation call (the prover's `KeccakRoundOperation`): its 24
/// rows are the rounds from `input`. `output` is carried, not read.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeccakOpC {
    pub timestamp: u64,
    pub input: [u64; 25],
    pub output: [u64; 25],
}

/// The same call as a table held as its keccak operations carries it
/// (`multilinear::records::KeccakRndOp`): the fields its rows read.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeccakOpInC {
    pub timestamp: u64,
    pub input: [u64; 25],
}

const _: () = {
    assert!(size_of::<LogC>() == 40);
    assert!(offset_of!(LogC, dst_val) == 32);
    assert!(size_of::<RegRowC>() == 32);
    assert!(offset_of!(RegRowC, old_ts_lo) == 24);
    assert!(offset_of!(RegRowC, address) == 28);
    assert!(offset_of!(RegRowC, is_read) == 30);
    assert!(size_of::<DecodeEntryC>() == 32);
    assert!(offset_of!(DecodeEntryC, read_register1) == 16);
    assert!(offset_of!(DecodeEntryC, mem_flags) == 31);
    assert!(size_of::<DecodeRunC>() == 16);
    assert!(offset_of!(DecodeRunC, len) == 12);
    assert!(size_of::<KeccakOpC>() == 408);
    assert!(offset_of!(KeccakOpC, output) == 208);
    assert!(size_of::<KeccakOpInC>() == 208);
    assert!(offset_of!(KeccakOpInC, input) == 8);
};

// SAFETY: each is `#[repr(C)]` plain data with no padding the kernels read
// (`RegRowC::pad` is a named field), so its bytes on the card are the struct
// the kernel declares.
unsafe impl DeviceRepr for LogC {}
unsafe impl DeviceRepr for RegRowC {}
unsafe impl DeviceRepr for DecodeEntryC {}
unsafe impl DeviceRepr for DecodeRunC {}
unsafe impl DeviceRepr for KeccakOpC {}
unsafe impl DeviceRepr for KeccakOpInC {}

impl DecodeRunC {
    /// The DECODE row of `pc` in `runs` (ascending and disjoint), as the CPU
    /// kernel finds it: the last run starting at or below `pc`, if it holds it.
    pub fn find(runs: &[DecodeRunC], pc: u64) -> Option<u32> {
        let at = runs.partition_point(|run| run.first_pc <= pc);
        let run = runs.get(at.checked_sub(1)?)?;
        let offset = pc - run.first_pc;
        (offset.is_multiple_of(4) && offset / 4 < u64::from(run.len))
            .then(|| run.row + (offset / 4) as u32)
    }
}

/// The event-timed phases of one fill or widen, on its stream.
#[derive(Clone, Copy, Debug)]
pub struct FillTimes {
    /// The records' (or packed bytes') copy to the card, its allocation
    /// included.
    pub upload_ms: f64,
    /// The kernel.
    pub kernel_ms: f64,
    /// The bytes copied.
    pub upload_bytes: usize,
}

struct Module {
    _module: Arc<CudaModule>,
    memw_r: CudaFunction,
    cpu: CudaFunction,
    keccak_rnd: CudaFunction,
    keccak_rnd_in: CudaFunction,
}

fn module() -> Result<&'static Module> {
    static MODULE: OnceLock<Module> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let be = backend()?;
    let module = be.ctx.load_module(Ptx::from_binary(FILL_CUBIN.to_vec()))?;
    let m = Module {
        memw_r: module.load_function("fill_memw_r")?,
        cpu: module.load_function("fill_cpu_from_logs")?,
        keccak_rnd: module.load_function("fill_keccak_rnd")?,
        keccak_rnd_in: module.load_function("fill_keccak_rnd_in")?,
        _module: module,
    };
    Ok(MODULE.get_or_init(|| m))
}

/// Three timing events on a stream: queued, uploaded, done.
struct Clock {
    start: CudaEvent,
    uploaded: CudaEvent,
    done: CudaEvent,
}

impl Clock {
    /// Events that time, the first recorded now.
    fn start(stream: &Arc<CudaStream>) -> Result<Self> {
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ctx = stream.context();
        let clock = Self {
            start: ctx.new_event(flags)?,
            uploaded: ctx.new_event(flags)?,
            done: ctx.new_event(flags)?,
        };
        clock.start.record(stream)?;
        Ok(clock)
    }

    /// Waits for the work queued so far and reads the two phases.
    fn finish(&self, upload_bytes: usize) -> Result<FillTimes> {
        Ok(FillTimes {
            upload_ms: f64::from(self.start.elapsed_ms(&self.uploaded)?),
            kernel_ms: f64::from(self.uploaded.elapsed_ms(&self.done)?),
            upload_bytes,
        })
    }
}

/// `records` on the card: a pageable copy on `stream`, as the widen path
/// copies packed bytes. An empty list uploads one default record so the
/// kernel gets a pointer; it reads none.
fn upload<T: DeviceRepr + Default>(
    stream: &Arc<CudaStream>,
    records: &[T],
) -> Result<CudaSlice<T>> {
    if records.is_empty() {
        stream.clone_htod(std::slice::from_ref(&T::default()))
    } else {
        stream.clone_htod(records)
    }
}

fn row_grid(rows: usize, block: u32) -> Result<u32> {
    u32::try_from(rows.div_ceil(block as usize)).map_err(|_| crate::invalid_value())
}

/// MEMW_R's `height` rows into `dst` (`height × MEMW_R_COLUMNS` words,
/// column-major): `rows`, then zero rows. All on `stream`; returns once the
/// kernel is done.
pub fn fill_memw_r_into(
    stream: &Arc<CudaStream>,
    rows: &[RegRowC],
    height: usize,
    dst: &mut CudaViewMut<u64>,
) -> Result<FillTimes> {
    ensure!(rows.len() <= height, "more rows than the table");
    ensure_eq!(dst.len(), height * MEMW_R_COLUMNS, "the table's words");
    if height == 0 {
        return Ok(FillTimes {
            upload_ms: 0.0,
            kernel_ms: 0.0,
            upload_bytes: 0,
        });
    }
    let m = module()?;
    let clock = Clock::start(stream)?;
    let records = upload(stream, rows)?;
    clock.uploaded.record(stream)?;
    let (n, height_u64) = (rows.len() as u64, height as u64);
    unsafe {
        stream
            .launch_builder(&m.memw_r)
            .arg(&records)
            .arg(&n)
            .arg(&height_u64)
            .arg(dst)
            .launch(LaunchConfig {
                grid_dim: (row_grid(height, MEMW_R_BLOCK)?, 1, 1),
                block_dim: (MEMW_R_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    clock.done.record(stream)?;
    clock.finish(size_of_val(rows))
}

/// A program's decode table on the card, uploaded once: the entries in pc
/// order and the runs of pcs 4 apart.
pub struct CardDecode {
    entries: CudaSlice<DecodeEntryC>,
    runs: CudaSlice<DecodeRunC>,
    nruns: u32,
}

impl CardDecode {
    /// `entries` and `runs` on `stream`. Refused unless the runs are ascending
    /// and disjoint and every row they name is an entry.
    pub fn upload(
        stream: &Arc<CudaStream>,
        entries: &[DecodeEntryC],
        runs: &[DecodeRunC],
    ) -> Result<Self> {
        ensure!(!entries.is_empty() && !runs.is_empty(), "a program");
        let nruns = u32::try_from(runs.len()).map_err(|_| crate::invalid_value())?;
        let mut end_pc = 0u64;
        for (k, run) in runs.iter().enumerate() {
            ensure!(run.len > 0, "a run holds a pc");
            ensure!(
                k == 0 || run.first_pc > end_pc,
                "runs ascending and disjoint"
            );
            ensure!(
                run.row as usize + run.len as usize <= entries.len(),
                "a run's rows are entries"
            );
            end_pc = run
                .first_pc
                .checked_add(4 * (u64::from(run.len) - 1))
                .ok_or_else(crate::invalid_value)?;
        }
        Ok(Self {
            entries: stream.clone_htod(entries)?,
            runs: stream.clone_htod(runs)?,
            nruns,
        })
    }
}

/// CPU's `height` rows into `dst` (`height × CPU_COLUMNS` words,
/// column-major): one a log, read with its entry of `decode`, then the padding
/// rows, every row's timestamp the table's own `4·(row + 1)`. Refused when a log's pc has
/// no DECODE row (the host fill's `MissingInstruction`). All on `stream`;
/// returns once the kernel is done.
pub fn fill_cpu_from_logs_into(
    stream: &Arc<CudaStream>,
    logs: &[LogC],
    decode: &CardDecode,
    height: usize,
    dst: &mut CudaViewMut<u64>,
) -> Result<FillTimes> {
    ensure!(logs.len() <= height, "more logs than the table");
    ensure_eq!(dst.len(), height * CPU_COLUMNS, "the table's words");
    if height == 0 {
        return Ok(FillTimes {
            upload_ms: 0.0,
            kernel_ms: 0.0,
            upload_bytes: 0,
        });
    }
    let m = module()?;
    let mut missing = stream.alloc_zeros::<u64>(1)?;
    let clock = Clock::start(stream)?;
    let records = upload(stream, logs)?;
    clock.uploaded.record(stream)?;
    let (n, height_u64) = (logs.len() as u64, height as u64);
    unsafe {
        stream
            .launch_builder(&m.cpu)
            .arg(&records)
            .arg(&n)
            .arg(&decode.entries)
            .arg(&decode.runs)
            .arg(&decode.nruns)
            .arg(&height_u64)
            .arg(dst)
            .arg(&mut missing)
            .launch(LaunchConfig {
                grid_dim: (row_grid(height, CPU_BLOCK)?, 1, 1),
                block_dim: (CPU_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    clock.done.record(stream)?;
    let times = clock.finish(size_of_val(logs))?;
    let missing = stream.clone_dtoh(&missing)?;
    stream.synchronize()?;
    ensure_eq!(missing[0], 0, "every log's pc has a DECODE row");
    Ok(times)
}

/// KECCAK_RND's rows `[skip, skip + rows)` of the table `ops` make (24 rows an
/// op) into `dst` (`rows × KECCAK_RND_COLUMNS` words, column-major), rows past
/// the last op zero. All on `stream`; returns once the kernel is done.
pub fn fill_keccak_rnd_into(
    stream: &Arc<CudaStream>,
    ops: &[KeccakOpC],
    skip: usize,
    rows: usize,
    dst: &mut CudaViewMut<u64>,
) -> Result<FillTimes> {
    ensure_eq!(dst.len(), rows * KECCAK_RND_COLUMNS, "the table's words");
    let nops = u32::try_from(ops.len()).map_err(|_| crate::invalid_value())?;
    if rows == 0 {
        return Ok(FillTimes {
            upload_ms: 0.0,
            kernel_ms: 0.0,
            upload_bytes: 0,
        });
    }
    let m = module()?;
    let clock = Clock::start(stream)?;
    let records = upload(stream, ops)?;
    clock.uploaded.record(stream)?;
    launch_keccak_rnd(stream, &m.keccak_rnd, &records, nops, skip, rows, dst)?;
    clock.done.record(stream)?;
    clock.finish(size_of_val(ops))
}

/// `fill_keccak_rnd`'s launch (or `fill_keccak_rnd_in`'s, as `function` says)
/// over records already on the card.
fn launch_keccak_rnd<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    records: &CudaSlice<T>,
    nops: u32,
    skip: usize,
    rows: usize,
    dst: &mut CudaViewMut<u64>,
) -> Result<()> {
    let (skip_u64, rows_u64) = (skip as u64, rows as u64);
    unsafe {
        stream
            .launch_builder(function)
            .arg(records)
            .arg(&nops)
            .arg(&skip_u64)
            .arg(&rows_u64)
            .arg(dst)
            .launch(LaunchConfig {
                grid_dim: (row_grid(rows, KECCAK_BLOCK)?, KECCAK_SECTIONS, 1),
                block_dim: (KECCAK_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

/// A table's records as they go up for the card to fill its columns from
/// them ([`crate::columns::TableUpload::Records`]).
#[derive(Clone, Copy, Debug)]
pub enum RecordsInput<'a> {
    /// KECCAK_RND's rows `[skip, skip + rows)` of the table `ops` make, 24
    /// rows an op; rows past the last op zero.
    KeccakRnd {
        ops: &'a [KeccakOpInC],
        skip: usize,
        rows: usize,
    },
}

impl RecordsInput<'_> {
    /// Rows of the table.
    pub fn rows(&self) -> usize {
        match self {
            Self::KeccakRnd { rows, .. } => *rows,
        }
    }

    /// Columns of the table.
    pub fn cols(&self) -> usize {
        match self {
            Self::KeccakRnd { .. } => KECCAK_RND_COLUMNS,
        }
    }

    /// The records' bytes.
    pub fn bytes(&self) -> usize {
        match self {
            Self::KeccakRnd { ops, .. } => size_of_val(*ops),
        }
    }
}

/// `records`' table into `dst` (`rows × cols` words, column-major), all on
/// `stream`: the records copied up as the widen path copies packed bytes and
/// the family's kernel launched, the copy freed in stream order, so a reader
/// on the same stream needs no further ordering.
pub(crate) fn fill_col_major_into(
    stream: &Arc<CudaStream>,
    records: RecordsInput<'_>,
    dst: &mut CudaViewMut<u64>,
) -> Result<()> {
    ensure_eq!(
        dst.len(),
        records.rows() * records.cols(),
        "the table's words"
    );
    if records.rows() == 0 {
        return Ok(());
    }
    let m = module()?;
    match records {
        RecordsInput::KeccakRnd { ops, skip, rows } => {
            let nops = u32::try_from(ops.len()).map_err(|_| crate::invalid_value())?;
            let ops_dev = upload(stream, ops)?;
            launch_keccak_rnd(stream, &m.keccak_rnd_in, &ops_dev, nops, skip, rows, dst)
        }
    }
}

/// Today's path, timed as the fills are: `packed`'s bytes, offsets and widths
/// uploaded, then widened column-major into `dst` (`rows × cols` words) by the
/// kernel [`crate::columns::DeviceColumns::upload_tables`] launches. All on
/// `stream`; returns once the kernel is done.
pub fn widen_col_major_timed(
    stream: &Arc<CudaStream>,
    packed: NarrowInput,
    dst: &mut CudaViewMut<u64>,
) -> Result<FillTimes> {
    let (rows, cols) = (packed.rows(), packed.cols());
    let cells = rows * cols;
    ensure_eq!(dst.len(), cells, "the table's words");
    let cells_u32 = u32::try_from(cells).map_err(|_| crate::invalid_value())?;
    if cells == 0 {
        return Ok(FillTimes {
            upload_ms: 0.0,
            kernel_ms: 0.0,
            upload_bytes: 0,
        });
    }
    let be = backend()?;
    let clock = Clock::start(stream)?;
    let (data, offsets, widths) = packed.parts();
    let data_dev = stream.clone_htod(data)?;
    let offsets_dev = stream.clone_htod(offsets)?;
    let widths_dev = stream.clone_htod(widths)?;
    clock.uploaded.record(stream)?;
    let (rows_u64, cols_u64) = (rows as u64, cols as u64);
    unsafe {
        stream
            .launch_builder(&be.widen_narrow_col_major)
            .arg(&data_dev)
            .arg(&offsets_dev)
            .arg(&widths_dev)
            .arg(dst)
            .arg(&rows_u64)
            .arg(&cols_u64)
            .launch(LaunchConfig::for_num_elems(cells_u32))?;
    }
    clock.done.record(stream)?;
    clock.finish(size_of_val(data) + size_of_val(offsets) + size_of_val(widths))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs() -> Vec<DecodeRunC> {
        vec![
            DecodeRunC {
                first_pc: 0x1000,
                row: 0,
                len: 3,
            },
            DecodeRunC {
                first_pc: 0x2000,
                row: 3,
                len: 1,
            },
            DecodeRunC {
                first_pc: 0x2008,
                row: 4,
                len: 2,
            },
        ]
    }

    /// Every pc of the runs finds its row, and the pcs around them none.
    #[test]
    fn a_pc_finds_the_row_of_its_run() {
        let runs = runs();
        let want = [
            (0x1000, Some(0)),
            (0x1004, Some(1)),
            (0x1008, Some(2)),
            (0x2000, Some(3)),
            (0x2008, Some(4)),
            (0x200c, Some(5)),
        ];
        for (pc, row) in want {
            assert_eq!(DecodeRunC::find(&runs, pc), row, "pc {pc:#x}");
        }
        for pc in [0, 0xffc, 0x1002, 0x100c, 0x1ffc, 0x2004, 0x2010, u64::MAX] {
            assert_eq!(DecodeRunC::find(&runs, pc), None, "pc {pc:#x}");
        }
        assert_eq!(DecodeRunC::find(&[], 0x1000), None);
    }
}
