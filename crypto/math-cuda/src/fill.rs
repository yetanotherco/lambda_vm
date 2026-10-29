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
/// MEMW_A's columns (`memw_aligned::cols::NUM_COLUMNS`).
pub const MEMW_A_COLUMNS: usize = 29;
/// LOAD's columns (`load::cols::NUM_COLUMNS`).
pub const LOAD_COLUMNS: usize = 18;
/// STORE's columns (`store::cols::NUM_COLUMNS`).
pub const STORE_COLUMNS: usize = 16;
/// SHIFT's columns (`shift::cols::NUM_COLUMNS`).
pub const SHIFT_COLUMNS: usize = 29;
/// MEMW's columns (`memw::cols::NUM_COLUMNS`).
pub const MEMW_COLUMNS: usize = 49;
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
const ROW_BLOCK: u32 = 256;
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

/// One MEMW_A row (the prover's `AlignedRow`): the base address, timestamp and
/// shared old timestamp, the value and old value packed a word each, the
/// width, and flags 0 or 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AlignedRowC {
    pub base_address: u64,
    pub timestamp: u64,
    pub old_timestamp: u64,
    pub value: u64,
    pub old: u64,
    pub width: u8,
    pub is_read: u8,
    pub is_register: u8,
}

/// One LOAD op (the prover's `LoadOperation`): the base address, the
/// timestamp, the width, whether it sign-extends (0 or 1) and the result's
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOpC {
    pub base_address: u64,
    pub timestamp: u64,
    pub width: u8,
    pub is_signed: u8,
    pub res: [u8; 8],
}

/// One STORE op (the prover's `StoreOperation`): the base address, the
/// timestamp, the value, and the width flags (0 or 1; none for one byte).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreOpC {
    pub base_address: u64,
    pub timestamp: u64,
    pub value: u64,
    pub write2: u8,
    pub write4: u8,
    pub write8: u8,
}

/// One SHIFT op (the prover's `ShiftOperation`): the full shift amount, the
/// input as four 16-bit halves, the amount's low byte, and flags 0 or 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShiftOpC {
    pub shift_amount: u64,
    pub in_halves: [u16; 4],
    pub shift: u8,
    pub direction: u8,
    pub is_signed: u8,
    pub word_instr: u8,
}

/// One MEMW op (the prover's `MemwOperation`): the base address, the
/// timestamp, the eight old timestamps, the eight value and old elements, and
/// the flags and width.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemwOpC {
    pub base_address: u64,
    pub timestamp: u64,
    pub old_timestamp: [u64; 8],
    pub value: [u32; 8],
    pub old: [u32; 8],
    pub is_register: u8,
    pub width: u8,
    pub is_read: u8,
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
    assert!(size_of::<AlignedRowC>() == 48);
    assert!(offset_of!(AlignedRowC, width) == 40);
    assert!(offset_of!(AlignedRowC, is_register) == 42);
    assert!(size_of::<LoadOpC>() == 32);
    assert!(offset_of!(LoadOpC, is_signed) == 17);
    assert!(offset_of!(LoadOpC, res) == 18);
    assert!(size_of::<StoreOpC>() == 32);
    assert!(offset_of!(StoreOpC, write2) == 24);
    assert!(size_of::<ShiftOpC>() == 24);
    assert!(offset_of!(ShiftOpC, in_halves) == 8);
    assert!(offset_of!(ShiftOpC, word_instr) == 19);
    assert!(size_of::<MemwOpC>() == 152);
    assert!(offset_of!(MemwOpC, value) == 80);
    assert!(offset_of!(MemwOpC, is_register) == 144);
};

// SAFETY: each is `#[repr(C)]` plain data whose padding no kernel reads
// (`RegRowC::pad` is a named field; the row families' structs end in
// padding), so its bytes on the card are the struct the kernel declares.
unsafe impl DeviceRepr for LogC {}
unsafe impl DeviceRepr for RegRowC {}
unsafe impl DeviceRepr for DecodeEntryC {}
unsafe impl DeviceRepr for DecodeRunC {}
unsafe impl DeviceRepr for KeccakOpC {}
unsafe impl DeviceRepr for KeccakOpInC {}
unsafe impl DeviceRepr for AlignedRowC {}
unsafe impl DeviceRepr for LoadOpC {}
unsafe impl DeviceRepr for StoreOpC {}
unsafe impl DeviceRepr for ShiftOpC {}
unsafe impl DeviceRepr for MemwOpC {}

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
    memw_a: CudaFunction,
    load: CudaFunction,
    store: CudaFunction,
    shift: CudaFunction,
    memw: CudaFunction,
    cpu: CudaFunction,
    keccak_rnd: CudaFunction,
    keccak_rnd_in: CudaFunction,
    lfm_p1w16_socket: CudaFunction,
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
        memw_a: module.load_function("fill_memw_a")?,
        load: module.load_function("fill_load")?,
        store: module.load_function("fill_store")?,
        shift: module.load_function("fill_shift")?,
        memw: module.load_function("fill_memw")?,
        cpu: module.load_function("fill_cpu_from_logs")?,
        keccak_rnd: module.load_function("fill_keccak_rnd")?,
        keccak_rnd_in: module.load_function("fill_keccak_rnd_in")?,
        lfm_p1w16_socket: module.load_function("fill_lfm_p1w16_socket")?,
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
                grid_dim: (row_grid(height, ROW_BLOCK)?, 1, 1),
                block_dim: (ROW_BLOCK, 1, 1),
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
    /// What [`Self::resident`] promised the backend for it.
    _room: Option<crate::device::DeviceReservation>,
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
            _room: None,
        })
    }

    /// [`Self::upload`] for every stream's use, under a promise of its bytes
    /// to the backend held as long as it is: uploaded on a stream of the pool
    /// and waited for, so a fill on any stream reads it. Refused when the
    /// backend declines the room.
    pub fn resident(entries: &[DecodeEntryC], runs: &[DecodeRunC]) -> Result<Self> {
        let be = backend()?;
        let bytes = (size_of_val(entries) + size_of_val(runs)) as u64;
        let Some(room) = be.reserve(bytes) else {
            crate::device::note_device_fallback();
            return Err(crate::invalid_value());
        };
        let stream = be.next_stream();
        let mut decode = Self::upload(&stream, entries, runs)?;
        stream.synchronize()?;
        decode._room = Some(room);
        Ok(decode)
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
    let missing = crate::readback::clone_dtoh(stream, &missing)?;
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
#[derive(Clone, Copy)]
pub enum RecordsInput<'a> {
    /// KECCAK_RND's rows `[skip, skip + rows)` of the table `ops` make, 24
    /// rows an op; rows past the last op zero.
    KeccakRnd {
        ops: &'a [KeccakOpInC],
        skip: usize,
        rows: usize,
    },
    /// CPU's `rows` rows: one a log of `logs`' pieces in order, read with its
    /// row of `decode` at the table's own timestamps; then the padding rows.
    Cpu {
        logs: &'a [&'a [LogC]],
        decode: &'a CardDecode,
        rows: usize,
    },
    /// MEMW_R's `rows` rows: `records`' pieces in order, then zero rows.
    MemwR {
        records: &'a [&'a [RegRowC]],
        rows: usize,
    },
    /// MEMW_A's `rows` rows: `records`' pieces in order, then zero rows.
    MemwA {
        records: &'a [&'a [AlignedRowC]],
        rows: usize,
    },
    /// LOAD's `rows` rows: `records`' pieces in order, then zero rows.
    Load {
        records: &'a [&'a [LoadOpC]],
        rows: usize,
    },
    /// STORE's `rows` rows: `records`' pieces in order, then zero rows.
    Store {
        records: &'a [&'a [StoreOpC]],
        rows: usize,
    },
    /// SHIFT's `rows` rows: `records`' pieces in order, then padding rows.
    Shift {
        records: &'a [&'a [ShiftOpC]],
        rows: usize,
    },
    /// MEMW's `rows` rows: `records`' pieces in order, then zero rows.
    Memw {
        records: &'a [&'a [MemwOpC]],
        rows: usize,
    },
}

impl std::fmt::Debug for RecordsInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let family = match self {
            Self::KeccakRnd { .. } => "KECCAK_RND",
            Self::Cpu { .. } => "CPU",
            Self::MemwR { .. } => "MEMW_R",
            Self::MemwA { .. } => "MEMW_A",
            Self::Load { .. } => "LOAD",
            Self::Store { .. } => "STORE",
            Self::Shift { .. } => "SHIFT",
            Self::Memw { .. } => "MEMW",
        };
        write!(
            f,
            "{family} records: {} rows, {} bytes",
            self.rows(),
            self.bytes()
        )
    }
}

impl RecordsInput<'_> {
    /// Rows of the table.
    pub fn rows(&self) -> usize {
        match self {
            Self::KeccakRnd { rows, .. }
            | Self::Cpu { rows, .. }
            | Self::MemwR { rows, .. }
            | Self::MemwA { rows, .. }
            | Self::Load { rows, .. }
            | Self::Store { rows, .. }
            | Self::Shift { rows, .. }
            | Self::Memw { rows, .. } => *rows,
        }
    }

    /// Columns of the table.
    pub fn cols(&self) -> usize {
        match self {
            Self::KeccakRnd { .. } => KECCAK_RND_COLUMNS,
            Self::Cpu { .. } => CPU_COLUMNS,
            Self::MemwR { .. } => MEMW_R_COLUMNS,
            Self::MemwA { .. } => MEMW_A_COLUMNS,
            Self::Load { .. } => LOAD_COLUMNS,
            Self::Store { .. } => STORE_COLUMNS,
            Self::Shift { .. } => SHIFT_COLUMNS,
            Self::Memw { .. } => MEMW_COLUMNS,
        }
    }

    /// The records' bytes (a CPU table's decode table aside: it is on the
    /// card already).
    pub fn bytes(&self) -> usize {
        match self {
            Self::KeccakRnd { ops, .. } => size_of_val(*ops),
            Self::Cpu { logs, .. } => logs.iter().map(|piece| size_of_val(*piece)).sum(),
            Self::MemwR { records, .. } => records.iter().map(|piece| size_of_val(*piece)).sum(),
            Self::MemwA { records, .. } => records.iter().map(|piece| size_of_val(*piece)).sum(),
            Self::Load { records, .. } => records.iter().map(|piece| size_of_val(*piece)).sum(),
            Self::Store { records, .. } => records.iter().map(|piece| size_of_val(*piece)).sum(),
            Self::Shift { records, .. } => records.iter().map(|piece| size_of_val(*piece)).sum(),
            Self::Memw { records, .. } => records.iter().map(|piece| size_of_val(*piece)).sum(),
        }
    }

    /// The records' count: logs, rows or ops.
    fn len(&self) -> usize {
        match self {
            Self::KeccakRnd { ops, .. } => ops.len(),
            Self::Cpu { logs, .. } => logs.iter().map(|piece| piece.len()).sum(),
            Self::MemwR { records, .. } => records.iter().map(|piece| piece.len()).sum(),
            Self::MemwA { records, .. } => records.iter().map(|piece| piece.len()).sum(),
            Self::Load { records, .. } => records.iter().map(|piece| piece.len()).sum(),
            Self::Store { records, .. } => records.iter().map(|piece| piece.len()).sum(),
            Self::Shift { records, .. } => records.iter().map(|piece| piece.len()).sum(),
            Self::Memw { records, .. } => records.iter().map(|piece| piece.len()).sum(),
        }
    }
}

/// What the fills of one upload check once the stream's work is done
/// ([`Self::passed`]): a CPU log whose pc has no DECODE row.
#[derive(Default)]
pub(crate) struct FillChecks {
    missing: Option<CudaSlice<u64>>,
}

impl FillChecks {
    /// The counter of logs with no DECODE row, zeroed on `stream` at first use.
    fn missing(&mut self, stream: &Arc<CudaStream>) -> Result<&mut CudaSlice<u64>> {
        if self.missing.is_none() {
            self.missing = Some(stream.alloc_zeros::<u64>(1)?);
        }
        self.missing.as_mut().ok_or_else(crate::invalid_value)
    }

    /// Whether every fill's check passed, read after `stream` ran them: every
    /// CPU log's pc found its DECODE row (the host fill's
    /// `MissingInstruction` otherwise).
    pub(crate) fn passed(&self, stream: &Arc<CudaStream>) -> Result<bool> {
        let Some(missing) = &self.missing else {
            return Ok(true);
        };
        let missing = crate::readback::clone_dtoh(stream, missing)?;
        stream.synchronize()?;
        Ok(missing.first() == Some(&0))
    }
}

/// `pieces` on the card, one after another in one buffer: each a pageable
/// copy on `stream`, as [`upload`] copies one slice. No record uploads one
/// default record so the kernel gets a pointer; it reads none.
fn upload_pieces<T: DeviceRepr + Default>(
    stream: &Arc<CudaStream>,
    pieces: &[&[T]],
) -> Result<CudaSlice<T>> {
    let n: usize = pieces.iter().map(|piece| piece.len()).sum();
    if n == 0 {
        return upload(stream, &[]);
    }
    if let [piece] = pieces {
        return upload(stream, piece);
    }
    // SAFETY: every element is written by the copies below, which cover
    // `0..n` piece by piece.
    let mut records = unsafe { crate::device::alloc_or_trim::<T>(stream, n) }?;
    let mut at = 0usize;
    for piece in pieces.iter().filter(|piece| !piece.is_empty()) {
        let mut view = records.slice_mut(at..at + piece.len());
        stream.memcpy_htod(*piece, &mut view)?;
        at += piece.len();
    }
    Ok(records)
}

/// Test-only fault: with the `test-faults` feature, a table filled from its
/// records is filled this many rows late (plus
/// `LAMBDA_VM_TEST_FAULT_CARD_FILL_SKIP`, read at the first fill): other words
/// than the host fill's, which the card tests and a proof's verifier must
/// catch.
#[cfg(feature = "test-faults")]
pub static FAULT_RECORDS_SKIP: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "test-faults")]
fn fault_records_skip() -> usize {
    static FROM_ENV: OnceLock<usize> = OnceLock::new();
    let from_env = *FROM_ENV.get_or_init(|| {
        std::env::var("LAMBDA_VM_TEST_FAULT_CARD_FILL_SKIP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    });
    from_env + FAULT_RECORDS_SKIP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Test-only fault: with the `test-faults` feature, a CPU table filled from
/// its records has two of its columns swapped, as a kernel that wrote each
/// into the other's place would: other words than the host fill's, which the
/// card tests and a proof's verifier must catch. The pair is
/// `LAMBDA_VM_TEST_FAULT_CARD_FILL_SWAP=<a>,<b>` (read at the first fill), or
/// else what this holds (`1 + a · CPU_COLUMNS + b`; 0 for none).
#[cfg(feature = "test-faults")]
pub static FAULT_CPU_SWAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "test-faults")]
fn fault_cpu_swap() -> Option<(usize, usize)> {
    static FROM_ENV: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    let from_env = *FROM_ENV.get_or_init(|| {
        let value = std::env::var("LAMBDA_VM_TEST_FAULT_CARD_FILL_SWAP").ok()?;
        let (a, b) = value.trim().split_once(',')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    });
    let held = FAULT_CPU_SWAP.load(std::sync::atomic::Ordering::Relaxed);
    from_env
        .or_else(|| {
            held.checked_sub(1)
                .map(|k| (k / CPU_COLUMNS, k % CPU_COLUMNS))
        })
        .filter(|&(a, b)| a != b && a < CPU_COLUMNS && b < CPU_COLUMNS)
}

/// Columns `a` and `b` of the `rows`-row table in `dst` swapped, on `stream`.
#[cfg(feature = "test-faults")]
fn swap_columns(
    stream: &Arc<CudaStream>,
    dst: &mut CudaViewMut<u64>,
    rows: usize,
    (a, b): (usize, usize),
) -> Result<()> {
    let column_a = stream.clone_dtod(&dst.slice(a * rows..(a + 1) * rows))?;
    let column_b = stream.clone_dtod(&dst.slice(b * rows..(b + 1) * rows))?;
    stream.memcpy_dtod(&column_b, &mut dst.slice_mut(a * rows..(a + 1) * rows))?;
    stream.memcpy_dtod(&column_a, &mut dst.slice_mut(b * rows..(b + 1) * rows))?;
    Ok(())
}

/// `records`' table into `dst` (`rows × cols` words, column-major), all on
/// `stream`: the records copied up as the widen path copies packed bytes
/// (piece by piece into one buffer) and the family's kernel launched, the copy
/// freed in stream order, so a reader on the same stream needs no further
/// ordering. What the kernel checks is counted into `checks`, read once the
/// stream's work is done ([`FillChecks::passed`]).
pub(crate) fn fill_col_major_into(
    stream: &Arc<CudaStream>,
    records: RecordsInput<'_>,
    dst: &mut CudaViewMut<u64>,
    checks: &mut FillChecks,
) -> Result<()> {
    ensure_eq!(
        dst.len(),
        records.rows() * records.cols(),
        "the table's words"
    );
    ensure!(records.len() <= records.rows() || matches!(records, RecordsInput::KeccakRnd { .. }));
    if records.rows() == 0 {
        return Ok(());
    }
    let m = module()?;
    match records {
        RecordsInput::KeccakRnd { ops, skip, rows } => {
            #[cfg(feature = "test-faults")]
            let skip = skip + fault_records_skip();
            let nops = u32::try_from(ops.len()).map_err(|_| crate::invalid_value())?;
            let ops_dev = upload(stream, ops)?;
            launch_keccak_rnd(stream, &m.keccak_rnd_in, &ops_dev, nops, skip, rows, dst)
        }
        RecordsInput::Cpu { logs, decode, rows } => {
            let n = records.len() as u64;
            let logs_dev = upload_pieces(stream, logs)?;
            let height = rows as u64;
            let missing = checks.missing(stream)?;
            unsafe {
                stream
                    .launch_builder(&m.cpu)
                    .arg(&logs_dev)
                    .arg(&n)
                    .arg(&decode.entries)
                    .arg(&decode.runs)
                    .arg(&decode.nruns)
                    .arg(&height)
                    .arg(&mut *dst)
                    .arg(missing)
                    .launch(LaunchConfig {
                        grid_dim: (row_grid(rows, CPU_BLOCK)?, 1, 1),
                        block_dim: (CPU_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            #[cfg(feature = "test-faults")]
            if let Some(pair) = fault_cpu_swap() {
                swap_columns(stream, dst, rows, pair)?;
            }
            Ok(())
        }
        RecordsInput::MemwR {
            records: pieces,
            rows,
        } => {
            let records_dev = upload_pieces(stream, pieces)?;
            launch_rows(stream, &m.memw_r, &records_dev, records.len(), rows, dst)
        }
        RecordsInput::MemwA {
            records: pieces,
            rows,
        } => {
            let records_dev = upload_pieces(stream, pieces)?;
            launch_rows(stream, &m.memw_a, &records_dev, records.len(), rows, dst)
        }
        RecordsInput::Load {
            records: pieces,
            rows,
        } => {
            let records_dev = upload_pieces(stream, pieces)?;
            launch_rows(stream, &m.load, &records_dev, records.len(), rows, dst)
        }
        RecordsInput::Store {
            records: pieces,
            rows,
        } => {
            let records_dev = upload_pieces(stream, pieces)?;
            launch_rows(stream, &m.store, &records_dev, records.len(), rows, dst)
        }
        RecordsInput::Shift {
            records: pieces,
            rows,
        } => {
            let records_dev = upload_pieces(stream, pieces)?;
            launch_rows(stream, &m.shift, &records_dev, records.len(), rows, dst)
        }
        RecordsInput::Memw {
            records: pieces,
            rows,
        } => {
            let records_dev = upload_pieces(stream, pieces)?;
            launch_rows(stream, &m.memw, &records_dev, records.len(), rows, dst)
        }
    }
}

/// A one-record-a-row family's fill over the `n` records on the card into
/// `dst`'s `rows` rows (zero past the records), on `stream`.
fn launch_rows<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    records: &CudaSlice<T>,
    n: usize,
    rows: usize,
    dst: &mut CudaViewMut<u64>,
) -> Result<()> {
    let (n, height) = (n as u64, rows as u64);
    unsafe {
        stream
            .launch_builder(function)
            .arg(records)
            .arg(&n)
            .arg(&height)
            .arg(&mut *dst)
            .launch(LaunchConfig {
                grid_dim: (row_grid(rows, ROW_BLOCK)?, 1, 1),
                block_dim: (ROW_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
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

/// The LFM width-16 Poseidon1 socket's row-major trace (`rows × width` words)
/// on the card, from `narrow`: each row's preprocessed prefix (`prep_w` words)
/// and its sixteen input lanes, `rows × (prep_w + 16)` words row-major. The
/// kernel (`fill_lfm_p1w16_socket`) copies those and writes the round witness
/// from `rounds_off` for the first `real_rows` rows, zeros past them — word for
/// word the host's `p1w16_socket::fill_row` trace. The narrow words cross the
/// bus through the staging pairs, as the R1 commit's own upload would; the
/// stream is synchronized before return, so the buffer is ready for any stream.
pub fn lfm_p1w16_socket_rows(
    narrow: &[u64],
    rows: usize,
    real_rows: usize,
    prep_w: usize,
    width: usize,
    rounds_off: usize,
) -> Result<CudaSlice<u64>> {
    const LANES: usize = 16;
    ensure!(rows > 0 && real_rows <= rows, "real rows within the table");
    ensure!(
        rounds_off >= prep_w + LANES && rounds_off <= width,
        "the witness after the inputs"
    );
    ensure_eq!(narrow.len(), rows * (prep_w + LANES), "the narrow words");
    let m = module()?;
    let stream = backend()?.next_stream();
    let mut narrow_dev = unsafe { stream.alloc::<u64>(narrow.len()) }?;
    crate::device::htod_staged(&stream, narrow, &mut narrow_dev.slice_mut(..))?;
    let mut out = unsafe { stream.alloc::<u64>(rows * width) }?;
    let (rows_u64, real_u64, prep_u64, width_u64, off_u64) = (
        rows as u64,
        real_rows as u64,
        prep_w as u64,
        width as u64,
        rounds_off as u64,
    );
    unsafe {
        stream
            .launch_builder(&m.lfm_p1w16_socket)
            .arg(&narrow_dev)
            .arg(&rows_u64)
            .arg(&real_u64)
            .arg(&prep_u64)
            .arg(&width_u64)
            .arg(&off_u64)
            .arg(&mut out)
            .launch(LaunchConfig {
                grid_dim: (row_grid(rows, ROW_BLOCK)?, 1, 1),
                block_dim: (ROW_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    stream.synchronize()?;
    drop(narrow_dev);
    Ok(out)
}
