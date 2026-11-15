//! D-ARGUE stage 1 on the card: a table's zerocheck rounds the fused way
//! (`thoughts/zf/gap2/fix2/D-ARGUE.md` §2.3–2.6, §4.2), over the factors the
//! GKR input layer already lifted.
//!
//! The kernels are in `kernels/sumcheck.cu` (`zc_*`); the host that drives them
//! — the program with its `ACC` steps, the bus column's coefficients, the
//! round messages out of the sums — is `multilinear::gpu_fused`, and the host
//! reference both are checked against is `multilinear::fused`.
//!
//! The lifted factors are only read: rounds 0 and 1 read their base limbs, and
//! both folds write a quarter-size copy the later rounds fold in place. So
//! today's rounds can still run over them afterwards, which is what the
//! cross-check does.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::Result;
use crate::device::{DeviceReservation, alloc_or_trim, alloc_zeros_or_trim, backend, htod_or_trim};
use crate::sumcheck::{FactorView, eq_table_ext3, launch_shape};

const BLOCK_DIM: u32 = 256;
const MAX_GRID: u32 = 4096;

/// How the session makes the bus column `L`, read once.
pub mod bus_rows {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU8, Ordering};

    static FORCED: AtomicU8 = AtomicU8::new(0);
    static FAULT: AtomicU8 = AtomicU8::new(0);

    /// Whether rounds 0 and 1's pass keeps each row's bus value
    /// `a₀ + Σ a_k·f_k` and `L` is those values folded by the two challenges.
    /// The weights sum to one, so these are the values `zc_bus_column` makes
    /// from the folded factors, without its product a term a quarter-row. On by
    /// default; `LAMBDA_VM_ZC_BUS_FROM_ROWS=0` makes `L` from the folded factors
    /// as before. The same messages either way.
    pub fn enabled() -> bool {
        match FORCED.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("LAMBDA_VM_ZC_BUS_FROM_ROWS").map_or(true, |v| v != "0" && v != "false")
        })
    }

    /// Forces [`enabled`] for sessions made after it; `None` back to the
    /// environment's word.
    #[doc(hidden)]
    pub fn force(on: Option<bool>) {
        FORCED.store(
            match on {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            },
            Ordering::Relaxed,
        );
    }

    /// ⛔ A FAULT, a negative control's only: the rows' values folded by the
    /// first challenge alone, so `L` is wrong.
    #[doc(hidden)]
    pub fn force_fault(on: bool) {
        FAULT.store(u8::from(on), Ordering::Relaxed);
    }

    pub(super) fn fault() -> bool {
        FAULT.load(Ordering::Relaxed) == 1
    }
}

/// Whether the factors no round reads are bound once at the hand-off, read once.
pub mod unread {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU8, Ordering};

    static FORCED: AtomicU8 = AtomicU8::new(0);
    static FAULT: AtomicU8 = AtomicU8::new(0);

    /// Whether a session whose bus column comes from the rows
    /// ([`super::bus_rows`]) leaves the factors no round reads (the bus's
    /// alone) out of the double fold and the folds. Their cells for the host
    /// tail are then made once from the base columns with every bound
    /// challenge at a time, on the card, at the hand-off's own read-back. The
    /// same values. On by default; `LAMBDA_VM_ZC_UNREAD_AT_END=0` folds them as
    /// before.
    pub fn enabled() -> bool {
        match FORCED.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("LAMBDA_VM_ZC_UNREAD_AT_END").map_or(true, |v| v != "0" && v != "false")
        })
    }

    /// Forces [`enabled`] for sessions made after it; `None` back to the
    /// environment's word.
    #[doc(hidden)]
    pub fn force(on: Option<bool>) {
        FORCED.store(
            match on {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            },
            Ordering::Relaxed,
        );
    }

    /// ⛔ A FAULT, a negative control's only: the unread factors' cells are not
    /// written, so the host tail reads what the gather left there.
    #[doc(hidden)]
    pub fn force_fault(on: bool) {
        FAULT.store(u8::from(on), Ordering::Relaxed);
    }

    pub(super) fn fault() -> bool {
        FAULT.load(Ordering::Relaxed) == 1
    }
}

/// [`unread`]'s split of a table's factors: those a round reads (their slots,
/// how many, and the fold's pointer table over their slabs and `L`) and those
/// no round reads (their slots on the card and here), with the end pass's own
/// partials.
struct Unread {
    reads: CudaSlice<u32>,
    count: u64,
    fold_table: CudaSlice<u64>,
    unread_dev: CudaSlice<u32>,
    unread: Vec<u32>,
    partials: CudaSlice<u64>,
}

/// Threads the end pass sums each unread factor over: a multiple of every cube
/// the host tail takes (at most this many cells).
const UNREAD_THREADS: u64 = 1024;

/// Scratch ceiling for the per-thread slot files, as the round kernel's.
const SLOT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const MAX_THREADS: u64 = 1 << 20;

/// The constraint part, lowered with its `ACC` steps: `nodes` two u64 a step;
/// its constants once as base values (the grid pass) and once as ext3 (the
/// later rounds); `betas` three u64 a root; `num_slots` the live set.
#[derive(Clone, Copy, Debug)]
pub struct FusedProgram<'a> {
    pub nodes: &'a [u64],
    pub base_consts: &'a [u64],
    pub ext_consts: &'a [u64],
    pub betas: &'a [u64],
    pub num_slots: usize,
}

/// The bus column `L = a₀ + Σ a_k·f_k`: each term's factor slot, its ext3
/// coefficient (three u64), and `a₀`.
#[derive(Clone, Copy, Debug)]
pub struct FusedBus<'a> {
    pub slots: &'a [u32],
    pub coeffs: &'a [u64],
    pub constant: [u64; 3],
}

/// One table's fused zerocheck on the card.
pub struct FusedZerocheck<'f> {
    stream: Arc<CudaStream>,
    source: FactorView<'f>,
    width: usize,
    /// The cube the lifted factors span.
    rows: usize,
    /// Cube indices left in the folded buffer (0 before `fold2`).
    len: usize,
    nodes: CudaSlice<u64>,
    num_nodes: u64,
    base_consts: CudaSlice<u64>,
    ext_consts: CudaSlice<u64>,
    betas: CudaSlice<u64>,
    num_slots: usize,
    term_slots: CudaSlice<u32>,
    term_coeffs: CudaSlice<u64>,
    num_terms: u64,
    constant: CudaSlice<u64>,
    eq_r: CudaSlice<u64>,
    eq_rho: CudaSlice<u64>,
    /// `width` factor slabs and then `L`'s, a quarter of the cube each.
    folded: CudaSlice<u64>,
    folded_ptrs: CudaSlice<u64>,
    /// The fold's pointer table — every folded factor, then `L` — built once:
    /// the addresses never move, and a table built per fold would have to be
    /// waited for before it could be dropped.
    fold_ptrs: CudaSlice<u64>,
    slots: CudaSlice<u64>,
    partials: CudaSlice<u64>,
    sums: CudaSlice<u64>,
    points: CudaSlice<u64>,
    node_ints: CudaSlice<u32>,
    scratch: CudaSlice<u64>,
    violation: CudaSlice<u64>,
    /// [`bus_rows`]: every row's bus value (`4q` ext3), its one-entry pointer
    /// table for the folds, and room for the two challenges' limbs.
    rows_l: Option<(CudaSlice<u64>, CudaSlice<u64>, CudaSlice<u64>)>,
    /// [`unread`]: the factors a round reads and those it does not.
    needed: Option<Unread>,
    _room: DeviceReservation,
}

/// Threads a slot file of `bytes_per_thread` affords under the budget.
fn ceiling(bytes_per_thread: u64) -> u64 {
    (SLOT_BUDGET_BYTES / bytes_per_thread.max(1)).clamp(32, MAX_THREADS)
}

/// Threads a fused launch takes in all: past this the card is full (170 SMs of
/// a few blocks each) and a wider launch only grows the slot file, which is
/// VRAM beside the argue's and DRAM traffic.
const MAX_LAUNCH_THREADS: u64 = 1 << 17;

/// Grid points and nodes a launch may own: one block row each.
pub const MAX_ROWS: usize = 32;

/// Threads a launch of `work` indices over `rows` block rows takes, a slot
/// file of `words` u64 a thread, at most `cap_threads` in all: `(grid.x, block)`.
fn fit(work: u64, rows: usize, words: u64, cap_threads: u64) -> Option<(u32, u32)> {
    if rows == 0 {
        return None;
    }
    let per_row = (ceiling(words * 8) / rows as u64)
        .min(cap_threads / rows as u64)
        .min(MAX_LAUNCH_THREADS / rows as u64);
    if per_row < 32 {
        return None;
    }
    let (mut grid, block) = launch_shape(per_row, work);
    grid = grid.min(MAX_GRID);
    while grid > 1 && grid as u64 * block as u64 > per_row {
        grid -= 1;
    }
    Some((grid, block))
}

/// The slot file's u64s: the grid pass's first launch (base slots) or the
/// first later round's (ext3), whichever is wider.
fn slot_words(rows: usize, num_slots: usize, grid_rows: usize, gruen_rows: usize) -> usize {
    let s = num_slots.max(1) as u64;
    let q = (rows / 4) as u64;
    let words = |work: u64, rows: usize, per: u64| -> u64 {
        fit(work, rows, per, u64::MAX).map_or(0, |(g, b)| g as u64 * b as u64 * rows as u64 * per)
    };
    words(q, grid_rows, s).max(words((q / 2).max(1), gruen_rows, 3 * s)) as usize
}

impl<'f> FusedZerocheck<'f> {
    /// Bytes the session holds beside the lifted factors: the folded copy and
    /// `L` at a quarter, the two weights at a quarter, the slot file its first
    /// launches need, and the partials.
    pub fn device_bytes(
        width: usize,
        rows: usize,
        num_slots: usize,
        grid_rows: usize,
        gruen_rows: usize,
    ) -> u64 {
        let q = (rows / 4) as u64;
        let folded = (width as u64 + 1) * q * 24;
        let weights = 2 * q * 24;
        let slots = slot_words(rows, num_slots, grid_rows, gruen_rows) as u64 * 8;
        let partials = (MAX_ROWS as u64) * (MAX_GRID as u64) * 24;
        folded + weights + slots + partials + (1 << 20)
    }

    /// Sets the session up, or `Ok(None)` when the card's budget refuses it —
    /// before anything is absorbed, so the caller runs today's rounds.
    ///
    /// `r_tail` and `rho_tail` are `r[2..]` and `ρ[2..]` (three u64 a
    /// coordinate): the grid pass's weights.
    pub fn new(
        source: FactorView<'f>,
        program: FusedProgram<'_>,
        bus: FusedBus<'_>,
        r_tail: &[u64],
        rho_tail: &[u64],
        grid_rows: usize,
        gruen_rows: usize,
    ) -> Result<Option<Self>> {
        let rows = source.rows;
        let width = source.width;
        assert!(
            rows >= 8 && rows.is_power_of_two(),
            "the grid needs four rows a group"
        );
        assert_eq!(
            1usize << (r_tail.len() / 3),
            rows / 4,
            "r[2..] spans a quarter"
        );
        assert_eq!(r_tail.len(), rho_tail.len(), "ρ[2..] spans a quarter");
        assert_eq!(
            bus.slots.len() * 3,
            bus.coeffs.len(),
            "three u64 a coefficient"
        );
        let be = backend()?;
        assert!(
            grid_rows <= MAX_ROWS && gruen_rows <= MAX_ROWS,
            "rows per launch"
        );
        let words = slot_words(rows, program.num_slots, grid_rows, gruen_rows);
        if words == 0 {
            return Ok(None);
        }
        let from_rows = bus_rows::enabled();
        // The unread factors' end pass needs the rows' bus column: no round
        // reads a factor to make `L` then.
        let at_end = from_rows && unread::enabled();
        let extra = if from_rows { rows as u64 * 24 } else { 0 }
            + if at_end {
                width as u64 * UNREAD_THREADS * 24
            } else {
                0
            };
        let Some(room) = be.reserve(
            Self::device_bytes(width, rows, program.num_slots, grid_rows, gruen_rows) + extra,
        ) else {
            return Ok(None);
        };
        let stream = source.stream.clone();
        let q = rows / 4;
        let nonempty = |v: &[u64]| -> Vec<u64> { if v.is_empty() { vec![0] } else { v.to_vec() } };
        let nodes = htod_or_trim(&stream, &nonempty(program.nodes))?;
        let base_consts = htod_or_trim(&stream, &nonempty(program.base_consts))?;
        let ext_consts = htod_or_trim(&stream, &nonempty(program.ext_consts))?;
        let betas = htod_or_trim(&stream, &nonempty(program.betas))?;
        let term_slots = htod_or_trim(
            &stream,
            if bus.slots.is_empty() {
                &[0u32][..]
            } else {
                bus.slots
            },
        )?;
        let term_coeffs = htod_or_trim(&stream, &nonempty(bus.coeffs))?;
        let constant = htod_or_trim(&stream, &bus.constant[..])?;
        let eq_r = eq_table_ext3(&stream, r_tail, q)?;
        let eq_rho = eq_table_ext3(&stream, rho_tail, q)?;
        // SAFETY: every slab is written by `fold2` and `zc_bus_column` before any
        // round reads it.
        let folded = unsafe { alloc_or_trim::<u64>(&stream, (width + 1) * q * 3) }?;
        let addresses: Vec<u64> = {
            let (base, _record) = folded.device_ptr(&stream);
            (0..=width).map(|k| base + (k * q * 24) as u64).collect()
        };
        let folded_ptrs = htod_or_trim(&stream, &addresses[..width])?;
        let fold_ptrs = htod_or_trim(&stream, &addresses)?;
        // SAFETY: a thread writes a slot before it reads it.
        let slots = unsafe { alloc_or_trim::<u64>(&stream, words) }?;
        let partials = unsafe { alloc_or_trim::<u64>(&stream, MAX_ROWS * MAX_GRID as usize * 3) }?;
        let sums = alloc_zeros_or_trim::<u64>(&stream, MAX_ROWS * 3)?;
        let points = alloc_zeros_or_trim::<u64>(&stream, MAX_ROWS)?;
        let node_ints = alloc_zeros_or_trim::<u32>(&stream, MAX_ROWS)?;
        let scratch = alloc_zeros_or_trim::<u64>(&stream, 12)?;
        let violation = alloc_zeros_or_trim::<u64>(&stream, 1)?;
        let needed = if at_end {
            // The slots a walk reads: its `VAR` steps' factors and its selected
            // `ACC` steps' selectors.
            let mut read = vec![false; width];
            for node in program.nodes.chunks_exact(2) {
                let op = node[0] as u32;
                let a = (node[0] >> 32) as usize;
                let b = node[1] as u32;
                if op == 1 && a < width {
                    read[a] = true;
                }
                if op == 6 && b != u32::MAX && (b as usize) < width {
                    read[b as usize] = true;
                }
            }
            let reads: Vec<u32> = (0..width as u32).filter(|&k| read[k as usize]).collect();
            let unread: Vec<u32> = (0..width as u32).filter(|&k| !read[k as usize]).collect();
            let mut fold_table: Vec<u64> = reads.iter().map(|&k| addresses[k as usize]).collect();
            fold_table.push(addresses[width]);
            let count = reads.len() as u64;
            let one_or =
                |v: &[u32]| -> Vec<u32> { if v.is_empty() { vec![0] } else { v.to_vec() } };
            let reads_dev = htod_or_trim(&stream, &one_or(&reads))?;
            let unread_dev = htod_or_trim(&stream, &one_or(&unread))?;
            let fold_dev = htod_or_trim(&stream, &fold_table)?;
            // SAFETY: the end pass writes every partial before it reads one.
            let partials = unsafe {
                alloc_or_trim::<u64>(&stream, unread.len().max(1) * UNREAD_THREADS as usize * 3)
            }?;
            Some(Unread {
                reads: reads_dev,
                count,
                fold_table: fold_dev,
                unread_dev,
                unread,
                partials,
            })
        } else {
            None
        };
        let rows_l = if from_rows {
            // SAFETY: `zc_bus_u_store` writes every row before the folds read it.
            let values = unsafe { alloc_or_trim::<u64>(&stream, rows * 3) }?;
            let address = {
                let (at, _record) = values.device_ptr(&stream);
                at
            };
            let table = htod_or_trim(&stream, &[address])?;
            let challenges = alloc_zeros_or_trim::<u64>(&stream, 6)?;
            Some((values, table, challenges))
        } else {
            None
        };
        Ok(Some(Self {
            stream,
            source,
            width,
            rows,
            len: 0,
            nodes,
            num_nodes: (program.nodes.len() / 2) as u64,
            base_consts,
            ext_consts,
            betas,
            num_slots: program.num_slots.max(1),
            term_slots,
            term_coeffs,
            num_terms: bus.slots.len() as u64,
            constant,
            eq_r,
            eq_rho,
            folded,
            folded_ptrs,
            fold_ptrs,
            slots,
            partials,
            sums,
            points,
            node_ints,
            scratch,
            violation,
            rows_l,
            needed,
            _room: room,
        }))
    }

    /// A launch over `work` indices with `rows` block rows and `words` slot
    /// u64s a thread, inside the slot file this session holds.
    fn shape(&self, work: u64, rows: usize, words: u64) -> Result<(u32, u32)> {
        fit(work, rows, words, self.slots.len() as u64 / words.max(1)).ok_or(
            cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY),
        )
    }

    /// Sums `rows × grid` partials into `rows` ext3 values and reads them back.
    fn reduce(&mut self, rows: usize, grid: u32) -> Result<Vec<u64>> {
        self.reduce_enqueue(rows, grid)?;
        self.collect(rows)
    }

    /// [`reduce`](Self::reduce)'s launch, not waited for.
    fn reduce_enqueue(&mut self, rows: usize, grid: u32) -> Result<()> {
        let be = backend()?;
        crate::round_slot::note_second_launch();
        let reduce_block = 256u32;
        unsafe {
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid as u64))
                .arg(&mut self.sums)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (reduce_block, 1, 1),
                    shared_mem_bytes: reduce_block * 3 * 8,
                })?;
        }
        Ok(())
    }

    /// The `rows` sums a queued launch left, read back once the stream is idle.
    pub fn collect(&self, rows: usize) -> Result<Vec<u64>> {
        let out = crate::readback::clone_dtoh(&self.stream, &self.sums.slice(0..rows * 3))?;
        self.stream.synchronize()?;
        Ok(out)
    }

    /// Rounds 0 and 1's pass (S1-5): `T` at each grid point `(a, b)` of
    /// `points` (three u64 each, in order), `U(b, c)` at `2b + c`, and the
    /// first row whose corner is not zero, if a corner was on the list and one
    /// was not. With `keep_corners` a corner is summed into `T` like any
    /// point and checked for nothing — random columns, in a test. An empty
    /// list — a table without roots, whose grid is all corners — launches no
    /// grid pass: `T` is zero.
    #[allow(clippy::type_complexity)]
    pub fn grid(
        &mut self,
        points: &[(u32, u32)],
        keep_corners: bool,
    ) -> Result<(Vec<u64>, Vec<u64>, Option<u64>)> {
        assert!(points.len() <= MAX_ROWS, "grid points per launch");
        let (t, violation) = if points.is_empty() {
            (Vec::new(), None)
        } else {
            self.grid_pass(points, keep_corners)?
        };
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        let (grid, block) = (
            q.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32,
            BLOCK_DIM,
        );
        let factors = self.source.ptrs;
        let config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 3 * 8,
        };
        unsafe {
            match self.rows_l.as_mut() {
                // Each row's bus value kept as well: `L`'s source ([`bus_rows`]).
                Some((values, _, _)) => {
                    self.stream
                        .launch_builder(&be.zc_bus_u_store)
                        .arg(factors)
                        .arg(&q)
                        .arg(&self.term_slots)
                        .arg(&self.term_coeffs)
                        .arg(&self.num_terms)
                        .arg(&self.constant)
                        .arg(&self.eq_rho)
                        .arg(&mut self.partials)
                        .arg(&self.source.stride)
                        .arg(values)
                        .launch(config)?;
                }
                None => {
                    self.stream
                        .launch_builder(&be.zc_bus_u)
                        .arg(factors)
                        .arg(&q)
                        .arg(&self.term_slots)
                        .arg(&self.term_coeffs)
                        .arg(&self.num_terms)
                        .arg(&self.constant)
                        .arg(&self.eq_rho)
                        .arg(&mut self.partials)
                        .arg(&self.source.stride)
                        .launch(config)?;
                }
            }
        }
        let u = self.reduce(4, grid)?;
        Ok((t, u, violation))
    }

    /// `zc_grid01` over a non-empty `points`: `T` at each, three u64 a point,
    /// and the first row whose corner is not zero.
    fn grid_pass(
        &mut self,
        points: &[(u32, u32)],
        keep_corners: bool,
    ) -> Result<(Vec<u64>, Option<u64>)> {
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        let packed: Vec<u64> = points
            .iter()
            .map(|&(a, b)| u64::from(a) | (u64::from(b) << 32))
            .collect();
        {
            let mut head = self.points.slice_mut(0..packed.len());
            self.stream.memcpy_htod(&packed, &mut head)?;
        }
        self.stream.memcpy_htod(&[u64::MAX], &mut self.violation)?;
        let (grid, block) = self.shape(q, points.len(), self.num_slots as u64)?;
        let factors = self.source.ptrs;
        unsafe {
            self.stream
                .launch_builder(&be.zc_grid01)
                .arg(factors)
                .arg(&q)
                .arg(&self.nodes)
                .arg(&self.num_nodes)
                .arg(&self.base_consts)
                .arg(&self.betas)
                .arg(&self.points)
                .arg(&self.eq_r)
                .arg(&mut self.slots)
                .arg(&mut self.partials)
                .arg(&mut self.violation)
                .arg(&u32::from(keep_corners))
                .arg(&self.source.stride)
                .launch(LaunchConfig {
                    grid_dim: (grid, points.len() as u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: block * 3 * 8,
                })?;
        }
        let t = self.reduce(points.len(), grid)?;
        let violation = crate::readback::clone_dtoh(&self.stream, &self.violation)?;
        self.stream.synchronize()?;
        Ok((t, (violation[0] != u64::MAX).then_some(violation[0])))
    }

    /// Both folds at once (`w[2a + b] = eq₁(s₀, a)·eq₁(s₁, b)`, twelve u64), into
    /// the quarter-size copy; then `L` there, and the weights halved to round
    /// 2's `eq(·_{>2})`. `challenges` are `s₀` then `s₁` (three u64 each), which
    /// the rows' bus values are folded by ([`bus_rows`]).
    pub fn fold2(&mut self, w: &[u64; 12], challenges: &[u64; 6]) -> Result<()> {
        assert_eq!(self.len, 0, "the double fold is the first");
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        self.stream.memcpy_htod(&w[..], &mut self.scratch)?;
        let factors = self.source.ptrs;
        // With [`unread`], only the factors a round reads.
        let selected = self.needed.as_ref().map(|u| (&u.reads, u.count));
        let width = selected.map_or(self.width as u64, |(_, count)| count);
        let total = width * q;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        if total > 0 {
            unsafe {
                match selected {
                    Some((reads, count)) => {
                        self.stream
                            .launch_builder(&be.zc_fold2_sel)
                            .arg(factors)
                            .arg(&q)
                            .arg(reads)
                            .arg(&count)
                            .arg(&self.scratch)
                            .arg(&mut self.folded)
                            .arg(&self.source.stride)
                            .launch(config)?;
                    }
                    None => {
                        self.stream
                            .launch_builder(&be.zc_fold2)
                            .arg(factors)
                            .arg(&q)
                            .arg(&width)
                            .arg(&self.scratch)
                            .arg(&mut self.folded)
                            .arg(&self.source.stride)
                            .launch(config)?;
                    }
                }
            }
        }
        self.len = q as usize;
        if self.rows_l.is_some() {
            self.bus_from_rows(challenges)?;
        } else {
            self.bus_column()?;
        }
        self.halve_weights()
    }

    /// `L` from the rows' bus values: folded by `s₀` (the high variable) and
    /// then `s₁` in place, and the first quarter copied into `L`'s slab.
    fn bus_from_rows(&mut self, challenges: &[u64; 6]) -> Result<()> {
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        let Some((values, table, limbs)) = self.rows_l.as_mut() else {
            return Ok(());
        };
        self.stream.memcpy_htod(&challenges[..], limbs)?;
        let folds: &[(usize, u64)] = if bus_rows::fault() {
            &[(0, 2 * q)]
        } else {
            &[(0, 2 * q), (1, q)]
        };
        for &(round, half) in folds {
            let one = 1u64;
            let grid = half.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
            let s = limbs.slice(round * 3..round * 3 + 3);
            unsafe {
                self.stream
                    .launch_builder(&be.sumcheck_fold_ext3)
                    .arg(&mut *table)
                    .arg(&half)
                    .arg(&one)
                    .arg(&s)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        let width = self.width;
        let qs = q as usize;
        let mut l = self.folded.slice_mut(width * qs * 3..(width + 1) * qs * 3);
        self.stream.memcpy_dtod(&values.slice(0..qs * 3), &mut l)?;
        Ok(())
    }

    /// `L` over the folded factors, into its slab.
    fn bus_column(&mut self) -> Result<()> {
        let be = backend()?;
        let rows = self.len as u64;
        let grid = rows.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let q = self.rows / 4;
        let mut l = self
            .folded
            .slice_mut(self.width * q * 3..(self.width + 1) * q * 3);
        unsafe {
            self.stream
                .launch_builder(&be.zc_bus_column)
                .arg(&self.folded_ptrs)
                .arg(&rows)
                .arg(&self.term_slots)
                .arg(&self.term_coeffs)
                .arg(&self.num_terms)
                .arg(&self.constant)
                .arg(&mut l)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Both weights to the next round's `eq(·_{>j+1})`: their halves added. The
    /// weights are always half the folded cube after this.
    pub fn halve_weights(&mut self) -> Result<()> {
        let be = backend()?;
        let half = (self.len / 2) as u64;
        let grid = half.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        for table in [&mut self.eq_r, &mut self.eq_rho] {
            unsafe {
                self.stream
                    .launch_builder(&be.zc_halve)
                    .arg(table)
                    .arg(&half)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        Ok(())
    }

    /// A later round (S1-4): `A` at each integer node of `nodes`, then `B(0)`
    /// and `B(1)`, three u64 each.
    pub fn round(&mut self, nodes: &[u32]) -> Result<Vec<u64>> {
        let rows = self.round_enqueue(nodes)?;
        self.collect(rows)
    }

    /// [`round`](Self::round)'s launches, queued and not waited for; the
    /// caller reads the sums with [`collect`](Self::collect) and the returned
    /// row count. The batched argue queues every table's round before it
    /// reads any back.
    pub fn round_enqueue(&mut self, nodes: &[u32]) -> Result<usize> {
        assert!(self.len >= 2, "a round needs a variable to bind");
        let rows = nodes.len() + 2;
        assert!(rows <= MAX_ROWS, "nodes per launch");
        let be = backend()?;
        let half = (self.len / 2) as u64;
        {
            let mut head = self.node_ints.slice_mut(0..nodes.len().max(1));
            let ints: Vec<u32> = if nodes.is_empty() {
                vec![0]
            } else {
                nodes.to_vec()
            };
            self.stream.memcpy_htod(&ints, &mut head)?;
        }
        let (grid, block) = self.shape(half, rows, 3 * self.num_slots as u64)?;
        let num_a = nodes.len() as u32;
        let q = self.rows / 4;
        let l = self
            .folded
            .slice(self.width * q * 3..(self.width + 1) * q * 3);
        unsafe {
            self.stream
                .launch_builder(&be.zc_round_gruen)
                .arg(&self.folded_ptrs)
                .arg(&half)
                .arg(&self.nodes)
                .arg(&self.num_nodes)
                .arg(&self.ext_consts)
                .arg(&self.betas)
                .arg(&self.node_ints)
                .arg(&num_a)
                .arg(&self.eq_r)
                .arg(&self.eq_rho)
                .arg(&l)
                .arg(&mut self.slots)
                .arg(&mut self.partials)
                .launch(LaunchConfig {
                    grid_dim: (grid, rows as u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: block * 3 * 8,
                })?;
        }
        self.reduce_enqueue(rows, grid)?;
        Ok(rows)
    }

    /// Binds the round's variable to `s` (three u64) in every folded factor and
    /// in `L`; the weights follow when `more` rounds are left on the card.
    pub fn fold(&mut self, s: &[u64; 3], more: bool) -> Result<()> {
        assert!(self.len >= 2, "a fold needs a variable to bind");
        let be = backend()?;
        let half = (self.len / 2) as u64;
        self.stream
            .memcpy_htod(&s[..], &mut self.scratch.slice_mut(0..3))?;
        // The factors and `L`, through the one pointer table of `width + 1` —
        // or, with [`unread`], of the factors a round reads and `L`.
        let (table, width) = match self.needed.as_mut() {
            Some(u) => (&mut u.fold_table, u.count + 1),
            None => (&mut self.fold_ptrs, self.width as u64 + 1),
        };
        let total = width * half;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_fold_ext3)
                .arg(table)
                .arg(&half)
                .arg(&width)
                .arg(&self.scratch)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.len /= 2;
        if more {
            self.halve_weights()?;
        }
        Ok(())
    }

    /// Cube indices left in the folded copy.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// What every factor has left — `width` of them, three u64 an element —
    /// gathered on the card and read back in one copy.
    /// Whether this session leaves the factors no round reads unfolded
    /// ([`unread`], read when it was made), so
    /// [`values_bound`](Self::values_bound) owes their cells.
    pub fn skips_unread(&self) -> bool {
        self.needed.as_ref().is_some_and(|u| !u.unread.is_empty())
    }

    /// [`values`](Self::values) for a session that leaves the factors no
    /// round reads unfolded: those factors' cells are made from the base
    /// columns with every bound challenge at once (`bound`, three u64 a
    /// challenge, the first bound first) and written into the same gathered
    /// heads, so one read-back takes every factor. The `eq` table of the bound
    /// challenges is built on the card, and the partials are the session's own.
    pub fn values_bound(&mut self, bound: &[u64]) -> Result<Vec<Vec<u64>>> {
        if !self.skips_unread() {
            return self.values();
        }
        let be = backend()?;
        let rows = self.rows as u64;
        let cells = self.len as u64;
        let spans = rows / cells.max(1);
        if (1u64 << (bound.len() / 3)) != spans || cells > UNREAD_THREADS {
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            ));
        }
        let e = eq_table_ext3(&self.stream, bound, spans as usize)?;
        let span = self.len * 3;
        let total = self.width * span;
        // SAFETY: the gather writes every word, the second pass the unread
        // factors' cells over theirs.
        let mut out = unsafe { alloc_or_trim::<u64>(&self.stream, total) }?;
        let width = self.width as u64;
        let grid = (total as u64)
            .div_ceil(BLOCK_DIM as u64)
            .clamp(1, MAX_GRID as u64) as u32;
        let factors = self.source.ptrs;
        let threads = UNREAD_THREADS;
        let fault = unread::fault();
        let Some(u) = self.needed.as_mut() else {
            return self.values();
        };
        let nsel = u.unread.len();
        unsafe {
            self.stream
                .launch_builder(&be.gather_factor_heads_ext3)
                .arg(&self.folded_ptrs)
                .arg(&width)
                .arg(&cells)
                .arg(&mut out)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            self.stream
                .launch_builder(&be.zc_bind_all)
                .arg(factors)
                .arg(&u.unread_dev)
                .arg(&rows)
                .arg(&cells)
                .arg(&e)
                .arg(&self.source.stride)
                .arg(&mut u.partials)
                .launch(LaunchConfig {
                    grid_dim: ((threads / u64::from(BLOCK_DIM)) as u32, nsel as u32, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            if !fault {
                let nsel_arg = nsel as u64;
                let cells_total = nsel as u64 * cells;
                self.stream
                    .launch_builder(&be.zc_bind_all_sum)
                    .arg(&u.partials)
                    .arg(&nsel_arg)
                    .arg(&cells)
                    .arg(&threads)
                    .arg(&u.unread_dev)
                    .arg(&mut out)
                    .launch(LaunchConfig {
                        grid_dim: (
                            cells_total
                                .div_ceil(u64::from(BLOCK_DIM))
                                .clamp(1, MAX_GRID as u64) as u32,
                            1,
                            1,
                        ),
                        block_dim: (BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        let flat = crate::readback::clone_dtoh(&self.stream, &out)?;
        self.stream.synchronize()?;
        Ok(flat.chunks_exact(span).map(<[u64]>::to_vec).collect())
    }

    pub fn values(&self) -> Result<Vec<Vec<u64>>> {
        let be = backend()?;
        let span = self.len * 3;
        let total = self.width * span;
        // SAFETY: the kernel writes every word of it, one thread per word.
        let mut out = unsafe { alloc_or_trim::<u64>(&self.stream, total) }?;
        let width = self.width as u64;
        let cells = self.len as u64;
        let grid = (total as u64)
            .div_ceil(BLOCK_DIM as u64)
            .clamp(1, MAX_GRID as u64) as u32;
        unsafe {
            self.stream
                .launch_builder(&be.gather_factor_heads_ext3)
                .arg(&self.folded_ptrs)
                .arg(&width)
                .arg(&cells)
                .arg(&mut out)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let flat = crate::readback::clone_dtoh(&self.stream, &out)?;
        self.stream.synchronize()?;
        Ok(flat.chunks_exact(span).map(<[u64]>::to_vec).collect())
    }

    /// `L`'s remaining values (a test's view).
    pub fn bus_values(&self) -> Result<Vec<u64>> {
        let q = self.rows / 4;
        let at = self.width * q * 3;
        let out =
            crate::readback::clone_dtoh(&self.stream, &self.folded.slice(at..at + self.len * 3))?;
        self.stream.synchronize()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table without roots has `d_C = 0`, so its grid `{0,1}²` is all
    /// corners and, with the corners skipped, no point is left to launch —
    /// FAST job 273's B arm, which divided by that empty row count.
    #[test]
    fn a_grid_of_corners_only_sizes_no_grid_launch() {
        let words = slot_words(1 << 20, 1, 0, 3);
        assert!(words > 0, "the later rounds still need their slot file");
        assert_eq!(fit(1 << 18, 0, 1, u64::MAX), None, "no rows, no launch");
    }
}
