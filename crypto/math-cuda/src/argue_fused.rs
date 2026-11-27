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

/// A second bus column (a table's local lookups, argued at their own point):
/// its coefficients and `ρ'[2..]`, the grid pass's weight for it.
#[derive(Clone, Copy, Debug)]
pub struct SecondBus<'a> {
    pub bus: FusedBus<'a>,
    pub rho_tail: &'a [u64],
}

/// The second bus on the card: its terms, its weight, and where its column
/// sits in the folded slab (after `L`'s).
struct Second {
    term_slots: CudaSlice<u32>,
    term_coeffs: CudaSlice<u64>,
    num_terms: u64,
    constant: CudaSlice<u64>,
    eq_rho: CudaSlice<u64>,
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
    /// A second bus column `L'` at its own point, or none: its column is the
    /// slab after `L`'s, folded with the factors; its round sums `B'(0)`,
    /// `B'(1)` follow the first bus's in every round's sums.
    second: Option<Second>,
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
        Self::device_bytes_with(width, rows, num_slots, grid_rows, gruen_rows, false)
    }

    /// [`Self::device_bytes`], with a second bus: its column and its weight,
    /// a quarter each, more.
    pub fn device_bytes_with(
        width: usize,
        rows: usize,
        num_slots: usize,
        grid_rows: usize,
        gruen_rows: usize,
        second: bool,
    ) -> u64 {
        let q = (rows / 4) as u64;
        let buses = 1 + u64::from(second);
        let folded = (width as u64 + buses) * q * 24;
        let weights = (1 + buses) * q * 24;
        let slots = slot_words(rows, num_slots, grid_rows, gruen_rows) as u64 * 8;
        let partials = (MAX_ROWS as u64) * (MAX_GRID as u64) * 24;
        folded + weights + slots + partials + (1 << 20)
    }

    /// Sets the session up, or declines before anything is absorbed. The
    /// caller supplies the reservation so it can retry a busy card before
    /// allocating the session.
    ///
    /// `r_tail` and `rho_tail` are `r[2..]` and `ρ[2..]` (three u64 a
    /// coordinate): the grid pass's weights. `second` is a second bus column
    /// at its own point, or none.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: FactorView<'f>,
        program: FusedProgram<'_>,
        bus: FusedBus<'_>,
        r_tail: &[u64],
        rho_tail: &[u64],
        grid_rows: usize,
        gruen_rows: usize,
        second: Option<SecondBus<'_>>,
        reserve: impl FnOnce(&crate::device::Backend, u64) -> Option<DeviceReservation>,
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
        if let Some(second) = &second {
            assert_eq!(
                second.rho_tail.len(),
                rho_tail.len(),
                "ρ'[2..] spans a quarter"
            );
            assert_eq!(
                second.bus.slots.len() * 3,
                second.bus.coeffs.len(),
                "three u64 a coefficient"
            );
        }
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
        let extra = if from_rows { rows as u64 * 24 } else { 0 };
        let Some(room) = reserve(
            be,
            Self::device_bytes_with(
                width,
                rows,
                program.num_slots,
                grid_rows,
                gruen_rows,
                second.is_some(),
            ) + extra,
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
        let buses = 1 + usize::from(second.is_some());
        let second = match second {
            None => None,
            Some(second) => Some(Second {
                term_slots: htod_or_trim(
                    &stream,
                    if second.bus.slots.is_empty() {
                        &[0u32][..]
                    } else {
                        second.bus.slots
                    },
                )?,
                term_coeffs: htod_or_trim(&stream, &nonempty(second.bus.coeffs))?,
                num_terms: second.bus.slots.len() as u64,
                constant: htod_or_trim(&stream, &second.bus.constant[..])?,
                eq_rho: eq_table_ext3(&stream, second.rho_tail, q)?,
            }),
        };
        // SAFETY: every slab is written by `fold2` and `zc_bus_column` before any
        // round reads it.
        let folded = unsafe { alloc_or_trim::<u64>(&stream, (width + buses) * q * 3) }?;
        let addresses: Vec<u64> = {
            let (base, _record) = folded.device_ptr(&stream);
            (0..width + buses)
                .map(|k| base + (k * q * 24) as u64)
                .collect()
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
            second,
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
        self.reduce_enqueue_at(rows, grid, 0)
    }

    /// [`reduce_enqueue`](Self::reduce_enqueue) into the sums from row `at` on.
    fn reduce_enqueue_at(&mut self, rows: usize, grid: u32, at: usize) -> Result<()> {
        let be = backend()?;
        crate::round_slot::note_second_launch();
        let reduce_block = 256u32;
        let mut out = self.sums.slice_mut(at * 3..(at + rows) * 3);
        unsafe {
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid as u64))
                .arg(&mut out)
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

    /// The second bus's `U'(b, c)` at `2b + c` (three u64 each) over the
    /// lifted factors, at its own weight: `None` without a second bus.
    pub fn second_grid(&mut self) -> Result<Option<Vec<u64>>> {
        let Some(second) = self.second.as_ref() else {
            return Ok(None);
        };
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        let (grid, block) = (
            q.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32,
            BLOCK_DIM,
        );
        unsafe {
            self.stream
                .launch_builder(&be.zc_bus_u)
                .arg(self.source.ptrs)
                .arg(&q)
                .arg(&second.term_slots)
                .arg(&second.term_coeffs)
                .arg(&second.num_terms)
                .arg(&second.constant)
                .arg(&second.eq_rho)
                .arg(&mut self.partials)
                .arg(&self.source.stride)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: block * 3 * 8,
                })?;
        }
        self.reduce(4, grid).map(Some)
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
        let width = self.width as u64;
        let total = width * q;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let factors = self.source.ptrs;
        unsafe {
            self.stream
                .launch_builder(&be.zc_fold2)
                .arg(factors)
                .arg(&q)
                .arg(&width)
                .arg(&self.scratch)
                .arg(&mut self.folded)
                .arg(&self.source.stride)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.len = q as usize;
        if self.rows_l.is_some() {
            self.bus_from_rows(challenges)?;
        } else {
            self.bus_column()?;
        }
        self.second_column()?;
        self.halve_weights()
    }

    /// `L'` over the folded factors, into the slab after `L`'s; nothing
    /// without a second bus.
    fn second_column(&mut self) -> Result<()> {
        let Some(second) = self.second.as_ref() else {
            return Ok(());
        };
        let be = backend()?;
        let rows = self.len as u64;
        let grid = rows.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let q = self.rows / 4;
        let at = (self.width + 1) * q * 3;
        let mut l = self.folded.slice_mut(at..at + q * 3);
        unsafe {
            self.stream
                .launch_builder(&be.zc_bus_column)
                .arg(&self.folded_ptrs)
                .arg(&rows)
                .arg(&second.term_slots)
                .arg(&second.term_coeffs)
                .arg(&second.num_terms)
                .arg(&second.constant)
                .arg(&mut l)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
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
        let tables = [&mut self.eq_r, &mut self.eq_rho]
            .into_iter()
            .chain(self.second.as_mut().map(|second| &mut second.eq_rho));
        for table in tables {
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
    /// and `B(1)`, then — with a second bus — `B'(0)` and `B'(1)`, three u64
    /// each.
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
        let extra = if self.second.is_some() { 2 } else { 0 };
        assert!(rows + extra <= MAX_ROWS, "nodes per launch");
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
        if self.second.is_some() {
            self.second_round_enqueue(half, rows)?;
        }
        Ok(rows + extra)
    }

    /// The second bus's `B'(0)` and `B'(1)` into the sums at rows `at` and
    /// `at + 1`: the round kernel with no node, so its two bus rows alone,
    /// over `L'` and its weight.
    fn second_round_enqueue(&mut self, half: u64, at: usize) -> Result<()> {
        let be = backend()?;
        let (grid, block) = self.shape(half, 2, 3 * self.num_slots as u64)?;
        let q = self.rows / 4;
        let from = (self.width + 1) * q * 3;
        let l = self.folded.slice(from..from + q * 3);
        let none = 0u32;
        let Some(second) = self.second.as_ref() else {
            return Ok(());
        };
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
                .arg(&none)
                .arg(&self.eq_r)
                .arg(&second.eq_rho)
                .arg(&l)
                .arg(&mut self.slots)
                .arg(&mut self.partials)
                .launch(LaunchConfig {
                    grid_dim: (grid, 2, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: block * 3 * 8,
                })?;
        }
        self.reduce_enqueue_at(2, grid, at)
    }

    /// Binds the round's variable to `s` (three u64) in every folded factor and
    /// in `L`; the weights follow when `more` rounds are left on the card.
    pub fn fold(&mut self, s: &[u64; 3], more: bool) -> Result<()> {
        assert!(self.len >= 2, "a fold needs a variable to bind");
        let be = backend()?;
        let half = (self.len / 2) as u64;
        self.stream
            .memcpy_htod(&s[..], &mut self.scratch.slice_mut(0..3))?;
        // The factors and `L` (and `L'`), through the one pointer table.
        let width = (self.width + 1 + usize::from(self.second.is_some())) as u64;
        let total = width * half;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_fold_ext3)
                .arg(&mut self.fold_ptrs)
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
