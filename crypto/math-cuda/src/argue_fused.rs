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

/// Measurement-only switches of the fused zerocheck, read once.
pub mod knobs {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU8, Ordering};

    static BUS_TWICE_FORCED: AtomicU8 = AtomicU8::new(0);

    /// `LAMBDA_VM_ZC_BUS_TWICE=1`: the bus column's two kernels run twice,
    /// each second run writing what the first wrote — the same messages, the
    /// bus's card time paid twice. A measurement arm; off by default.
    pub fn bus_twice() -> bool {
        match BUS_TWICE_FORCED.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("LAMBDA_VM_ZC_BUS_TWICE").is_ok_and(|v| v == "1" || v == "true")
        })
    }

    /// The slot files' budget in bytes: `LAMBDA_VM_ZC_SLOT_BUDGET_MIB` (a
    /// measurement arm: a launch's thread count, and so its summation order
    /// only, changes; field sums are exact) or 512 MiB, the default.
    pub fn slot_budget() -> u64 {
        static BUDGET: OnceLock<u64> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            std::env::var("LAMBDA_VM_ZC_SLOT_BUDGET_MIB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|&mib| (8..=4096).contains(&mib))
                .map_or(super::SLOT_BUDGET_BYTES, |mib| mib << 20)
        })
    }

    static BUS_FROM_GRID_FORCED: AtomicU8 = AtomicU8::new(0);
    static EQ_LEVELS_FORCED: AtomicU8 = AtomicU8::new(0);
    static BUS_ONLY_AT_END_FORCED: AtomicU8 = AtomicU8::new(0);

    fn word(forced: &AtomicU8, cell: &OnceLock<bool>, name: &str) -> bool {
        match forced.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => *cell.get_or_init(|| std::env::var(name).is_ok_and(|v| v == "1" || v == "true")),
        }
    }

    fn set(forced: &AtomicU8, on: Option<bool>) {
        forced.store(
            match on {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            },
            Ordering::Relaxed,
        );
    }

    /// `LAMBDA_VM_ZC_BUS_FROM_GRID=1`: rounds 0 and 1's pass keeps each row's
    /// bus value, and the bus column is that, folded by their two challenges —
    /// the same values as `zc_bus_column` makes from the folded factors (the
    /// weights sum to one), without its product a term a row. Default off.
    pub fn bus_from_grid() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        word(&BUS_FROM_GRID_FORCED, &ON, "LAMBDA_VM_ZC_BUS_FROM_GRID") || bus_only_at_end()
    }

    /// `LAMBDA_VM_ZC_EQ_LEVELS=1`: every round's weights built once a session,
    /// one level a round (`gkr_eq_levels_ext3`), instead of halved by a launch
    /// a round. The same values. Default off.
    pub fn eq_levels() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        word(&EQ_LEVELS_FORCED, &ON, "LAMBDA_VM_ZC_EQ_LEVELS")
    }

    /// `LAMBDA_VM_ZC_BUS_ONLY_AT_END=1`: the factors no round reads (the
    /// bus's alone) are not folded; the cells the host tail takes are made
    /// from the base columns once, every bound challenge at a time. The same
    /// values. Implies [`bus_from_grid`], whose column needs no folded factor.
    /// Default off.
    pub fn bus_only_at_end() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        word(&BUS_ONLY_AT_END_FORCED, &ON, "LAMBDA_VM_ZC_BUS_ONLY_AT_END")
    }

    static STAGE_FAULT: AtomicU8 = AtomicU8::new(0);

    /// ⛔ A FAULT for the variants' negative controls, a test's only: 1 skips
    /// the bus rows' second fold, 2 reads the weights one level off, 3 leaves
    /// one unread factor's cells unbound. 0 (the default) is none.
    #[doc(hidden)]
    pub fn force_stage_fault(fault: u8) {
        STAGE_FAULT.store(fault, Ordering::Relaxed);
    }

    pub(super) fn stage_fault() -> u8 {
        STAGE_FAULT.load(Ordering::Relaxed)
    }

    static SHARED_FORCED: AtomicU8 = AtomicU8::new(0);

    /// `LAMBDA_VM_ZC_SHARED_SLOTS=<log2 half>` (1 to 30): a later round over at
    /// most `2^<log2 half>` cube indices keeps each block's slot file in shared
    /// memory (`zc_round_gruen_shared`) when it fits. The same sums. `None`
    /// (unset, the default) is today's global slot file everywhere.
    pub fn shared_slots() -> Option<u32> {
        static WORD: OnceLock<Option<u32>> = OnceLock::new();
        match SHARED_FORCED.load(Ordering::Relaxed) {
            0 => *WORD.get_or_init(|| {
                std::env::var("LAMBDA_VM_ZC_SHARED_SLOTS")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .filter(|bits| (1..=30).contains(bits))
            }),
            255 => None,
            bits => Some(u32::from(bits)),
        }
    }

    /// Forces [`shared_slots`]: `Some(None)` off, `Some(Some(bits))` on,
    /// `None` back to the environment's word.
    #[doc(hidden)]
    pub fn force_shared_slots(mode: Option<Option<u8>>) {
        SHARED_FORCED.store(
            match mode {
                None => 0,
                Some(None) => 255,
                Some(Some(bits)) => bits.clamp(1, 30),
            },
            Ordering::Relaxed,
        );
    }

    static ONE_WALK_FORCED: AtomicU8 = AtomicU8::new(0);

    /// `LAMBDA_VM_ZC_GRID_ONE_WALK=1`: rounds 0 and 1's pass walks the program
    /// once a group for every grid point (`zc_grid01_all_*`, 5, 9, 12 or 16
    /// points), on the same slot file, instead of once a point. The same sums.
    /// Default off.
    pub fn grid_one_walk() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        word(&ONE_WALK_FORCED, &ON, "LAMBDA_VM_ZC_GRID_ONE_WALK")
    }

    /// Forces [`grid_one_walk`]; `None` back to the environment's word.
    #[doc(hidden)]
    pub fn force_grid_one_walk(on: Option<bool>) {
        set(&ONE_WALK_FORCED, on);
    }

    /// Forces the three words above; `None` back to the environment's.
    #[doc(hidden)]
    pub fn force_stage(bus_from_grid: Option<bool>, eq_levels: Option<bool>, at_end: Option<bool>) {
        set(&BUS_FROM_GRID_FORCED, bus_from_grid);
        set(&EQ_LEVELS_FORCED, eq_levels);
        set(&BUS_ONLY_AT_END_FORCED, at_end);
    }

    /// Forces [`bus_twice`]; `None` back to the environment's word.
    #[doc(hidden)]
    pub fn force_bus_twice(on: Option<bool>) {
        BUS_TWICE_FORCED.store(
            match on {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            },
            Ordering::Relaxed,
        );
    }
}

const BLOCK_DIM: u32 = 256;
const MAX_GRID: u32 = 4096;

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

/// [`knobs::bus_only_at_end`]'s split of a table's factors: those a round
/// reads (their slots, how many, and the fold's pointer table over their slabs
/// and `L`) and those no round reads (their slots on the card and here).
struct Unread {
    reads: CudaSlice<u32>,
    count: u64,
    fold_table: CudaSlice<u64>,
    unread_dev: CudaSlice<u32>,
    unread: Vec<u32>,
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
    /// What the optional timers ([`crate::ktime`]) file this session's
    /// launches under: its program's step bucket, its steps, and the factor
    /// reads a walk of it makes.
    depth: u32,
    steps: u64,
    vars: u64,
    /// Scratch weights for the halvings' pricing arm (`zc.halve` twice).
    shadow_eq: Option<CudaSlice<u64>>,
    /// [`knobs::bus_from_grid`]: every row's bus value (`4q` ext3), its
    /// one-entry pointer table for the fold, and the two challenges' limbs.
    l0: Option<(CudaSlice<u64>, CudaSlice<u64>, CudaSlice<u64>)>,
    /// [`knobs::eq_levels`]: both weights' levels and their variable count
    /// `J`, and the halvings asked for so far.
    levels: Option<(CudaSlice<u64>, CudaSlice<u64>, u32)>,
    halvings: usize,
    /// [`knobs::bus_only_at_end`]: the factors a round reads (their slots, and
    /// the fold's pointer table over their slabs and `L`), and those no round
    /// reads (slots, on the card and here).
    needed: Option<Unread>,
    _room: DeviceReservation,
}

/// Threads a slot file of `bytes_per_thread` affords under the budget.
fn ceiling(bytes_per_thread: u64) -> u64 {
    (knobs::slot_budget() / bytes_per_thread.max(1)).clamp(32, MAX_THREADS)
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

/// Dynamic shared memory a shared-slot round block may take.
const SHARED_SLOT_BYTES: u32 = 96 * 1024;

/// The block of the shared-slot rounds for a program of `num_slots` live ext3
/// slots: the largest power of two up to 256 threads whose reduction words and
/// slot files fit in [`SHARED_SLOT_BYTES`], or `None` when not even one thread's
/// does (or the card refuses the size).
fn shared_block(num_slots: u64) -> Option<u32> {
    static ATTR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ready = *ATTR.get_or_init(|| {
        backend().is_ok_and(|be| {
            be.ctx
                .bind_to_thread()
                .and_then(|()| {
                    be.zc_round_gruen_shared.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        SHARED_SLOT_BYTES as i32,
                    )
                })
                .is_ok()
        })
    });
    if !ready {
        return None;
    }
    let per_thread = 24 + num_slots.max(1) * 24;
    let mut block = 256u64;
    while block > 1 && block * per_thread > u64::from(SHARED_SLOT_BYTES) {
        block /= 2;
    }
    (block * per_thread <= u64::from(SHARED_SLOT_BYTES)).then_some(block as u32)
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
        let (from_grid, levels_on, at_end) = (
            knobs::bus_from_grid(),
            knobs::eq_levels(),
            knobs::bus_only_at_end(),
        );
        // The opt-in variants' buffers: the rows' bus values, the weights' levels.
        let extra = if from_grid { rows as u64 * 24 } else { 0 }
            + if levels_on {
                2 * (rows as u64 / 4) * 24
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
        let steps = (program.nodes.len() / 2) as u64;
        // A walk's factor reads: its `VAR` steps and its selected `ACC` steps.
        let vars = program
            .nodes
            .chunks_exact(2)
            .filter(|node| {
                let op = node[0] as u32;
                op == 1 || (op == 6 && (node[1] as u32) != u32::MAX)
            })
            .count() as u64;
        let l0 = if from_grid {
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
        let levels = if levels_on {
            let j = (r_tail.len() / 3) as u32;
            let mut out = Vec::with_capacity(2);
            for tail in [r_tail, rho_tail] {
                // SAFETY: the kernel writes every cell.
                let mut table = unsafe { alloc_or_trim::<u64>(&stream, q * 3) }?;
                let point = htod_or_trim(&stream, &nonempty(tail))?;
                let low = 0u32;
                unsafe {
                    stream
                        .launch_builder(&be.gkr_eq_levels_ext3)
                        .arg(&point)
                        .arg(&low)
                        .arg(&j)
                        .arg(&mut table)
                        .launch(LaunchConfig {
                            grid_dim: ((q as u64).div_ceil(256).min(4096) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                out.push(table);
            }
            let rho_levels = out.pop().expect("two tables");
            let r_levels = out.pop().expect("two tables");
            Some((r_levels, rho_levels, j))
        } else {
            None
        };
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
            let reads_dev = htod_or_trim(
                &stream,
                if reads.is_empty() {
                    &[0u32][..]
                } else {
                    &reads
                },
            )?;
            let unread_dev = htod_or_trim(
                &stream,
                if unread.is_empty() {
                    &[0u32][..]
                } else {
                    &unread
                },
            )?;
            let fold_dev = htod_or_trim(&stream, &fold_table)?;
            Some(Unread {
                reads: reads_dev,
                count,
                fold_table: fold_dev,
                unread_dev,
                unread,
            })
        } else {
            None
        };
        let session = Self {
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
            depth: crate::ktime::log2_bucket(steps),
            steps,
            vars,
            shadow_eq: None,
            l0,
            levels,
            halvings: 0,
            needed,
            _room: room,
        };
        if crate::ktime::enabled() {
            session.describe(grid_rows, gruen_rows);
        }
        Ok(Some(session))
    }

    /// Under the timers: one line on this session's shape and launches, and,
    /// once a process, the argue kernels' registers, spills and occupancy.
    fn describe(&self, grid_rows: usize, gruen_rows: usize) {
        static ATTRS: std::sync::Once = std::sync::Once::new();
        if let Ok(be) = backend() {
            ATTRS.call_once(|| {
                let mut line = String::from("ARGUE KERNEL ATTRS (registers a thread · local bytes a thread · blocks an SM at 256 threads):");
                for (name, function, shared) in [
                    ("zc_grid01", &be.zc_grid01, 256 * 24),
                    ("zc_round_gruen", &be.zc_round_gruen, 256 * 24),
                    ("zc_bus_u", &be.zc_bus_u, 256 * 24),
                    ("zc_fold2", &be.zc_fold2, 0),
                    ("zc_bus_column", &be.zc_bus_column, 0),
                    ("sumcheck_fold_ext3", &be.sumcheck_fold_ext3, 0),
                    ("gkr_round_gruen", &be.gkr_round_gruen, 256 * 24),
                    ("gkr_round_gruen_real", &be.gkr_round_gruen_real, 256 * 24),
                    ("fraction_fold_ext3", &be.fraction_fold_ext3, 0),
                    ("gkr_input_from_columns", &be.gkr_input_from_columns, 0),
                ] {
                    let (regs, local, blocks) = crate::ktime::attributes(function, 256, shared);
                    line.push_str(&format!(" {name} {regs} · {local} · {blocks} |"));
                }
                eprintln!("{line}");
            });
        }
        let q = (self.rows / 4) as u64;
        let slots = self.num_slots as u64;
        let grid = (grid_rows > 0)
            .then(|| self.shape(q, grid_rows, slots).ok())
            .flatten();
        let rounds = self.shape((q / 2).max(1), gruen_rows, 3 * slots).ok();
        let show = |shape: Option<(u32, u32)>, rows: usize, words: u64| match shape {
            Some((g, b)) => format!(
                "{g}×{b} threads × {rows} rows, slot file {:.1} MiB",
                (u64::from(g) * u64::from(b) * rows as u64 * words * 8) as f64 / 1048576.0
            ),
            None => "none".to_string(),
        };
        eprintln!(
            "ZC SESSION: rows 2^{} · width {} · steps {} (bucket {}, factor reads {}) · live slots {} · bus terms {} · grid {} · rounds {}",
            self.rows.trailing_zeros(),
            self.width,
            self.steps,
            self.depth,
            self.vars,
            self.num_slots,
            self.num_terms,
            show(grid, grid_rows, slots),
            show(rounds, gruen_rows, 3 * slots),
        );
    }

    /// A start event for the next launch, when the timers are on.
    fn tick(&self) -> crate::ktime::Mark {
        crate::ktime::start(&self.stream)
    }

    /// The launch since `mark` done, filed under `kind` and this session's
    /// step bucket with the bytes and work it moves.
    fn tock(&self, kind: &'static str, mark: crate::ktime::Mark, bytes: u64, work: u64) {
        crate::ktime::end(&self.stream, mark, kind, self.depth, bytes, work);
    }

    /// [`tock`](Self::tock) under an explicit depth: the rounds file theirs as
    /// `100 × step bucket + log2 half`, so their cost reads by round size too.
    fn tock_at(
        &self,
        kind: &'static str,
        depth: u32,
        mark: crate::ktime::Mark,
        bytes: u64,
        work: u64,
    ) {
        crate::ktime::end(&self.stream, mark, kind, depth, bytes, work);
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
        for _ in 0..crate::ktime::times("zc.reduce") {
            let t0 = self.tick();
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
            self.tock("zc.reduce", t0, rows as u64 * grid as u64 * 24, rows as u64);
        }
        Ok(())
    }

    /// The `rows` sums a queued launch left, read back once the stream is idle.
    pub fn collect(&self, rows: usize) -> Result<Vec<u64>> {
        let _wait = crate::ktime::HostWait::new("zc.collect", self.depth);
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
        // Twice under the measurement arm: the second run writes what the
        // first wrote.
        for _ in 0..if knobs::bus_twice() || crate::ktime::twice("zc.bus") {
            2
        } else {
            1
        } {
            let t0 = self.tick();
            let config = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: block * 3 * 8,
            };
            unsafe {
                match self.l0.as_mut() {
                    // Each row's bus value kept as well: the bus column's source.
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
            self.tock(
                "zc.bus_u",
                t0,
                q * 4 * self.num_terms * 8 + q * 24,
                q * 4 * self.num_terms,
            );
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
        // One walk for every point ([`knobs::grid_one_walk`]): one block row of
        // the same threads a point row has today, each holding every point's
        // value of a slot — the same slot file.
        let one_walk = if knobs::grid_one_walk() {
            match points.len() {
                5 => Some(&be.zc_grid01_all_5),
                9 => Some(&be.zc_grid01_all_9),
                12 => Some(&be.zc_grid01_all_12),
                16 => Some(&be.zc_grid01_all_16),
                _ => None,
            }
        } else {
            None
        };
        if let Some(kernel) = one_walk {
            for _ in 0..crate::ktime::times("zc.grid") {
                let t0 = self.tick();
                unsafe {
                    self.stream
                        .launch_builder(kernel)
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
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: block * 3 * 8,
                        })?;
                }
                self.tock(
                    "zc.grid",
                    t0,
                    q * (self.vars * 32 + 24),
                    q * points.len() as u64 * self.steps,
                );
            }
            let t = self.reduce(points.len(), grid)?;
            let violation = crate::readback::clone_dtoh(&self.stream, &self.violation)?;
            self.stream.synchronize()?;
            return Ok((t, (violation[0] != u64::MAX).then_some(violation[0])));
        }
        for _ in 0..crate::ktime::times("zc.grid") {
            let t0 = self.tick();
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
            self.tock(
                "zc.grid",
                t0,
                q * points.len() as u64 * (self.vars * 32 + 24),
                q * points.len() as u64 * self.steps,
            );
        }
        let t = self.reduce(points.len(), grid)?;
        let violation = crate::readback::clone_dtoh(&self.stream, &self.violation)?;
        self.stream.synchronize()?;
        Ok((t, (violation[0] != u64::MAX).then_some(violation[0])))
    }

    /// Both folds at once (`w[2a + b] = eq₁(s₀, a)·eq₁(s₁, b)`, twelve u64), into
    /// the quarter-size copy; then `L` there, and the weights halved to round
    /// 2's `eq(·_{>2})`.
    pub fn fold2(&mut self, w: &[u64; 12]) -> Result<()> {
        self.fold2_bound(w, None)
    }

    /// [`fold2`](Self::fold2) with rounds 0 and 1's challenges (`s₀` then
    /// `s₁`, three u64 each), which the opt-in bus column
    /// ([`knobs::bus_from_grid`]) folds its rows' values by. A session that
    /// keeps them refuses a double fold without them.
    pub fn fold2_bound(&mut self, w: &[u64; 12], challenges: Option<&[u64; 6]>) -> Result<()> {
        assert_eq!(self.len, 0, "the double fold is the first");
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        self.stream.memcpy_htod(&w[..], &mut self.scratch)?;
        let factors = self.source.ptrs;
        let selected = self.needed.as_ref().map(|u| (&u.reads, u.count));
        let width = selected.map_or(self.width as u64, |(_, count)| count);
        let total = width * q;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        for _ in 0..crate::ktime::times("zc.fold2") {
            let t0 = self.tick();
            if total > 0 {
                unsafe {
                    match selected {
                        // Only the factors a round reads.
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
            self.tock("zc.fold2", t0, width * q * 56, width * q * 4);
        }
        self.len = q as usize;
        if self.l0.is_some() {
            let Some(challenges) = challenges else {
                return Err(cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                ));
            };
            self.bus_from_rows(challenges)?;
        } else {
            self.bus_column()?;
        }
        self.halve_weights()
    }

    /// `L` from the rows' bus values: folded by `s₀` (the high variable) and
    /// then `s₁`, in place, and its first quarter copied into `L`'s slab.
    fn bus_from_rows(&mut self, challenges: &[u64; 6]) -> Result<()> {
        let be = backend()?;
        let q = (self.rows / 4) as u64;
        let Some((values, table, limbs)) = self.l0.as_mut() else {
            return Ok(());
        };
        self.stream.memcpy_htod(&challenges[..], limbs)?;
        let t0 = crate::ktime::start(&self.stream);
        let rounds: &[(usize, u64)] = if knobs::stage_fault() == 1 {
            &[(0, 2 * q)]
        } else {
            &[(0, 2 * q), (1, q)]
        };
        for &(round, half) in rounds {
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
        crate::ktime::end(
            &self.stream,
            t0,
            "zc.bus_rows",
            self.depth,
            q * 4 * 24 * 2,
            q * 4,
        );
        Ok(())
    }

    /// Whether this session leaves the factors no round reads unfolded
    /// ([`knobs::bus_only_at_end`], read when it was made), so
    /// [`bound_unread`](Self::bound_unread) owes their cells.
    pub fn skips_unread(&self) -> bool {
        self.needed.as_ref().is_some_and(|u| !u.unread.is_empty())
    }

    /// Every bound challenge at once for the factors no round reads
    /// ([`knobs::bus_only_at_end`]): `e` is the `eq` table of the bound
    /// challenges (three u64 a cell, the first bound in the high bit), and the
    /// result each such factor's cells over the cube left — `(slot, limbs)`.
    pub fn bound_unread(&self, e: &[u64]) -> Result<Vec<(usize, Vec<u64>)>> {
        let Some(Unread {
            unread_dev, unread, ..
        }) = self.needed.as_ref()
        else {
            return Ok(Vec::new());
        };
        if unread.is_empty() {
            return Ok(Vec::new());
        }
        let be = backend()?;
        let rows = self.rows as u64;
        let cells = self.len as u64;
        assert_eq!(
            (e.len() / 3) as u64 * cells,
            rows,
            "the table spans the bound variables"
        );
        let nsel = unread.len();
        let threads: u64 = if nsel >= 32 { 4096 } else { 16384 }.max(cells);
        let e_dev = htod_or_trim(&self.stream, e)?;
        // SAFETY: pass one writes every partial, pass two every cell.
        let mut partials =
            unsafe { alloc_or_trim::<u64>(&self.stream, nsel * threads as usize * 3) }?;
        let mut out = unsafe { alloc_or_trim::<u64>(&self.stream, nsel * cells as usize * 3) }?;
        let factors = self.source.ptrs;
        let t0 = crate::ktime::start(&self.stream);
        unsafe {
            self.stream
                .launch_builder(&be.zc_bind_all)
                .arg(factors)
                .arg(unread_dev)
                .arg(&rows)
                .arg(&cells)
                .arg(&e_dev)
                .arg(&self.source.stride)
                .arg(&mut partials)
                .launch(LaunchConfig {
                    grid_dim: ((threads / u64::from(BLOCK_DIM)) as u32, nsel as u32, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            let nsel_arg = nsel as u64;
            let total = nsel as u64 * cells;
            self.stream
                .launch_builder(&be.zc_bind_all_sum)
                .arg(&partials)
                .arg(&nsel_arg)
                .arg(&cells)
                .arg(&threads)
                .arg(&mut out)
                .launch(LaunchConfig {
                    grid_dim: (
                        total
                            .div_ceil(u64::from(BLOCK_DIM))
                            .clamp(1, MAX_GRID as u64) as u32,
                        1,
                        1,
                    ),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        crate::ktime::end(
            &self.stream,
            t0,
            "zc.bound_unread",
            self.depth,
            nsel as u64 * rows * 8,
            nsel as u64 * rows,
        );
        let flat = crate::readback::clone_dtoh(&self.stream, &out)?;
        self.stream.synchronize()?;
        let span = cells as usize * 3;
        let skip = usize::from(knobs::stage_fault() == 3);
        Ok(unread
            .iter()
            .zip(flat.chunks_exact(span))
            .skip(skip)
            .map(|(&slot, limbs)| (slot as usize, limbs.to_vec()))
            .collect())
    }

    /// `L` over the folded factors, into its slab.
    fn bus_column(&mut self) -> Result<()> {
        let be = backend()?;
        let rows = self.len as u64;
        let grid = rows.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let q = self.rows / 4;
        // Twice under the measurement arm: the second run writes what the
        // first wrote.
        for _ in 0..if knobs::bus_twice() || crate::ktime::twice("zc.bus") {
            2
        } else {
            1
        } {
            let t0 = self.tick();
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
            self.tock(
                "zc.bus_column",
                t0,
                rows * (self.num_terms * 24 + 24),
                rows * self.num_terms,
            );
        }
        Ok(())
    }

    /// Both weights to the next round's `eq(·_{>j+1})`: their halves added. The
    /// weights are always half the folded cube after this.
    pub fn halve_weights(&mut self) -> Result<()> {
        self.halvings += 1;
        // The levels were built once ([`knobs::eq_levels`]): nothing to launch.
        if self.levels.is_some() {
            return Ok(());
        }
        let be = backend()?;
        let half = (self.len / 2) as u64;
        let grid = half.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        // Both halvings under one timer pair: the loop holds both weights.
        let t0 = self.tick();
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
        self.tock("zc.halve", t0, half * 144, half);
        // The pricing arm: the same two halvings again, on scratch tables of the
        // weights' size, so nothing the rounds read changes.
        if crate::ktime::twice("zc.halve") {
            if self.shadow_eq.is_none() {
                self.shadow_eq = Some(alloc_zeros_or_trim::<u64>(&self.stream, self.eq_r.len())?);
            }
            if let Some(shadow) = self.shadow_eq.as_mut() {
                for _ in 0..2 {
                    unsafe {
                        self.stream
                            .launch_builder(&be.zc_halve)
                            .arg(&mut *shadow)
                            .arg(&half)
                            .launch(LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (BLOCK_DIM, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                }
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
        // The shared-slot rounds ([`knobs::shared_slots`]): a block whose slot
        // file fits in shared memory, at most `SHARED_SLOT_BYTES` a block.
        let shared_block = knobs::shared_slots()
            .filter(|bits| half <= 1u64 << bits)
            .and_then(|_| shared_block(self.num_slots as u64));
        let (grid, block) = match shared_block {
            Some(block) => (
                half.div_ceil(u64::from(block)).clamp(1, MAX_GRID as u64) as u32,
                block,
            ),
            None => self.shape(half, rows, 3 * self.num_slots as u64)?,
        };
        let num_a = nodes.len() as u32;
        let q = self.rows / 4;
        let l = self
            .folded
            .slice(self.width * q * 3..(self.width + 1) * q * 3);
        // The weights this round reads: today's tables, halved in place, or the
        // level the halvings so far stand at — level `k − 1` after `k`, at
        // `1 + 2^J − 2^{J−k+1}`, `2^{J−k}` cells (`gkr_eq_levels_ext3`'s layout).
        let (eq_r, eq_rho) = match self.levels.as_ref() {
            Some((r_levels, rho_levels, j)) => {
                let (j, mut k) = (*j as usize, self.halvings);
                if knobs::stage_fault() == 2 && k > 1 {
                    k -= 1;
                }
                if k == 0 || k > j || ((1usize << (j - k)) as u64) < half {
                    return Err(cudarc::driver::DriverError(
                        cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                    ));
                }
                let at = 1 + (1usize << j) - (1usize << (j - k + 1));
                let span = at * 3..(at + (1usize << (j - k))) * 3;
                (r_levels.slice(span.clone()), rho_levels.slice(span))
            }
            None => (self.eq_r.slice(..), self.eq_rho.slice(..)),
        };
        for _ in 0..crate::ktime::times("zc.rounds") {
            let t0 = self.tick();
            let (kernel, shared_bytes) = match shared_block {
                Some(block) => (
                    &be.zc_round_gruen_shared,
                    block * 3 * 8 + block * self.num_slots as u32 * 24,
                ),
                None => (&be.zc_round_gruen, block * 3 * 8),
            };
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&self.folded_ptrs)
                    .arg(&half)
                    .arg(&self.nodes)
                    .arg(&self.num_nodes)
                    .arg(&self.ext_consts)
                    .arg(&self.betas)
                    .arg(&self.node_ints)
                    .arg(&num_a)
                    .arg(&eq_r)
                    .arg(&eq_rho)
                    .arg(&l)
                    .arg(&mut self.slots)
                    .arg(&mut self.partials)
                    .launch(LaunchConfig {
                        grid_dim: (grid, rows as u32, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    })?;
            }
            self.tock_at(
                "zc.rounds",
                self.depth * 100 + half.trailing_zeros(),
                t0,
                half * (u64::from(num_a) * self.vars * 48 + 48 + (u64::from(num_a) + 2) * 24),
                half * u64::from(num_a) * self.steps,
            );
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
        // The factors and `L`, through the one pointer table of `width + 1` — or,
        // with [`knobs::bus_only_at_end`], of the factors a round reads and `L`.
        let (table, width) = match self.needed.as_mut() {
            Some(u) => (&mut u.fold_table, u.count + 1),
            None => (&mut self.fold_ptrs, self.width as u64 + 1),
        };
        let total = width * half;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let t0 = crate::ktime::start(&self.stream);
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
        self.tock("zc.fold", t0, width * half * 72, width * half);
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
        let t0 = self.tick();
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
        self.tock("zc.gather", t0, total as u64 * 16, width * cells);
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
