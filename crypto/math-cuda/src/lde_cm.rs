//! Column-major coset LDE engine (GAP K1), behind `LAMBDA_VM_GAP_K1=1`.
//!
//! The production LDEs ([`crate::lde`]) run one butterfly level per launch over
//! the whole matrix: an LDE of `2^22 × 245` at blowup 2 makes ~33 whole-matrix
//! DRAM passes. This engine instead transforms the columns a CHUNK at a time,
//! the chunk sized so its working set stays in L2 between launches, and runs
//! 4..8 levels per launch out of registers and shared memory
//! (`kernels/ntt_cm.cu`). DRAM then sees the trace about once and the LDE about
//! once, and the LDE comes out column-major — the layout every downstream kernel
//! reads — so the row-major paths also lose their in-place transpose.
//!
//! Per chunk of `C` columns the driver runs:
//!
//! 1. the iNTT of each column, DIF (natural in, bit-reversed out) with the
//!    inverse roots and no `1/n`: the first pass reads the caller's source and
//!    writes the chunk's scratch `W` (`C × n`), so the source is never written;
//!    the last pass multiplies position `p` by `weights[rev(p)]` — the caller's
//!    coset weights, which carry the `1/n`;
//! 2. the forward NTT at `N = n·blowup`, DIT (bit-reversed in, natural out): the
//!    first pass reads `W` with the coset spread fused in (position `B·q` is
//!    `W[q]`, every other position zero — never stored), the rest run in place
//!    in the destination column.
//!
//! The field values are those of the legacy pipeline — the LDE of a column is
//! one polynomial evaluated on one coset, and both pipelines use the caller's
//! weights and the same roots of unity — so roots and every downstream value
//! are unchanged; only the non-canonical representation of a value may differ,
//! which nothing downstream observes (the hashes and the serialisation
//! canonicalise).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;

use crate::Result;
use crate::device::Backend;

/// The knob: `LAMBDA_VM_GAP_K1=1` routes the production LDEs through this
/// engine. Default off; read once per process.
pub const K1_ENV: &str = "LAMBDA_VM_GAP_K1";

thread_local! {
    /// Per-thread override of [`K1_ENV`], for parity tests that run both
    /// engines in one process. See [`with_k1`].
    static K1_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the calling thread's LDEs take this engine.
pub fn k1_enabled() -> bool {
    if let Some(on) = K1_OVERRIDE.with(Cell::get) {
        return on;
    }
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        let on = std::env::var(K1_ENV).is_ok_and(|v| v == "1");
        // One line per process, so every box log states which LDE it ran.
        eprintln!(
            "[gpu] GAP K1 column-major LDE: {}",
            if on { "on" } else { "off" }
        );
        on
    })
}

/// Run `f` with the engine forced on or off for the calling thread — the
/// entry points decide on the calling thread before any work is queued, so a
/// test can compare both engines on the same inputs in one process.
#[doc(hidden)]
pub fn with_k1<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            K1_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(K1_OVERRIDE.with(|c| c.replace(Some(on))));
    f()
}

/// Columns this engine has extended, process-wide: the mechanism counter a
/// knob-on run is checked against (a knob that routes nothing reads zero).
static K1_COLUMNS: AtomicU64 = AtomicU64::new(0);

/// See [`K1_COLUMNS`].
pub fn k1_columns() -> u64 {
    K1_COLUMNS.load(Ordering::Relaxed)
}

/// Smallest trace a column can have here: a pass holds 16 elements per
/// thread, so a transform needs at least 2^4 of them.
pub const MIN_LOG_N: u32 = 4;

/// Largest blowup the fused spread covers: the first forward pass must hold a
/// whole coset of `B` spread positions (`B <= 2^k`, `k >= 4`).
pub const MAX_LOG_BLOWUP: u32 = 4;

/// Whether an `n`-row, blowup-`b` LDE fits the engine's shapes. Callers take
/// the legacy path otherwise.
pub fn supports(n: usize, blowup: usize) -> bool {
    n.is_power_of_two()
        && blowup.is_power_of_two()
        && blowup >= 2
        && n.trailing_zeros() >= MIN_LOG_N
        && blowup.trailing_zeros() <= MAX_LOG_BLOWUP
        && (n * blowup).trailing_zeros() <= GoldilocksField::TWO_ADICITY as u32
}

/// Pass sizes for a length-`2^l` transform: `ceil(l / 8)` passes of 4..=8
/// levels, within one of each other. Mirrored by the host KAT's `plan`.
pub(crate) fn plan(l: u32) -> Vec<u32> {
    assert!(l >= MIN_LOG_N, "transform too short for the engine: 2^{l}");
    let p = l.div_ceil(8);
    let mut ks = vec![l / p; p as usize];
    for k in ks.iter_mut().take((l % p) as usize) {
        *k += 1;
    }
    debug_assert!(ks.iter().all(|&k| (4..=8).contains(&k)));
    ks
}

/// log2 of the lanes a block spans: 256 threads at 16 elements each hold
/// `4096 >> k` sub-transforms' worth, capped by what the level window allows
/// (`2^s` lanes of `lo` for a strided pass, `2^(l-k)` sub-transforms for the
/// contiguous one). Mirrored by the host KAT's `pass_log_t`.
fn pass_log_t(l: u32, s: u32, k: u32) -> u32 {
    let cap = 12 - k;
    let lim = if s > 0 { s } else { l - k };
    cap.min(lim)
}

/// Kernel flags, as in `kernels/ntt_cm.cu`.
const F_SPREAD: u32 = 1;
const F_GATHER: u32 = 2;
const F_STORE_W: u32 = 4;

/// The windowed root tables the kernels take: `t[w·256 + j] = root^(j << 8w)`
/// for the field's primitive `2^32`-th root (`fwd`) and its inverse (`inv`).
pub(crate) struct RootTables {
    fwd: CudaSlice<u64>,
    inv: CudaSlice<u64>,
}

fn windowed(root: FieldElement<GoldilocksField>) -> Vec<u64> {
    let mut out = Vec::with_capacity(1024);
    let mut base = root;
    for _ in 0..4 {
        let mut acc = FieldElement::<GoldilocksField>::one();
        for _ in 0..256 {
            out.push(*acc.value());
            acc = &acc * &base;
        }
        // base^(2^8) for the next window.
        for _ in 0..8 {
            base = base.square();
        }
    }
    out
}

pub(crate) fn root_tables(be: &Backend) -> Result<&'static RootTables> {
    static TABLES: OnceLock<RootTables> = OnceLock::new();
    if let Some(t) = TABLES.get() {
        return Ok(t);
    }
    let root = GoldilocksField::get_primitive_root_of_unity(32).expect("2^32-th root");
    let root_inv = root.inv().expect("root is non-zero");
    let stream = be.next_stream();
    let fwd = stream.clone_htod(&windowed(root))?;
    let inv = stream.clone_htod(&windowed(root_inv))?;
    stream.synchronize()?;
    // A racing initialiser's tables are dropped; either set is the same bytes.
    let _ = TABLES.set(RootTables { fwd, inv });
    Ok(TABLES.get().expect("root tables just set"))
}

/// Columns per chunk: as many whole columns (trace scratch plus LDE) as fit
/// in 60% of L2, at least one, at most `m` — ZisK's chunk rule, which leaves
/// room for the tables and the stream's other traffic. The scratch is one
/// chunk of trace columns, so it too stays within 60% of L2.
fn chunk_cols(be: &Backend, n: usize, lde: usize, m: usize) -> usize {
    static L2: OnceLock<usize> = OnceLock::new();
    let l2 = *L2.get_or_init(|| {
        be.ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .ok()
            .filter(|&b| b > 0)
            .map(|b| b as usize)
            .unwrap_or(32 << 20)
    });
    let per_col = (n + lde) * 8;
    // gridDim.y is the column: at most 65535 per launch.
    (l2 * 3 / 5 / per_col).clamp(1, m.clamp(1, 65535))
}

/// The order a chunk loop walks the columns in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChunkOrder {
    /// Any source that no chunk's output overlaps.
    Ascending,
    /// A compact source at the head of the destination buffer (column `c` at
    /// `c·n`): chunk `[c0, c0 + C)` writes `[c0·N, (c0 + C)·N)`, which covers
    /// source columns `[c0·B, (c0 + C)·B)` — all at or above `c0`, so walking
    /// down, every one of them has been read (the chunk's own ones by its
    /// first pass, before its output is written).
    Descending,
}

/// What the source columns hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Input {
    /// Evaluations on the trace domain, natural order: iNTT, weights, NTT.
    Evals,
    /// Coefficients, natural order: weights, NTT (no iNTT).
    Coeffs,
}

/// Where column `c` of an operand lives: `ptr + c·stride` elements.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cols {
    pub ptr: sys::CUdeviceptr,
    pub stride: u64,
}

impl Cols {
    fn at(self, c: usize) -> sys::CUdeviceptr {
        self.ptr + (c as u64) * self.stride * 8
    }
}

/// One pass launch over `cols` columns.
#[allow(clippy::too_many_arguments)]
fn launch_pass(
    stream: &CudaStream,
    be: &Backend,
    dit: bool,
    k: u32,
    l: u32,
    s: u32,
    lb: u32,
    flags: u32,
    input: Cols,
    output: Cols,
    roots: sys::CUdeviceptr,
    wtab: sys::CUdeviceptr,
    cols: usize,
) -> Result<()> {
    let log_t = pass_log_t(l, s, k);
    let threads = (1u32 << log_t) << (k - 4);
    let cfg = LaunchConfig {
        grid_dim: (1u32 << (l - k - log_t), cols as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (1u32 << k) * ((1u32 << log_t) + 1) * 8,
    };
    let idx = (k - 4) as usize;
    let f = if dit {
        &be.ntt_cm_dit[idx]
    } else {
        &be.ntt_cm_dif[idx]
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input.ptr)
            .arg(&input.stride)
            .arg(&output.ptr)
            .arg(&output.stride)
            .arg(&roots)
            .arg(&wtab)
            .arg(&l)
            .arg(&s)
            .arg(&log_t)
            .arg(&lb)
            .arg(&flags)
            .launch(cfg)?;
    }
    Ok(())
}

/// The coset LDE of `m` columns on `stream`: column `c` of `src` (`n` values,
/// natural order, `input` says evaluations or coefficients) becomes column `c`
/// of `dst` (`n·blowup` evaluations on the coset, natural order, stride
/// `n·blowup`). `weights` are the caller's coset weights in natural order:
/// `g^j / n` for [`Input::Evals`], `g^j` for [`Input::Coeffs`].
///
/// `src` is only read, and is left intact unless it lies inside `dst` — which
/// [`ChunkOrder::Descending`] then makes safe for a compact head source.
/// Nothing synchronises; the scratch is freed stream-ordered.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lde_columns(
    stream: &Arc<CudaStream>,
    be: &Backend,
    src: Cols,
    dst: sys::CUdeviceptr,
    m: usize,
    n: usize,
    blowup: usize,
    weights: &[u64],
    input: Input,
    order: ChunkOrder,
) -> Result<()> {
    assert!(supports(n, blowup), "K1 LDE shape n={n} blowup={blowup}");
    assert_eq!(weights.len(), n, "one weight per trace row");
    if m == 0 {
        return Ok(());
    }
    let lde = n * blowup;
    let log_n = n.trailing_zeros();
    let lb = blowup.trailing_zeros();
    let log_lde = log_n + lb;
    let tables = root_tables(be)?;
    let dst = Cols {
        ptr: dst,
        stride: lde as u64,
    };

    // The weights at bit-reversed positions: the iNTT leaves coefficient
    // `rev(p)` at `p`, and the coefficient form gathers at the same index.
    let mut wbr = stream.clone_htod(weights)?;
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute)
            .arg(&mut wbr)
            .arg(&(n as u64))
            .arg(&(log_n as u64))
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    let chunk = chunk_cols(be, n, lde, m);
    // The chunk's trace-domain scratch: the iNTT's output (evaluations), or a
    // copy of the coefficients (so a source inside `dst` is read before the
    // chunk's output overwrites it). SAFETY: every element a later pass reads
    // is written first — by the iNTT's first pass or by the copy.
    let scratch = unsafe { stream.alloc::<u64>(chunk * n) }?;
    let (fwd_roots, _fr) = tables.fwd.device_ptr(stream);
    let (inv_roots, _ir) = tables.inv.device_ptr(stream);
    let (wtab, _wr) = wbr.device_ptr(stream);
    let (scratch_ptr, _sr) = scratch.device_ptr(stream);
    let w = Cols {
        ptr: scratch_ptr,
        stride: n as u64,
    };
    if input == Input::Coeffs {
        // Raw driver copies below: the context must be current on this thread.
        be.ctx.bind_to_thread()?;
    }

    let inv_plan = plan(log_n);
    let fwd_plan = plan(log_lde);
    let chunks: Vec<(usize, usize)> = (0..m)
        .step_by(chunk)
        .map(|c0| (c0, chunk.min(m - c0)))
        .collect();
    let walk: Box<dyn Iterator<Item = &(usize, usize)>> = match order {
        ChunkOrder::Ascending => Box::new(chunks.iter()),
        ChunkOrder::Descending => Box::new(chunks.iter().rev()),
    };
    for &(c0, cols) in walk {
        let chunk_src = Cols {
            ptr: src.at(c0),
            stride: src.stride,
        };
        let chunk_dst = Cols {
            ptr: dst.at(c0),
            stride: dst.stride,
        };
        match input {
            Input::Evals => {
                // DIF passes descend to s = 0.
                let mut s = log_n;
                for (i, &k) in inv_plan.iter().enumerate() {
                    s -= k;
                    let last = i + 1 == inv_plan.len();
                    launch_pass(
                        stream,
                        be,
                        false,
                        k,
                        log_n,
                        s,
                        0,
                        if last { F_STORE_W } else { 0 },
                        if i == 0 { chunk_src } else { w },
                        w,
                        inv_roots,
                        wtab,
                        cols,
                    )?;
                }
            }
            Input::Coeffs => {
                for i in 0..cols {
                    // SAFETY: both ranges are `n` elements inside live
                    // allocations (the caller's source column, the scratch),
                    // queued on the stream every pass of this chunk runs on.
                    unsafe {
                        sys::cuMemcpyDtoDAsync_v2(
                            w.at(i),
                            chunk_src.at(i),
                            n * 8,
                            stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
            }
        }
        let spread = match input {
            Input::Evals => F_SPREAD,
            Input::Coeffs => F_SPREAD | F_GATHER,
        };
        // DIT passes ascend from s = 0.
        let mut s = 0;
        for (i, &k) in fwd_plan.iter().enumerate() {
            launch_pass(
                stream,
                be,
                true,
                k,
                log_lde,
                s,
                lb,
                if i == 0 { spread } else { 0 },
                if i == 0 { w } else { chunk_dst },
                chunk_dst,
                fwd_roots,
                wtab,
                cols,
            )?;
            s += k;
        }
    }
    K1_COLUMNS.fetch_add(m as u64, Ordering::Relaxed);
    Ok(())
}

/// Hash the leaves of a column-major matrix: `num_cols` columns of `num_rows`
/// rows at `cols` (column stride `col_stride` elements), `rows_per_leaf`
/// bit-reversed rows per leaf, into `leaves` (`num_rows / rows_per_leaf`
/// digests of 32 bytes). The column-major twins of the row-major leaf
/// kernels: each absorbs the same elements in the same order (row by row,
/// column by column), which the parity suite pins.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_col_major_leaves(
    hash: crate::DeviceHash,
    stream: &CudaStream,
    be: &Backend,
    cols: sys::CUdeviceptr,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    rows_per_leaf: usize,
    leaves: sys::CUdeviceptr,
) -> Result<()> {
    use crate::DeviceHash;
    assert!(
        rows_per_leaf == 1 || rows_per_leaf == 2,
        "rows_per_leaf must be 1 or 2"
    );
    // Every kernel derives rows as `__brevll(..) >> (64 - log_num_rows)`, UB
    // at `log_num_rows == 0`.
    assert!(num_rows >= 2 && num_rows.is_power_of_two());
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let threads = num_rows / rows_per_leaf as u64;
    let (kernel, cfg) = match (hash, rows_per_leaf) {
        (DeviceHash::Keccak256, 2) => (
            &be.keccak256_leaves_base_row_pair_batched,
            crate::merkle::keccak_launch_cfg(threads),
        ),
        (DeviceHash::Keccak256, _) => (
            &be.keccak256_leaves_base_batched,
            crate::merkle::keccak_launch_cfg(threads),
        ),
        (DeviceHash::Blake3, 2) => (
            &be.blake3_leaves_base_row_pair_batched,
            crate::blake3::blake3_launch_cfg(threads),
        ),
        (DeviceHash::Blake3, _) => (
            &be.blake3_leaves_base_batched,
            crate::blake3::blake3_launch_cfg(threads),
        ),
        (DeviceHash::Rpx256, 2) => (
            &be.rpx_leaves_base_row_pair_batched,
            crate::rpx::rpx_launch_cfg(threads),
        ),
        (DeviceHash::Rpx256, _) => (
            &be.rpx_leaves_base_batched,
            crate::rpx::rpx_launch_cfg(threads),
        ),
        (DeviceHash::Rpo256 | DeviceHash::Poseidon, _) => {
            unimplemented!("{hash:?} device commit not yet ported (column-major leaves)")
        }
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&cols)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(&leaves)
            .launch(cfg)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_cover_every_level_in_passes_of_four_to_eight() {
        for l in MIN_LOG_N..=32 {
            let ks = plan(l);
            assert_eq!(ks.iter().sum::<u32>(), l, "l = {l}");
            assert_eq!(ks.len() as u32, l.div_ceil(8), "l = {l}: pass count");
            assert!(ks.iter().all(|&k| (4..=8).contains(&k)), "l = {l}: {ks:?}");
            let (lo, hi) = (ks.iter().min().unwrap(), ks.iter().max().unwrap());
            assert!(hi - lo <= 1, "l = {l}: uneven {ks:?}");
        }
    }

    /// Every (s, k) window the driver launches: DIT ascending from s = 0, DIF
    /// descending to s = 0.
    fn windows(l: u32) -> Vec<(u32, u32)> {
        let ks = plan(l);
        let mut out = Vec::new();
        let mut s = 0;
        for &k in &ks {
            out.push((s, k));
            s += k;
        }
        let mut s = l;
        for &k in &ks {
            s -= k;
            out.push((s, k));
        }
        out
    }

    #[test]
    fn a_block_never_exceeds_256_threads_or_its_window() {
        for l in MIN_LOG_N..=28 {
            for (s, k) in windows(l) {
                let log_t = pass_log_t(l, s, k);
                let threads = (1u32 << log_t) << (k - 4);
                assert!(threads <= 256, "l={l} s={s} k={k}: {threads} threads");
                if s > 0 {
                    assert!(log_t <= s, "l={l} s={s} k={k}: lanes beyond 2^s");
                } else {
                    assert!(
                        log_t <= l - k,
                        "l={l} k={k}: more sub-transforms than exist"
                    );
                }
                assert!(s + k <= l, "l={l} s={s} k={k}: window past the transform");
                let smem = (1u32 << k) * ((1u32 << log_t) + 1) * 8;
                assert!(
                    smem <= 48 * 1024,
                    "l={l} s={s} k={k}: {smem} B of shared memory"
                );
            }
        }
    }

    #[test]
    fn windowed_tables_are_powers_of_the_two_adic_root() {
        let root = GoldilocksField::get_primitive_root_of_unity(32).unwrap();
        let t = windowed(root);
        assert_eq!(t.len(), 1024);
        for w in 0..4u32 {
            for j in [0u64, 1, 2, 17, 128, 255] {
                let want = root.pow(j << (8 * w));
                assert_eq!(
                    FieldElement::<GoldilocksField>::from_raw(t[(w * 256) as usize + j as usize]),
                    want,
                    "window {w} entry {j}"
                );
            }
        }
        // The top window is the radix table the kernels read: ω_256^j.
        let w256 = GoldilocksField::get_primitive_root_of_unity(8).unwrap();
        for j in 0..128u64 {
            assert_eq!(
                FieldElement::<GoldilocksField>::from_raw(t[768 + j as usize]),
                w256.pow(j),
                "radix entry {j}"
            );
        }
    }

    #[test]
    fn shapes_outside_the_engine_are_declined() {
        assert!(supports(1 << 4, 2));
        assert!(supports(1 << 22, 16));
        assert!(!supports(1 << 3, 2), "trace below 2^4");
        assert!(!supports(1 << 10, 1), "blowup 1 has no spread");
        assert!(!supports(1 << 10, 32), "blowup beyond the fused spread");
        assert!(!supports(3 << 10, 2), "not a power of two");
    }
}
