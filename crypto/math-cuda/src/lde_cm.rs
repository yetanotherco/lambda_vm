//! Column-major coset LDE engine — the device LDE of the commits.
//!
//! The legacy LDEs in [`crate::lde`] run one butterfly level per launch over
//! the whole matrix: an LDE of `2^22 × 245` at blowup 2 makes ~33 whole-matrix
//! DRAM passes. This engine instead transforms the columns a CHUNK at a time,
//! the chunk sized so its working set stays in L2 between launches, and runs
//! 4..8 levels per launch out of registers and shared memory
//! (`kernels/ntt_cm.cu`). DRAM then sees the trace about once and the LDE about
//! once, and the LDE comes out column-major — the layout every downstream kernel
//! reads — so the row-major commits also lose their in-place transpose. On the
//! W3/K0 shapes it runs within 1.2–1.4× of ZisK's production LDE and cuts the
//! whole row-major commit to 0.58–0.67× (`thoughts/zf/gap/fix/NTT.md`).
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
//!
//! The WHIR commit's encoding (`whir::commit_from`) is one column of the same
//! forward transform with no coset: the coefficients, spread at the blowup's
//! spacing, go through the forward passes ([`spread_ntt_column`]) with the
//! spread fused into the first. It reads an 8 KiB windowed root table, where
//! the legacy encoding reads `Backend::fwd_twiddles_for`: `2^(log_n − 1)`
//! values per codeword size, built on the host and cached on the card for the
//! life of the process, outside the VRAM ledger (512 MiB for a `2^27`
//! codeword, 2 GiB for `2^29`). The encoding takes passes of up to 10 levels
//! ([`plan_wide`]): three passes at the production `2^28` instead of four
//! ([`with_ntt_wide`] holds [`plan`]'s for a parity test). Every pass is bound by memory
//! bandwidth, so the passes are the cost; the trace LDEs keep [`plan`].
//!
//! It is the default wherever it applies ([`engine_enabled`]). The legacy path
//! stays for what the engine does not take — a caller that wants the row-major
//! LDE back on the host, and shapes outside [`supports`] (for the encoding,
//! [`spread_supports`]) — and, whole, under a test's [`with_engine`].

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

thread_local! {
    /// Per-thread override of the engine, for parity tests that run both
    /// pipelines (the engine and the legacy per-level one, the same values,
    /// roots and proofs) in one process. See [`with_engine`].
    static ENGINE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the calling thread's LDEs and WHIR encodings take this engine (where
/// it applies): yes, unless a test's [`with_engine`] says otherwise.
pub fn engine_enabled() -> bool {
    ENGINE_OVERRIDE.with(Cell::get).unwrap_or(true)
}

/// Run `f` with the engine forced on or off for the calling thread — the
/// entry points decide on the calling thread before any work is queued, so a
/// test can compare both pipelines on the same inputs in one process.
#[doc(hidden)]
pub fn with_engine<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENGINE_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(ENGINE_OVERRIDE.with(|c| c.replace(Some(on))));
    f()
}

/// Columns this engine has extended, process-wide: the counter a test reads to
/// know the engine, and not the legacy path, did the work.
static COLUMNS_EXTENDED: AtomicU64 = AtomicU64::new(0);

/// See [`COLUMNS_EXTENDED`].
pub fn columns_extended() -> u64 {
    COLUMNS_EXTENDED.load(Ordering::Relaxed)
}

/// Codewords [`spread_ntt_column`] has encoded, process-wide: the counter a
/// test reads to know the engine, and not the legacy path, did a WHIR encoding.
static CODEWORDS_ENCODED: AtomicU64 = AtomicU64::new(0);

/// See [`CODEWORDS_ENCODED`].
pub fn codewords_encoded() -> u64 {
    CODEWORDS_ENCODED.load(Ordering::Relaxed)
}

thread_local! {
    /// Per-thread override of the wide plan, for parity tests that run both
    /// plans in one process. See [`with_ntt_wide`].
    static NTT_WIDE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the calling thread's WHIR encodings take [`plan_wide`]: always, but
/// under a [`with_ntt_wide`] that holds [`plan`]'s passes of up to 8. The
/// codeword's values are the same either way; only a value's non-canonical
/// representation may differ, as between this engine and the legacy one.
pub fn ntt_wide_enabled() -> bool {
    NTT_WIDE_OVERRIDE.with(Cell::get).unwrap_or(true)
}

/// Run `f` with the wide plan forced on or off for the calling thread.
#[doc(hidden)]
pub fn with_ntt_wide<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            NTT_WIDE_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(NTT_WIDE_OVERRIDE.with(|c| c.replace(Some(on))));
    f()
}

/// Codewords [`spread_ntt_column`] encoded with a pass of more than 8 levels,
/// process-wide: what a test reads to know the wide kernels ran.
static WIDE_CODEWORDS_ENCODED: AtomicU64 = AtomicU64::new(0);

/// See [`WIDE_CODEWORDS_ENCODED`].
pub fn wide_codewords_encoded() -> u64 {
    WIDE_CODEWORDS_ENCODED.load(Ordering::Relaxed)
}

/// Smallest trace a column can have here: a pass holds 16 elements per
/// thread, so a transform needs at least 2^4 of them.
pub const MIN_LOG_N: u32 = 4;

/// Largest blowup the fused spread covers at EVERY transform size: the first
/// forward pass must hold a whole coset of `B` spread positions (`B <= 2^k`),
/// and the shortest first pass is `k = 4`.
pub const MAX_LOG_BLOWUP: u32 = 4;

/// Largest spacing [`spread_ntt_column`] takes where the transform's first
/// pass is long enough to hold it ([`spread_supports`]): `2^6`, the WHIR
/// encoding's blowup 64. Every encode of `2^20` positions or more has a first
/// pass of at least 6 levels in either plan.
pub const MAX_SPREAD_LOG_BLOWUP: u32 = 6;

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

/// Whether a length-`2^log_n` transform of an input spread at spacing
/// `2^log_blowup` fits [`spread_ntt_column`] in the plan in force on the
/// calling thread: the first pass must hold a whole spacing (`2^log_blowup <=
/// 2^k`) — always up to [`MAX_LOG_BLOWUP`], and up to
/// [`MAX_SPREAD_LOG_BLOWUP`] where the first pass has at least `log_blowup`
/// levels (the spread stage tests a position's low `log_blowup` bits inside
/// one first-pass sub-transform). Spacing 1 (`log_blowup = 0`) is a plain
/// transform of a bit-reversed input. A shape outside takes the legacy path.
pub fn spread_supports(log_n: u32, log_blowup: u32) -> bool {
    log_n >= MIN_LOG_N
        && log_n <= GoldilocksField::TWO_ADICITY as u32
        && log_blowup <= log_n
        && (log_blowup <= MAX_LOG_BLOWUP
            || (log_blowup <= MAX_SPREAD_LOG_BLOWUP && log_blowup <= spread_plan(log_n)[0]))
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

/// The largest pass [`plan_wide`] takes: the kernels instantiate wide passes
/// of 9 and 10 levels (`ntt_cm_dit_k9`, `_k10`).
pub(crate) const WIDE_MAX_K: u32 = 10;

/// Pass sizes for the WHIR encoding's length-`2^l` transform
/// ([`ntt_wide_enabled`]): `ceil(l / 10)` passes of 4..=10 levels, within one of
/// each other — `[10, 9, 9]` at `2^28`. Where that is as many passes as
/// [`plan`] it is [`plan`]. Mirrored by the host KAT's `plan_wide`.
pub(crate) fn plan_wide(l: u32) -> Vec<u32> {
    assert!(l >= MIN_LOG_N, "transform too short for the engine: 2^{l}");
    let p = l.div_ceil(WIDE_MAX_K);
    let mut ks = vec![l / p; p as usize];
    for k in ks.iter_mut().take((l % p) as usize) {
        *k += 1;
    }
    debug_assert!(ks.iter().all(|&k| (4..=WIDE_MAX_K).contains(&k)));
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

/// A pass's dynamic shared memory: its `2^k × (T + 1)` tile, and on a wide
/// pass (k > 8) a skew of `T` words every 16 rows. Mirrors the kernels'
/// `tile_words`. The radix table is static: `max(2^(k−1), 128)` words.
fn tile_bytes(k: u32, log_t: u32) -> u32 {
    let mut words = (1u32 << k) * ((1u32 << log_t) + 1);
    if k > 8 {
        words += (1u32 << (k - 4)) << log_t;
    }
    words * 8
}

/// Kernel flags, as in `kernels/ntt_cm.cu`.
const F_SPREAD: u32 = 1;
const F_GATHER: u32 = 2;
const F_STORE_W: u32 = 4;
const F_MOBIUS: u32 = 8;

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
        shared_mem_bytes: tile_bytes(k, log_t),
    };
    // The wide passes are DIT only (the encoding's forward transform).
    debug_assert!(dit || k <= 8, "no DIF pass of {k} levels");
    let f = match (dit, k) {
        (true, 9..) => &be.ntt_cm_dit_wide[(k - 9) as usize],
        (true, _) => &be.ntt_cm_dit[(k - 4) as usize],
        (false, _) => &be.ntt_cm_dif[(k - 4) as usize],
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
    assert!(
        supports(n, blowup),
        "column-engine LDE shape n={n} blowup={blowup}"
    );
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
    COLUMNS_EXTENDED.fetch_add(m as u64, Ordering::Relaxed);
    Ok(())
}

/// The plan [`spread_ntt_column`] takes on the calling thread.
fn spread_plan(log_n: u32) -> Vec<u32> {
    if ntt_wide_enabled() {
        plan_wide(log_n)
    } else {
        plan(log_n)
    }
}

/// How many low Möbius levels [`spread_ntt_column`]'s first pass holds for a
/// length-`2^log_n` transform at spacing `2^log_blowup`, in the plan in force
/// on the calling thread: a block stages `2^(k + log T)` positions, so
/// `2^(k + log T − log_blowup)` consecutive coefficients — `12 − log_blowup`
/// levels once the transform reaches 2^12.
pub(crate) fn fused_mobius_levels(log_n: u32, log_blowup: u32) -> u32 {
    let k = spread_plan(log_n)[0];
    k + pass_log_t(log_n, 0, k) - log_blowup
}

/// The forward transform of ONE spread column — the WHIR commit's encoding
/// (`whir::commit_from`; the legacy path is `ntt.cu`'s `lift_spread` then
/// `ntt::run_ntt_body`), fused: position `2^log_blowup · q` of the
/// length-`2^log_n` DIT input (bit-reversed in, natural out) is `src[q]`,
/// every other position zero, and `dst` receives the natural-order transform.
/// The spread is the first pass's load, so the padding is never stored or
/// read, and the transform is `ceil(log_n / 8)` passes instead of the spread
/// pass, the fused 8-level pass and one 5-level tile per five levels above it
/// — `ceil(log_n / 10)` under [`plan_wide`], the default.
/// `src` (`2^(log_n - log_blowup)` values) and `dst` (`2^log_n`) must not
/// overlap. Nothing synchronises.
///
/// With `mobius_levels` (non-zero, and then exactly
/// [`fused_mobius_levels`]) the first pass also runs the encoding's Möbius
/// levels `[0, mobius_levels)` over the coefficients it stages: the caller has
/// run the others (the levels commute).
pub(crate) fn spread_ntt_column(
    stream: &CudaStream,
    be: &Backend,
    src: sys::CUdeviceptr,
    dst: sys::CUdeviceptr,
    log_n: u32,
    log_blowup: u32,
    mobius_levels: u32,
) -> Result<()> {
    assert!(
        spread_supports(log_n, log_blowup),
        "a spread transform of 2^{log_n} at spacing 2^{log_blowup} is outside the engine"
    );
    assert!(
        mobius_levels == 0 || mobius_levels == fused_mobius_levels(log_n, log_blowup),
        "the first pass of 2^{log_n} holds {} Möbius levels, not {mobius_levels}",
        fused_mobius_levels(log_n, log_blowup)
    );
    let tables = root_tables(be)?;
    let (fwd_roots, _fr) = tables.fwd.device_ptr(stream);
    let one = |ptr| Cols { ptr, stride: 0 };
    let ks = spread_plan(log_n);
    let first_flags = if mobius_levels > 0 {
        F_SPREAD | F_MOBIUS
    } else {
        F_SPREAD
    };
    let mut s = 0;
    for (i, &k) in ks.iter().enumerate() {
        let first = i == 0;
        launch_pass(
            stream,
            be,
            true,
            k,
            log_n,
            s,
            log_blowup,
            if first { first_flags } else { 0 },
            one(if first { src } else { dst }),
            one(dst),
            fwd_roots,
            0,
            1,
        )?;
        s += k;
    }
    CODEWORDS_ENCODED.fetch_add(1, Ordering::Relaxed);
    if ks.iter().any(|&k| k > 8) {
        WIDE_CODEWORDS_ENCODED.fetch_add(1, Ordering::Relaxed);
    }
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
    if hash == DeviceHash::Poseidon1 {
        return crate::p1_stark::launch_leaves_cols_ptr(
            stream,
            cols,
            col_stride,
            num_cols,
            num_rows,
            rows_per_leaf,
            leaves,
        );
    }
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
        (DeviceHash::Poseidon1, _) => unreachable!("Poseidon1 leaves launched above"),
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

    #[test]
    fn wide_plans_cover_every_level_in_passes_of_four_to_ten() {
        for l in MIN_LOG_N..=32 {
            let ks = plan_wide(l);
            assert_eq!(ks.iter().sum::<u32>(), l, "l = {l}");
            assert_eq!(ks.len() as u32, l.div_ceil(10), "l = {l}: pass count");
            assert!(
                ks.iter().all(|&k| (4..=WIDE_MAX_K).contains(&k)),
                "l = {l}: {ks:?}"
            );
            let (lo, hi) = (ks.iter().min().unwrap(), ks.iter().max().unwrap());
            assert!(hi - lo <= 1, "l = {l}: uneven {ks:?}");
            if ks.len() == plan(l).len() {
                assert_eq!(
                    ks,
                    plan(l),
                    "l = {l}: as many passes as today's, today's plan"
                );
            }
        }
        assert_eq!(plan_wide(28), [10, 9, 9], "the production encoding");
        assert_eq!(plan(28), [7, 7, 7, 7], "today's production encoding");
        // The sizes where the plans differ up to 2^30 (the host KAT pins 9, 10
        // and 17..20 against today's plan).
        let differ: Vec<u32> = (MIN_LOG_N..=30)
            .filter(|&l| plan_wide(l) != plan(l))
            .collect();
        assert_eq!(differ, [9, 10, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30]);
    }

    /// A first pass at the lane cap stages 2^12 positions in either plan, so it
    /// holds the low `12 − log_blowup` Möbius levels; below 2^12, all of them.
    #[test]
    fn the_first_pass_holds_the_low_mobius_levels() {
        for wide in [false, true] {
            with_ntt_wide(wide, || {
                for lb in 0..=MAX_LOG_BLOWUP {
                    for log_n in 12.max(lb + MIN_LOG_N)..=30 {
                        assert_eq!(fused_mobius_levels(log_n, lb), 12 - lb, "2^{log_n} lb {lb}");
                    }
                }
                assert_eq!(
                    fused_mobius_levels(11, 1),
                    10,
                    "every level of a 2^10 stack"
                );
                assert_eq!(fused_mobius_levels(28, 1), 11, "the production encoding");
            });
        }
    }

    /// The wide plan unless a test's override holds the 8-level one, and the
    /// override ends with its closure.
    #[test]
    fn the_wide_ntt_runs_unless_overridden() {
        assert!(ntt_wide_enabled(), "the default is the wide plan");
        assert!(with_ntt_wide(true, ntt_wide_enabled));
        assert!(!with_ntt_wide(false, ntt_wide_enabled));
        assert!(ntt_wide_enabled());
    }

    /// Every (s, k) window the driver launches: DIT ascending from s = 0, DIF
    /// descending to s = 0, and the encoding's wide DIT passes.
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
        let mut s = 0;
        for &k in &plan_wide(l) {
            out.push((s, k));
            s += k;
        }
        out
    }

    #[test]
    fn a_block_never_exceeds_256_threads_or_its_window() {
        for l in MIN_LOG_N..=32 {
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
                // The tile, and the static radix table: 2^(k-1) words, 128 at least.
                let smem = tile_bytes(k, log_t) + (1u32 << (k - 1).max(7)) * 8;
                assert!(
                    smem <= 48 * 1024,
                    "l={l} s={s} k={k}: {smem} B of shared memory"
                );
                if k <= 8 {
                    assert_eq!(
                        tile_bytes(k, log_t),
                        (1u32 << k) * ((1u32 << log_t) + 1) * 8,
                        "l={l} s={s} k={k}: today's tile"
                    );
                }
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
    fn spread_shapes_outside_the_engine_are_declined() {
        assert!(spread_supports(4, 0), "spacing 1 is a plain transform");
        assert!(spread_supports(24, 2), "the WHIR production shape");
        assert!(spread_supports(8, 4));
        assert!(!spread_supports(3, 0), "transform below 2^4");
        assert!(!spread_supports(33, 2), "beyond the two-adicity");
        // Spacings 32 and 64 where the first pass holds them, in both plans:
        // every encode of 2^20 positions or more, the WHIR blowup-64 encodes.
        for wide in [false, true] {
            with_ntt_wide(wide, || {
                for log_n in 20..=32 {
                    for lb in [5, 6] {
                        assert!(
                            spread_supports(log_n, lb),
                            "2^{log_n} at 2^{lb}, wide {wide}"
                        );
                    }
                }
                assert!(!spread_supports(30, 7), "spacing beyond the widened bound");
            });
        }
        // ... and not where a first pass is shorter: today's plan of 2^9 is
        // [5, 4], so spacing 64 stays on the legacy path and 32 does not.
        with_ntt_wide(false, || {
            assert_eq!(plan(9), vec![5, 4]);
            assert!(!spread_supports(9, 6), "2^9 at 2^6: a 5-level first pass");
            assert!(
                spread_supports(9, 5),
                "2^9 at 2^5: a 5-level first pass holds it"
            );
        });
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
