use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static HEAP_READER: OnceLock<fn() -> Option<usize>> = OnceLock::new();

pub fn set_heap_reader(f: fn() -> Option<usize>) {
    let _ = HEAP_READER.set(f);
}

pub fn heap_bytes() -> Option<usize> {
    HEAP_READER.get().and_then(|f| f())
}

pub type HeapSnapshot = (&'static str, usize);

pub fn snap(label: &'static str) -> Option<HeapSnapshot> {
    heap_bytes().map(|b| (label, b))
}

pub struct ProveHeapProfile {
    pub before: Option<usize>,
    pub after_execute: Option<usize>,
    pub after_trace_build: Option<usize>,
    pub after_air: Option<usize>,
}

/// Sub-operation timing breakdown for a single table in Rounds 2-4.
#[derive(Clone, Debug, Default)]
pub struct TableSubOps {
    /// evaluator.evaluate()
    pub constraints: Duration,
    /// decompose_and_extend_d2
    pub comp_decompose: Duration,
    /// commit_bit_reversed (composition-polynomial commit step)
    pub comp_commit: Duration,
    /// Round 3: barycentric OOD evaluation
    pub ood: Duration,
    /// Round 4: compute_deep_composition_poly_evaluations
    pub deep_comp: Duration,
    /// Round 4: interpolate_fft + evaluate_fft
    pub deep_extend: Duration,
    /// fri::commit_phase_from_evaluations
    pub fri_commit: Duration,
    /// Round 4: grinding + FRI query + Merkle openings
    pub queries: Duration,
}

/// Sub-operation breakdown for Round 1 aux commit pass.
#[derive(Clone, Debug, Default)]
pub struct Round1SubOps {
    /// Main trace: expand_columns_to_lde (LDE/FFT)
    pub main_lde: Duration,
    /// Main trace: commit_bit_reversed (Merkle)
    pub main_merkle: Duration,
    /// Aux trace: expand_columns_to_lde (LDE/FFT)
    pub aux_lde: Duration,
    /// Aux trace: commit_bit_reversed (Merkle)
    pub aux_merkle: Duration,
}

/// Timing data collected inside `multi_prove`.
pub struct MultiProveTiming {
    pub prepass: Duration,
    pub main_commits: Duration,
    pub aux_build: Duration,
    pub aux_commit: Duration,
    pub rounds_2_4: Duration,
    /// Sub-op breakdown for Round 1 (main + aux LDE vs Merkle).
    pub round1_sub: Round1SubOps,
    /// (name, rows, duration, sub_ops) per table for rounds 2-4.
    pub table_timings: Vec<(String, usize, Duration, TableSubOps)>,
    pub heap_snapshots: Vec<HeapSnapshot>,
}

/// Round 1 sub-timings: atomics so parallel rayon workers can accumulate safely.
static R1_MAIN_LDE_US: AtomicU64 = AtomicU64::new(0);
static R1_MAIN_MERKLE_US: AtomicU64 = AtomicU64::new(0);
static R1_AUX_LDE_US: AtomicU64 = AtomicU64::new(0);
static R1_AUX_MERKLE_US: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static TIMING_DATA: RefCell<Option<MultiProveTiming>> = const { RefCell::new(None) };
    /// Round 2 sub-timings: (constraints, fft, merkle)
    static R2_SUB: RefCell<Option<(Duration, Duration, Duration)>> = const { RefCell::new(None) };
    /// Round 4 sub-timings: (fft, merkle, deep_comp, queries)
    static R4_SUB: RefCell<Option<(Duration, Duration, Duration, Duration)>> = const { RefCell::new(None) };
    /// Assembled sub-ops from prove_rounds_2_to_4 (without reconstruct_round1 LDE time).
    static ROUND_SUB_OPS: RefCell<Option<TableSubOps>> = const { RefCell::new(None) };
}

pub fn store(data: MultiProveTiming) {
    TIMING_DATA.with(|cell| {
        *cell.borrow_mut() = Some(data);
    });
}

pub fn take() -> Option<MultiProveTiming> {
    TIMING_DATA.with(|cell| cell.borrow_mut().take())
}

pub fn accum_r1_main(lde: Duration, merkle: Duration) {
    R1_MAIN_LDE_US.fetch_add(lde.as_micros() as u64, Ordering::Relaxed);
    R1_MAIN_MERKLE_US.fetch_add(merkle.as_micros() as u64, Ordering::Relaxed);
}

pub fn accum_r1_aux(lde: Duration, merkle: Duration) {
    R1_AUX_LDE_US.fetch_add(lde.as_micros() as u64, Ordering::Relaxed);
    R1_AUX_MERKLE_US.fetch_add(merkle.as_micros() as u64, Ordering::Relaxed);
}

pub fn take_r1_sub() -> Round1SubOps {
    Round1SubOps {
        main_lde: Duration::from_micros(R1_MAIN_LDE_US.swap(0, Ordering::Relaxed)),
        main_merkle: Duration::from_micros(R1_MAIN_MERKLE_US.swap(0, Ordering::Relaxed)),
        aux_lde: Duration::from_micros(R1_AUX_LDE_US.swap(0, Ordering::Relaxed)),
        aux_merkle: Duration::from_micros(R1_AUX_MERKLE_US.swap(0, Ordering::Relaxed)),
    }
}

/// Reset all instrument state. Call at the start of `multi_prove` to avoid
/// stale data from a previous run in the same process.
///
/// Note: thread-local stores (R2_SUB, R4_SUB, ROUND_SUB_OPS) are only cleared
/// for the calling thread. Rayon worker threads are not reset — stale data is
/// possible if a previous run panicked without consuming stored values.
/// In practice this is safe because store/take pairs always execute within the
/// same rayon task closure.
pub fn reset_all() {
    R1_MAIN_LDE_US.store(0, Ordering::Relaxed);
    R1_MAIN_MERKLE_US.store(0, Ordering::Relaxed);
    R1_AUX_LDE_US.store(0, Ordering::Relaxed);
    R1_AUX_MERKLE_US.store(0, Ordering::Relaxed);
    TIMING_DATA.with(|cell| {
        cell.borrow_mut().take();
    });
    R2_SUB.with(|cell| {
        cell.borrow_mut().take();
    });
    R4_SUB.with(|cell| {
        cell.borrow_mut().take();
    });
    ROUND_SUB_OPS.with(|cell| {
        cell.borrow_mut().take();
    });
}

pub fn store_r2_sub(constraints: Duration, fft: Duration, merkle: Duration) {
    R2_SUB.with(|cell| *cell.borrow_mut() = Some((constraints, fft, merkle)));
}

pub fn take_r2_sub() -> Option<(Duration, Duration, Duration)> {
    R2_SUB.with(|cell| cell.borrow_mut().take())
}

pub fn store_r4_sub(fft: Duration, merkle: Duration, deep_comp: Duration, queries: Duration) {
    R4_SUB.with(|cell| *cell.borrow_mut() = Some((fft, merkle, deep_comp, queries)));
}

pub fn take_r4_sub() -> Option<(Duration, Duration, Duration, Duration)> {
    R4_SUB.with(|cell| cell.borrow_mut().take())
}

pub fn store_round_sub_ops(data: TableSubOps) {
    ROUND_SUB_OPS.with(|cell| {
        *cell.borrow_mut() = Some(data);
    });
}

pub fn take_round_sub_ops() -> Option<TableSubOps> {
    ROUND_SUB_OPS.with(|cell| cell.borrow_mut().take())
}
