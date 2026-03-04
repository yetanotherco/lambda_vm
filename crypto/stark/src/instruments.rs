use std::cell::RefCell;
use std::time::Duration;

/// Sub-operation timing breakdown for a single table in Rounds 2-4.
#[derive(Clone, Debug, Default)]
pub struct TableSubOps {
    /// reconstruct_round1 (expand_pool_to_lde)
    pub trace_lde: Duration,
    /// evaluator.evaluate()
    pub constraints: Duration,
    /// decompose_and_extend_d2
    pub comp_decompose: Duration,
    /// commit_composition_polynomial
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
    /// Main trace: expand_pool_to_lde (LDE/FFT)
    pub main_lde: Duration,
    /// Main trace: commit_columns_bit_reversed (Merkle)
    pub main_merkle: Duration,
    /// Aux trace: expand_pool_to_lde (LDE/FFT)
    pub aux_lde: Duration,
    /// Aux trace: commit_columns_bit_reversed (Merkle)
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
}

thread_local! {
    static TIMING_DATA: RefCell<Option<MultiProveTiming>> = const { RefCell::new(None) };
    /// Round 1 sub-timings accumulated across the main-commit and aux-commit loops.
    static R1_SUB: RefCell<Round1SubOps> = const { RefCell::new(Round1SubOps {
        main_lde: Duration::ZERO, main_merkle: Duration::ZERO,
        aux_lde: Duration::ZERO, aux_merkle: Duration::ZERO,
    }) };
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
    R1_SUB.with(|cell| {
        let mut s = cell.borrow_mut();
        s.main_lde += lde;
        s.main_merkle += merkle;
    });
}

pub fn accum_r1_aux(lde: Duration, merkle: Duration) {
    R1_SUB.with(|cell| {
        let mut s = cell.borrow_mut();
        s.aux_lde += lde;
        s.aux_merkle += merkle;
    });
}

pub fn take_r1_sub() -> Round1SubOps {
    R1_SUB.with(|cell| {
        std::mem::replace(
            &mut *cell.borrow_mut(),
            Round1SubOps::default(),
        )
    })
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
