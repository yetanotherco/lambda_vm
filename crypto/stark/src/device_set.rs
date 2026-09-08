//! The device working set of one table, term by term — and the pure admission
//! predicate over it.
//!
//! Two consumers read this model and must agree: the per-table scheduler's VRAM
//! throttle (`prover::VramGate`, which bounds the SUM of the tables proved
//! concurrently and runs on every build, GPU or not) and the GPU dispatch
//! layer's admission (`gpu_lde::admit`, which asks whether ONE table fits the
//! card at all). Keeping the arithmetic here, free of any `cuda` gate, is what
//! lets both read one model instead of each carrying a copy — and is why the
//! LogUp aux build's set ([`aux_build_device_set`]) lives here too rather than
//! beside its dispatch: the throttle spends it through
//! [`fused_task_peak_bytes`] on every build, and the build's own admission
//! spends it on a `cuda` one.
//!
//! Every term mirrors an allocation in `math_cuda` after the in-place LDE
//! transpose of #956 (one LDE buffer, never two); `one_lde_buffer::vram_arm`
//! measures the commit's three big terms and the doc on
//! `DEFAULT_MEMPOOL_RELEASE_THRESHOLD_BYTES` records the numbers. The model
//! carries no blanket safety factor: the card's admission budget is 80% of
//! device memory, and the 20% it leaves is what the context, the module code
//! and the retained pool live in.

/// Bytes per Goldilocks element on device.
pub const BASE_BYTES: u64 = 8;

/// Bytes per ext3 element on device — three adjacent base columns.
pub const EXT3_BYTES: u64 = 3 * BASE_BYTES;

/// Bytes of one Merkle node. Every commitment hash the device dispatches on
/// emits a 32-byte digest — a four-felt Goldilocks digest is exactly 32
/// canonical bytes — so the node buffer costs the same under every hash.
pub const MERKLE_NODE_BYTES: u64 = 32;

/// Cap on the in-place transpose's device scratch, mirrored from
/// `math_cuda::lde::INPLACE_TRANSPOSE_SCRATCH_BYTES` (private there). The
/// admission wants a bound, not the block geometry.
pub const INPLACE_TRANSPOSE_SCRATCH_CAP_BYTES: u64 = 256 << 20;

/// `(2 · leaves − 1) · 32` for the row-pair tree over `lde_size` rows.
pub const fn full_tree_bytes(lde_size: u64) -> u64 {
    lde_size.saturating_sub(1).saturating_mul(MERKLE_NODE_BYTES)
}

/// Bytes of `cols` ext3 columns over `rows` rows.
pub const fn ext3_bytes(rows: u64, cols: u64) -> u64 {
    rows.saturating_mul(cols).saturating_mul(EXT3_BYTES)
}

/// Bytes of `cols` base columns over `rows` rows.
pub const fn base_bytes(rows: u64, cols: u64) -> u64 {
    rows.saturating_mul(cols).saturating_mul(BASE_BYTES)
}

/// The device working set one fused row-major commit allocates, term by term
/// (`math_cuda::lde::coset_lde_row_major_inner`): ONE LDE buffer, the optional
/// trace-domain snapshot, the full Merkle node buffer, and the small scratch
/// (coset weights plus the capped transpose scratch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitDeviceSet {
    /// `lde_size · base_cols · 8`: the row-major LDE, transposed in place.
    pub lde_bytes: u64,
    /// `n · base_cols · 8`: the pre-NTT column-major snapshot the LogUp
    /// fingerprint kernel reads in place (main commits only).
    pub snapshot_bytes: u64,
    /// `(2 · leaves − 1) · 32` with `leaves = lde_size / 2`: one full row-pair
    /// tree. The preprocessed split path builds two, sequentially on one
    /// stream — the precomputed tree is downloaded and freed before the
    /// multiplicity tree is allocated — so one is the peak there too.
    pub tree_bytes: u64,
    /// Coset weights (`n · 8`) plus the transpose scratch cap.
    pub scratch_bytes: u64,
}

impl CommitDeviceSet {
    pub const fn total(&self) -> u64 {
        self.lde_bytes
            .saturating_add(self.snapshot_bytes)
            .saturating_add(self.tree_bytes)
            .saturating_add(self.scratch_bytes)
    }
}

/// Size one fused commit's device set. `base_cols` counts BASE-FIELD columns:
/// `m` for a base table, `3m` for an ext3 one (the ext3 row-major layout is
/// three adjacent base columns per element). `snapshot` is whether the
/// trace-domain column-major snapshot is retained (the main commits do, the
/// aux commits do not).
pub fn commit_device_set(
    n: usize,
    base_cols: usize,
    blowup: usize,
    snapshot: bool,
) -> CommitDeviceSet {
    let n = n as u64;
    let cols = base_cols as u64;
    let lde = n.saturating_mul(blowup as u64);
    CommitDeviceSet {
        lde_bytes: base_bytes(lde, cols),
        snapshot_bytes: if snapshot { base_bytes(n, cols) } else { 0 },
        tree_bytes: full_tree_bytes(lde),
        scratch_bytes: n
            .saturating_mul(BASE_BYTES)
            .saturating_add(INPLACE_TRANSPOSE_SCRATCH_CAP_BYTES),
    }
}

/// The device working set of the LogUp aux build on the resident path
/// (`math_cuda::logup::logup_aux_resident`), term by term as the code
/// allocates it. `I` interactions over `n` trace rows, `out` term columns
/// (`num_out_cols`: committed + 1 virtual), aux columns = `out` (committed +
/// the accumulated one), every LogUp value ext3 (24 B):
///
/// - `main_bytes`: the column-major main upload, `cols · n · 8`, or 0 when the
///   R1 snapshot is read in place (`ResidentMain::Dev`).
/// - `fingerprint_bytes`: `I · n · 24`, alive from the fingerprint kernel to
///   the end of the build.
/// - `inverse_scratch_bytes`: the batch inverse's prefix, suffix and output
///   (`math_cuda::inverse::batch_inverse_ext3_dev`), each `I · n · 24`; prefix
///   and suffix die when it returns, the output survives as the reciprocals.
/// - `terms_bytes`: `out · n · 24`; `scan_bytes`: row sum, scan output and the
///   accumulated column, `3 · n · 24`; `aux_bytes`: the row-major aux buffer,
///   `out · n · 24`.
///
/// The peak is one of two phases: the inverse (`main + 4 · fingerprints`) or
/// the term/assemble phase (`main + 2 · fingerprints + terms + scan + aux`);
/// [`AuxBuildDeviceSet::total`] is the larger. Descriptor arrays and block
/// totals are kilobytes and not counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxBuildDeviceSet {
    pub main_bytes: u64,
    pub fingerprint_bytes: u64,
    pub inverse_scratch_bytes: u64,
    pub terms_bytes: u64,
    pub scan_bytes: u64,
    pub aux_bytes: u64,
}

impl AuxBuildDeviceSet {
    /// The inverse phase: main + fingerprints + prefix + suffix + reciprocals.
    pub const fn inverse_phase(&self) -> u64 {
        self.main_bytes
            .saturating_add(self.fingerprint_bytes)
            .saturating_add(self.inverse_scratch_bytes)
    }

    /// The term/assemble phase: main + fingerprints + reciprocals + terms +
    /// scan + aux.
    pub const fn term_phase(&self) -> u64 {
        self.main_bytes
            .saturating_add(self.fingerprint_bytes)
            .saturating_add(self.fingerprint_bytes)
            .saturating_add(self.terms_bytes)
            .saturating_add(self.scan_bytes)
            .saturating_add(self.aux_bytes)
    }

    /// The peak of the build: the larger phase.
    pub const fn total(&self) -> u64 {
        let inverse = self.inverse_phase();
        let term = self.term_phase();
        if inverse > term { inverse } else { term }
    }
}

/// Size the resident aux build's device set: `n` trace rows, `main_cols` main
/// columns, `num_interactions` bus interactions, `num_out_cols` term columns;
/// `main_resident` is whether the main trace is read from the R1 snapshot
/// (no upload).
pub fn aux_build_device_set(
    n: usize,
    main_cols: usize,
    num_interactions: usize,
    num_out_cols: usize,
    main_resident: bool,
) -> AuxBuildDeviceSet {
    let n = n as u64;
    let fingerprints = ext3_bytes(n, num_interactions as u64);
    AuxBuildDeviceSet {
        main_bytes: if main_resident {
            0
        } else {
            n.saturating_mul(main_cols as u64)
                .saturating_mul(BASE_BYTES)
        },
        fingerprint_bytes: fingerprints,
        inverse_scratch_bytes: fingerprints.saturating_mul(3),
        terms_bytes: ext3_bytes(n, num_out_cols as u64),
        scan_bytes: ext3_bytes(n, 3),
        aux_bytes: ext3_bytes(n, num_out_cols as u64),
    }
}

/// The shape the rounds-2–4 model takes: what the AIR and the domain fix
/// before any device work starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableShape {
    /// Trace rows (the interpolation domain).
    pub n: usize,
    pub blowup: usize,
    /// Base-field main columns, preprocessed ones included.
    pub main_cols: usize,
    /// Ext3 aux (LogUp) columns.
    pub aux_cols: usize,
    /// Composition-polynomial parts: `composition_poly_degree_bound(n) / n`.
    pub num_parts: usize,
    /// OOD evaluation points per trace column:
    /// `transition_offsets.len() · step_size`.
    pub num_eval_points: usize,
}

/// The device set of one table across rounds 2–4, term by term — what the
/// scheduler's throttle admits a table's fused task against. Everything the
/// main commit left resident stays counted (the LDE, its snapshot, its tree),
/// and each later round adds what it allocates on top:
///
/// - R1 aux: the aux LDE (`lde · aux · 24`), the resident aux trace the LogUp
///   build left behind (`n · (aux + 1) · 24`, one extra column for the running
///   sum), and the aux tree;
/// - R2: `H` (`lde · 24`), the parts (`num_parts · lde · 24`) — `H` is still
///   alive while they are decomposed — and the parts tree;
/// - R3: the inverted denominators on the trace domain (`k · n · 24`);
/// - R4: the inverted denominators on the LDE (`(1 + k) · lde · 24`), the DEEP
///   codeword (`lde · 24`), the FRI layer chain (geometric, bounded by one more
///   codeword) and its trees (bounded by one full tree).
///
/// Rounds do not overlap inside one table, so this is an upper bound on any
/// instant of the task, not a sum of the rounds' peaks; the counted R3/R4
/// transients are small next to the resident LDEs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableDeviceSet {
    pub main: CommitDeviceSet,
    pub aux_bytes: u64,
    pub composition_bytes: u64,
    pub deep_fri_bytes: u64,
}

impl TableDeviceSet {
    pub const fn total(&self) -> u64 {
        self.main
            .total()
            .saturating_add(self.aux_bytes)
            .saturating_add(self.composition_bytes)
            .saturating_add(self.deep_fri_bytes)
    }
}

/// Size one table's rounds-2–4 device set for `shape`.
pub fn table_device_set(shape: TableShape) -> TableDeviceSet {
    let TableShape {
        n,
        blowup,
        main_cols,
        aux_cols,
        num_parts,
        num_eval_points,
    } = shape;
    // The trace-domain snapshot has one consumer, the LogUp aux build, so
    // the main commit keeps it exactly when the table has an aux trace
    // (`IsAIR::has_aux_trace` is `trace_layout().1 != 0`, this field).
    let main = commit_device_set(n, main_cols, blowup, aux_cols != 0);
    let (n, k, aux, parts) = (
        n as u64,
        num_eval_points as u64,
        aux_cols as u64,
        num_parts as u64,
    );
    let lde = n.saturating_mul(blowup as u64);
    let aux_bytes = if aux == 0 {
        0
    } else {
        ext3_bytes(lde, aux)
            .saturating_add(ext3_bytes(n, aux + 1))
            .saturating_add(full_tree_bytes(lde))
    };
    let composition_bytes = if parts == 0 {
        0
    } else {
        ext3_bytes(lde, 1 + parts).saturating_add(full_tree_bytes(lde))
    };
    let deep_fri_bytes = ext3_bytes(n, k)
        .saturating_add(ext3_bytes(lde, 1 + k))
        .saturating_add(ext3_bytes(lde, 2))
        .saturating_add(full_tree_bytes(lde));
    TableDeviceSet {
        main,
        aux_bytes,
        composition_bytes,
        deep_fri_bytes,
    }
}

// This module is the SIZE model and nothing else: what a stage puts on the
// card, so the gate can decide whether it fits. It is deliberately not the
// prover's table walk. The walk is a scheduling policy, keyed on a weight that
// is kept for the order it produces rather than for any byte it names, and it
// lives with the scheduler in `prover::table_walk_weight`. Sorting the walk by
// this model instead cost a measured 5.9 GiB of host peak at the q=20 wrap.

/// What the admission predicate decided for one dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// No CUDA backend — no GPU, or cubins that would not load. The host path
    /// is the only one; `math_cuda::device::backend` already warned once. A
    /// GPU-less host is not the production pipeline, so this is not an abort.
    NoDevice,
    /// Below the launch-overhead floor: the host path is the faster one.
    BelowFloor { lde_size: usize, floor: usize },
    /// Fits the card's admission budget.
    Admitted { bytes: u64, budget: u64 },
    /// Does not fit the card even alone.
    OverBudget { bytes: u64, budget: u64 },
}

impl Admission {
    pub const fn is_admitted(&self) -> bool {
        matches!(self, Admission::Admitted { .. })
    }
}

/// The pure predicate, floor and budget supplied. The row floor is checked
/// first — a table below it never asks the device for anything, whatever its
/// width — then the bytes ceiling.
pub const fn admit_bytes(lde_size: usize, bytes: u64, floor: usize, budget: u64) -> Admission {
    if lde_size < floor {
        return Admission::BelowFloor { lde_size, floor };
    }
    if bytes > budget {
        return Admission::OverBudget { bytes, budget };
    }
    Admission::Admitted { bytes, budget }
}

/// The admission arithmetic, with the floor and the budget supplied: no
/// device, no backend, pure numbers.
#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    /// `detect_vram_budget_bytes` on a 32 GiB card: 80% of the total.
    const CARD_32_GIB_BUDGET: u64 = 32 * GIB / 5 * 4;
    /// The dispatch layer's row floor (`gpu_lde::DEFAULT_GPU_LDE_THRESHOLD`).
    const FLOOR: usize = 1 << 14;

    /// The synthetic over-budget table: 2^22 rows x 612 columns at blowup 2.
    /// Its LDE alone is 38.25 GiB; with the snapshot and the tree the commit's
    /// device set is 57.9 GiB against a 25.6 GiB budget.
    #[test]
    fn the_over_budget_shape_is_over_budget() {
        let n = 1usize << 22;
        let set = commit_device_set(n, 612, 2, true);
        assert_eq!(set.lde_bytes, (n as u64) * 2 * 612 * 8);
        assert_eq!(set.snapshot_bytes, (n as u64) * 612 * 8);
        assert_eq!(set.tree_bytes, ((n as u64) * 2 - 1) * 32);
        assert!(set.lde_bytes > 38 * GIB && set.lde_bytes < 39 * GIB);
        assert!(set.total() > 57 * GIB && set.total() < 58 * GIB);
        assert!(matches!(
            admit_bytes(n * 2, set.total(), FLOOR, CARD_32_GIB_BUDGET),
            Admission::OverBudget { .. }
        ));
    }

    /// LFM_HASH under RPO — 2^21 rows x (436 value + 13 preprocessed) columns —
    /// fits the card at blowup 2 with one LDE buffer and does not at blowup 4.
    #[test]
    fn lfm_hash_rpo_fits_at_blowup_2_and_not_at_4() {
        let n = 1usize << 21;
        let b2 = commit_device_set(n, 449, 2, true);
        assert!(b2.total() > 21 * GIB && b2.total() < 22 * GIB, "{b2:?}");
        assert!(admit_bytes(n * 2, b2.total(), FLOOR, CARD_32_GIB_BUDGET).is_admitted());
        let b4 = commit_device_set(n, 449, 4, true);
        assert!(b4.total() > 35 * GIB && b4.total() < 36 * GIB, "{b4:?}");
        assert!(matches!(
            admit_bytes(n * 4, b4.total(), FLOOR, CARD_32_GIB_BUDGET),
            Admission::OverBudget { .. }
        ));
    }

    /// The whole-table set of LFM_HASH under RPO at blowup 2 — 3 aux columns,
    /// two parts, two eval points — is ~23 GiB: it proves alone inside the
    /// budget, and nothing else proves beside it. The old throttle model
    /// (two LDE buffers plus 256 B per LDE row) put the same table at
    /// 28.3 GiB, over the budget it is actually under.
    #[test]
    fn lfm_hash_rpo_whole_table_set() {
        let shape = TableShape {
            n: 1 << 21,
            blowup: 2,
            main_cols: 449,
            aux_cols: 3,
            num_parts: 2,
            num_eval_points: 2,
        };
        let set = table_device_set(shape);
        assert!(set.total() > 23 * GIB && set.total() < 24 * GIB, "{set:?}");
        assert!(set.total() <= CARD_32_GIB_BUDGET);
        assert!(2 * set.total() > CARD_32_GIB_BUDGET);
        let old_model = (1u64 << 22) * (449 * 8 + 3 * 24) * 2 + (1u64 << 22) * 256;
        assert!(old_model > CARD_32_GIB_BUDGET);
    }

    /// One LFM_BALU chunk of 2^22 rows (14 main, 2 aux, 2 parts, 2 points) is
    /// ~4.9 GiB under this model — the R1 set is 1.8 GiB; the resident aux
    /// trace, `H`, the denominators, DEEP and FRI add the rest — so five prove
    /// concurrently inside the budget. The sizing behind
    /// `prover::lfm::chunking::BaluChunking`, whose test calls this model.
    #[test]
    fn a_balu_chunk_is_five_gib() {
        let set = table_device_set(TableShape {
            n: 1 << 22,
            blowup: 2,
            main_cols: 14,
            aux_cols: 2,
            num_parts: 2,
            num_eval_points: 2,
        });
        assert!(
            set.main.total() > GIB + GIB / 2 && set.main.total() < 2 * GIB,
            "{set:?}"
        );
        assert!(
            set.total() > 4 * GIB + GIB / 2 && set.total() < 5 * GIB + GIB / 2,
            "{set:?}"
        );
        assert!(5 * set.total() <= CARD_32_GIB_BUDGET);
        assert!(6 * set.total() > CARD_32_GIB_BUDGET);
    }

    /// A table without aux or parts (d=1 with no lookups) counts only what it
    /// allocates.
    #[test]
    fn absent_rounds_cost_nothing() {
        let set = table_device_set(TableShape {
            n: 1 << 16,
            blowup: 2,
            main_cols: 8,
            aux_cols: 0,
            num_parts: 0,
            num_eval_points: 1,
        });
        assert_eq!(set.aux_bytes, 0);
        assert_eq!(set.composition_bytes, 0);
        assert!(set.deep_fri_bytes > 0);
    }

    /// LFM_BLAKE3 at q=41 (2^19 rows, 3,076 main columns, 1,261 interactions,
    /// 631 term columns) — the shape the residency probe hit. Reading the R1
    /// snapshot in place, the build's inverse phase alone is ~59 GiB, more
    /// than twice the budget of a 32 GiB card; uploading the main trace
    /// instead adds its 12.02 GiB on top. The old code allocated it unchecked.
    #[test]
    fn the_blake3_wrap_aux_build_does_not_fit_a_32_gib_card() {
        let n = 1usize << 19;
        let set = aux_build_device_set(n, 3076, 1261, 631, true);
        let fp = (n as u64) * 1261 * 24;
        assert_eq!(set.main_bytes, 0);
        assert_eq!(set.fingerprint_bytes, fp);
        assert_eq!(set.inverse_scratch_bytes, 3 * fp);
        assert_eq!(set.inverse_phase(), 4 * fp);
        assert!(set.inverse_phase() > set.term_phase());
        assert_eq!(set.total(), set.inverse_phase());
        assert!(set.total() > 2 * CARD_32_GIB_BUDGET, "{}", set.total());
        assert!(matches!(
            admit_bytes(usize::MAX, set.total(), 0, CARD_32_GIB_BUDGET),
            Admission::OverBudget { .. }
        ));
        let uploaded = aux_build_device_set(n, 3076, 1261, 631, false);
        assert_eq!(uploaded.main_bytes, (n as u64) * 3076 * 8);
        assert_eq!(uploaded.total() - set.total(), uploaded.main_bytes);
    }

    /// The pinned wrap's hash chip (LFM_HASH under RPX: 2^20 rows, 329
    /// committed columns, 13 aux columns → 26 interactions, 13 term columns)
    /// builds its aux in under 3 GiB and is admitted on a 32 GiB card.
    #[test]
    fn the_rpx_hash_chip_aux_build_fits() {
        let set = aux_build_device_set(1 << 20, 329, 26, 13, true);
        assert!(set.total() < 3 * GIB, "{}", set.total());
        assert!(admit_bytes(usize::MAX, set.total(), 0, CARD_32_GIB_BUDGET).is_admitted());
    }

    /// With few interactions per term column the term/assemble phase is the
    /// larger one and `total` must report it.
    #[test]
    fn the_peak_is_whichever_phase_is_larger() {
        // 2 interactions, 8 term columns: inverse = 4·fp = 8·n·24; term =
        // 2·fp + terms + scan + aux = (4 + 8 + 3 + 8)·n·24.
        let set = aux_build_device_set(1 << 10, 4, 2, 8, true);
        assert!(set.term_phase() > set.inverse_phase());
        assert_eq!(set.total(), set.term_phase());
        assert_eq!(set.term_phase(), (1u64 << 10) * 23 * 24);
    }

    /// The aux commit has no snapshot; its ext3 columns count as three base
    /// columns each.
    #[test]
    fn aux_sets_have_no_snapshot() {
        let set = commit_device_set(1 << 20, 3 * 3, 2, false);
        assert_eq!(set.snapshot_bytes, 0);
        assert_eq!(set.lde_bytes, ext3_bytes(1 << 21, 3));
    }

    /// The row floor is checked before the ceiling: a tiny table with an
    /// absurd byte count is "too small", never "over budget" — it will not ask
    /// the device for anything.
    #[test]
    fn the_floor_is_checked_before_the_budget() {
        assert!(matches!(
            admit_bytes(1 << 13, u64::MAX, FLOOR, CARD_32_GIB_BUDGET),
            Admission::BelowFloor { .. }
        ));
        assert!(matches!(
            admit_bytes(FLOOR, u64::MAX, FLOOR, CARD_32_GIB_BUDGET),
            Admission::OverBudget { .. }
        ));
    }

    /// FRI re-derives admission at width 1: a narrow transient over a large
    /// domain always clears the ceiling. The floor stays a row count, so it
    /// does not degenerate there either.
    #[test]
    fn fri_at_width_one_never_degenerates() {
        let n0 = 1usize << 24;
        let bytes = ext3_bytes(n0 as u64, 1) + full_tree_bytes(n0 as u64);
        assert!(admit_bytes(n0, bytes, FLOOR, CARD_32_GIB_BUDGET).is_admitted());
    }

    /// A budget of `u64::MAX` (query failed) makes the ceiling inert — the
    /// floor alone decides, which is the pre-admission behaviour.
    #[test]
    fn an_unbounded_budget_is_inert() {
        assert!(admit_bytes(1 << 20, u64::MAX - 1, FLOOR, u64::MAX).is_admitted());
    }

    /// The table's committed width is what the model takes: the row floor is
    /// width-blind on purpose, the ceiling is not.
    #[test]
    fn width_moves_the_ceiling_not_the_floor() {
        let n = 1usize << 21;
        let narrow = commit_device_set(n, 4, 2, true);
        let wide = commit_device_set(n, 612, 2, true);
        assert!(admit_bytes(n * 2, narrow.total(), FLOOR, CARD_32_GIB_BUDGET).is_admitted());
        assert!(matches!(
            admit_bytes(n * 2, wide.total(), FLOOR, CARD_32_GIB_BUDGET),
            Admission::OverBudget { .. }
        ));
    }
}
