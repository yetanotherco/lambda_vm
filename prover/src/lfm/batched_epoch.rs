//! Assembly — the BATCHED epoch's challenge replay, the M-8 spine.
//!
//! The batched counterpart of [`super::epoch`]: the same discipline — one cell
//! per value, every consumer reads it, challenges are the transcript's and
//! never the arena's — over a different walk. The order authority is
//! `stark::batched::verifier::replay_epoch_transcript`, which is itself pinned
//! to the prover's ENDING TRANSCRIPT STATE by
//! `replay_matches_the_provers_ending_state`; this module replays exactly that
//! sequence:
//!
//! - the SHAPE HISTOGRAM, before the first root (Recommendation S: the epoch
//!   commits to what it is before any challenge is drawn);
//! - every preprocessed table's root FROM THE AIR SET, per table in table
//!   order — the same [`super::epoch::RootCells`] + `PrepSource` provenance
//!   machinery the per-table program uses, verbatim; the DECODE cells feed the
//!   attestation join unchanged;
//! - `main_root`; the shared LogUp pair; `aux_root`; every table's `L`;
//! - ALL constraint-batching `β`s consecutively; `parts_root`;
//! - per table: `z` (drawn once and constrained outside both domains — the
//!   [`super::epoch::emit_z_ood`] disposition), both OOD blocks COLUMN-major,
//!   then the claimed parts;
//! - ALL DEEP `γ`s consecutively;
//! - round 4 (`derive_batched_fri_challenges`): the histogram A SECOND TIME,
//!   one shared DEEP-mix `α`, per committed layer `ζ` sampled THEN the root
//!   absorbed, the final `ζ` iff the codeword folds, the terminal
//!   coefficients, grinding, and ONE shared query-index set — `h_max − 1`
//!   bits per query in the TALLEST domain, which every shorter consumer
//!   REDUCES (`fri/mmcs.rs`'s index convention) rather than re-draws.
//!
//! **No forks, no index separators.** `fork_table` is dead on this path: the
//! whole epoch is one transcript, which is the wrap-side economy the campaign
//! is after — one path per round per query instead of one per table per round.

use crate::tables::types::FE;

use stark::config::Commitment;
use stark::fri::batched::{BatchedFriLayout, FriInstancePlan};

use super::builder::{Bit, Ext, Felt, LfmBuilder};
use super::epoch::RootCells;
use super::transcript_replay::TranscriptReplay;

/// The batched FRI's program shape: production's own layout and partition,
/// captured at emit time so the instance-class split UNROLLS into
/// straight-line code — there is deliberately no second in-machine derivation
/// of either.
///
/// `total_folds` comes from the batched class's `h_max`, the terminal length
/// and `effective_k` from its `h_min` ([`BatchedFriLayout::new`]'s floor);
/// standalone tables keep terminal-only instances and appear in
/// [`FriInstancePlan::standalone`].
#[derive(Clone, Debug)]
pub struct BatchedFriShape {
    pub layout: BatchedFriLayout,
    pub plan: FriInstancePlan,
}

impl BatchedFriShape {
    /// Derive from the epoch's LDE heights — the same call the host verifier
    /// makes, so the two cannot disagree about the partition.
    pub fn new(heights: &[usize], blowup_log: u32, final_poly_log_degree: u32) -> Self {
        let plan = FriInstancePlan::new(heights, blowup_log, final_poly_log_degree)
            .expect("the epoch's heights must partition");
        let layout =
            BatchedFriLayout::new(plan.h_max, plan.h_min, blowup_log, final_poly_log_degree);
        Self { layout, plan }
    }

    pub fn num_committed(&self) -> usize {
        self.layout.num_committed
    }

    pub fn num_terminal_coeffs(&self) -> usize {
        1usize << self.layout.effective_k
    }

    /// Bits one shared query index carries — `sample_u64(2^(h_max − 1))` in
    /// the TALLEST domain.
    pub fn index_bits(&self) -> usize {
        self.plan.h_max - 1
    }
}

/// One table's slice of the batched spine — every field program shape.
#[derive(Clone, Debug)]
pub struct BatchedTableShape {
    /// `log2` of the trace length; with the epoch blowup this is the table's
    /// LDE height, the `z`-guard's domain and the histogram's `h`.
    pub log2_trace_length: u32,
    /// Whether the table carries a bus contribution `L`.
    pub has_contribution: bool,
    /// `(width, height)` of the current-row OOD block.
    pub ood_current_dims: (usize, usize),
    /// `(width, height)` of the pruned next-row OOD block.
    pub ood_next_dims: (usize, usize),
    /// Composition-poly parts.
    pub num_parts: usize,
}

/// The whole batched epoch's spine shape.
#[derive(Clone, Debug)]
pub struct BatchedEpochShape {
    pub tables: Vec<BatchedTableShape>,
    /// `log2` LDE height per table, table order — the histogram's heights and
    /// the FRI's index space.
    pub heights: Vec<usize>,
    /// Total committed width per table (main + aux + parts columns), the
    /// histogram's widths — `EpochShape::total_widths`, precomputed host-side.
    pub total_widths: Vec<usize>,
    pub log2_blowup: u32,
    pub coset_offset: FE,
    /// Whether ANY table has a RAP — fixes the aux root's and the shared
    /// LogUp draw's presence together.
    pub has_aux: bool,
    pub fri: BatchedFriShape,
    pub grinding_factor: u8,
    pub num_queries: usize,
}

impl BatchedEpochShape {
    fn check(&self) {
        assert_eq!(self.tables.len(), self.heights.len());
        assert_eq!(self.tables.len(), self.total_widths.len());
        assert!(!self.tables.is_empty(), "an epoch has tables");
        for (t, h) in self.tables.iter().zip(&self.heights) {
            assert_eq!(
                t.log2_trace_length + self.log2_blowup,
                *h as u32,
                "a table's histogram height IS its LDE height"
            );
            assert!(t.num_parts > 0, "a composition polynomial has parts");
        }
        assert_eq!(
            self.has_aux,
            self.tables.iter().any(|t| t.has_contribution),
            "the aux round exists exactly when some table contributes"
        );
    }
}

/// A preprocessed root as the spine absorbs it — the same three provenances
/// as the per-table program's Phase A, with the same absorb economies: a
/// program-text root absorbs as literal bytes (no splice arithmetic), a
/// derived or hinted one as its cells.
pub enum BatchedPrepRoot<'a> {
    /// BITWISE / KECCAK_RC / PAGE zero-init: a function of the options alone.
    Constant(&'a Commitment),
    /// REGISTER (derived in-machine) or DECODE (hinted, attestation-joined).
    Cells(&'a RootCells),
}

/// The proof-carried cells the batched spine absorbs — the caller's cells,
/// hinted once and handed here, never re-hinted. This is the assembly join
/// surface: the same values go on to the constraint legs, the DEEP crossing,
/// the mixed walks and the LogUp closure.
pub struct BatchedEpochAbsorbs<'a> {
    /// Per table in table order: the preprocessed root, `Some` exactly when
    /// the AIR is preprocessed.
    pub prep_roots: &'a [Option<BatchedPrepRoot<'a>>],
    pub main_root: &'a RootCells,
    /// Present exactly when [`BatchedEpochShape::has_aux`].
    pub aux_root: Option<&'a RootCells>,
    /// Per table: the bus contribution `L`, `Some` exactly when the table's
    /// shape says so. The LogUp closure sums THESE cells.
    pub contributions: &'a [Option<Ext>],
    pub parts_root: &'a RootCells,
    /// Per table: the OOD data, row-major as the proof carries it.
    pub ood: &'a [BatchedTableOod<'a>],
    /// The batched instance's committed layer roots, fold order.
    pub fri_roots: &'a [RootCells],
    /// The batched instance's terminal coefficients, low-to-high.
    pub fri_coeffs: &'a [Ext],
    /// The grinding nonce, present exactly when `grinding_factor > 0`.
    pub nonce: Option<Felt>,
}

/// One table's OOD cells.
pub struct BatchedTableOod<'a> {
    pub current: &'a [Ext],
    pub next: &'a [Ext],
    pub parts: &'a [Ext],
}

/// The batched epoch's challenges, as cells.
pub struct BatchedEpochChallenges {
    /// The shared LogUp pair `(z, α)`.
    pub lookup: (Ext, Ext),
    /// One constraint-batching `β` per table, table order.
    pub betas: Vec<Ext>,
    /// One OOD point per table, table order.
    pub zs: Vec<Ext>,
    /// One DEEP `γ` per table, table order.
    pub gammas: Vec<Ext>,
    /// The shared DEEP-mix `α` — powers are assigned by `plan.batched`
    /// POSITION, not table index.
    pub alpha: Ext,
    /// `ζ₀ .. ζ_C` of the ONE batched instance.
    pub zetas: Vec<Ext>,
    /// Per query: the SHARED index bits, low-to-high, `h_max − 1` of them in
    /// the tallest domain. Every shorter round/table REDUCES by dropping low
    /// bits; nothing re-draws.
    pub iota_bits: Vec<Vec<Bit>>,
}

/// The canonical shape-histogram binding (`absorb_shape_histogram`), as ONE
/// constant byte run — every height and width is program shape. Production
/// absorbs it twice (the spine's head and round 4), and so does the machine.
pub fn emit_shape_histogram(t: &mut TranscriptReplay, heights: &[usize], widths: &[usize]) {
    assert_eq!(
        heights.len(),
        widths.len(),
        "the shape histogram needs one width per height"
    );
    let mut bytes = Vec::with_capacity(8 + 16 * heights.len());
    bytes.extend_from_slice(&(heights.len() as u64).to_le_bytes());
    for (h, w) in heights.iter().zip(widths) {
        bytes.extend_from_slice(&(*h as u64).to_le_bytes());
        bytes.extend_from_slice(&(*w as u64).to_le_bytes());
    }
    t.append_const_bytes(&bytes);
}

/// Replay the whole batched epoch transcript. `t` must be positioned right
/// after the statement absorb — there is no Phase A and no fork on this path.
pub fn emit_batched_epoch_challenges(
    b: &mut LfmBuilder,
    t: &mut TranscriptReplay,
    shape: &BatchedEpochShape,
    absorbs: &BatchedEpochAbsorbs<'_>,
) -> BatchedEpochChallenges {
    shape.check();
    let n = shape.tables.len();
    assert_eq!(absorbs.prep_roots.len(), n, "one prep slot per table");
    assert_eq!(
        absorbs.contributions.len(),
        n,
        "one contribution slot per table"
    );
    assert_eq!(absorbs.ood.len(), n, "one OOD bundle per table");
    assert_eq!(
        absorbs.aux_root.is_some(),
        shape.has_aux,
        "the aux root's presence is shape"
    );
    for (table, (t_shape, l)) in shape.tables.iter().zip(absorbs.contributions).enumerate() {
        assert_eq!(
            l.is_some(),
            t_shape.has_contribution,
            "table {table}: the contribution's presence is shape"
        );
    }
    for (table, (t_shape, ood)) in shape.tables.iter().zip(absorbs.ood).enumerate() {
        assert_eq!(
            ood.current.len(),
            t_shape.ood_current_dims.0 * t_shape.ood_current_dims.1,
            "table {table}: the current-row OOD block must match its dims"
        );
        assert_eq!(
            ood.next.len(),
            t_shape.ood_next_dims.0 * t_shape.ood_next_dims.1,
            "table {table}: the next-row OOD block must match its dims"
        );
        assert_eq!(
            ood.parts.len(),
            t_shape.num_parts,
            "table {table}: one cell per part"
        );
    }
    assert_eq!(
        absorbs.fri_roots.len(),
        shape.fri.num_committed(),
        "one root per committed layer"
    );
    assert_eq!(
        absorbs.fri_coeffs.len(),
        shape.fri.num_terminal_coeffs(),
        "the terminal coefficient count is shape"
    );
    assert_eq!(
        absorbs.nonce.is_some(),
        shape.grinding_factor > 0,
        "a nonce exists exactly when grinding is on"
    );

    // ---- Recommendation S: the histogram, before the first root.
    emit_shape_histogram(t, &shape.heights, &shape.total_widths);

    // ---- every preprocessed root, from the AIR set, table order.
    //
    // Misaligned appends, same as `replay_phase_a` and for the same reason:
    // the statement leaves the first segment's cursor at shift 3
    // (`statement_replay`'s module doc prices this), and the histogram —
    // 8 + 16·n bytes, ≡ 0 (mod 4) — does not move it. Every segment after
    // the first sample starts with the 32-byte reversed digest, so all the
    // downstream absorbs are aligned.
    for root in absorbs.prep_roots.iter().flatten() {
        match root {
            BatchedPrepRoot::Constant(bytes) => t.append_const_bytes(&bytes[..]),
            BatchedPrepRoot::Cells(cells) => t.append_halves_misaligned(&cells.halves()),
        }
    }
    t.append_halves_misaligned(&absorbs.main_root.halves());

    // ---- the shared LogUp pair.
    let lookup = (t.sample_ext(b), t.sample_ext(b));

    // ---- aux root, then every table's L.
    if let Some(root) = absorbs.aux_root {
        t.append_halves(&root.halves());
    }
    for l in absorbs.contributions.iter().flatten() {
        super::epoch::append_ext_cell(b, t, *l);
    }

    // ---- ALL betas, consecutively.
    let betas: Vec<Ext> = (0..n).map(|_| t.sample_ext(b)).collect();

    t.append_halves(&absorbs.parts_root.halves());

    // ---- per table: z, both OOD blocks COLUMN-major, parts.
    let mut zs = Vec::with_capacity(n);
    for (t_shape, ood) in shape.tables.iter().zip(absorbs.ood) {
        let z = t.sample_ext(b);
        super::epoch::assert_z_outside_domains_raw(
            b,
            z,
            t_shape.log2_trace_length,
            shape.log2_blowup,
            shape.coset_offset,
        );
        for (dims, block) in [
            (t_shape.ood_current_dims, ood.current),
            (t_shape.ood_next_dims, ood.next),
        ] {
            let (width, height) = dims;
            for col in 0..width {
                for row in 0..height {
                    super::epoch::append_ext_cell(b, t, block[row * width + col]);
                }
            }
        }
        for part in ood.parts {
            super::epoch::append_ext_cell(b, t, *part);
        }
        zs.push(z);
    }

    // ---- ALL gammas, consecutively.
    let gammas: Vec<Ext> = (0..n).map(|_| t.sample_ext(b)).collect();

    // ---- round 4: the histogram again, α, ζ-then-root, terminal, grinding,
    // and the ONE shared query-index set.
    emit_shape_histogram(t, &shape.heights, &shape.total_widths);
    let alpha = t.sample_ext(b);

    let mut zetas = Vec::with_capacity(shape.fri.num_committed() + 1);
    for root in absorbs.fri_roots {
        // Sample FIRST, absorb SECOND — a ζ drawn after its own layer root is
        // a challenge the prover answers rather than one that binds them.
        zetas.push(t.sample_ext(b));
        t.append_halves(&root.halves());
    }
    if shape.fri.layout.total_folds > 0 {
        zetas.push(t.sample_ext(b));
    }
    for c in absorbs.fri_coeffs {
        super::epoch::append_ext_cell(b, t, *c);
    }

    if let Some(nonce) = absorbs.nonce {
        let seed = t.state(b);
        let halves = super::epoch::nonce_halves(b, nonce);
        super::epoch::emit_grinding_check(b, seed, halves, shape.grinding_factor);
        t.append_halves(&halves);
    }

    let iota_bits = (0..shape.num_queries)
        .map(|_| t.sample_u64_pow2(b, shape.fri.index_bits()))
        .collect();

    BatchedEpochChallenges {
        lookup,
        betas,
        zs,
        gammas,
        alpha,
        zetas,
        iota_bits,
    }
}
