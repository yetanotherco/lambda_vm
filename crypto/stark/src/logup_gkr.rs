//! LogUp-GKR adapter: the glue between a table's bus interactions and the
//! batch GKR protocol in [`crate::gkr`].
//!
//! In [`crate::lookup::LogUpMode::Gkr`] the per-table LogUp sums are proven by
//! a batch GKR over fraction-summation trees instead of committed term/acc
//! columns. Each interacting table commits exactly two auxiliary columns:
//!
//! - column [`GKR_AUX_KERNEL_COL`]: the Lagrange kernel `l[i] = eq(bits(i), r)`
//!   at the GKR random point `r`, bound by the boundary constraint
//!   `l[0] = ∏(1 − r_j)` and the `γ^K·l²` self-check folded into the bridge.
//! - column [`GKR_AUX_SIGMA_COL`]: the bridge running sum `σ`, whose circular
//!   transition constraint telescopes to `⟨l, col_j⟩ = c_j` for every main
//!   column `j` referenced by an interaction (Schwartz-Zippel over `γ`).
//!
//! This module holds the prover-side leaf/tree/claim computation, the shared
//! challenge-vector layout, and the verifier-side claim reconstruction. The
//! constraint *emission* for the bridge lives in [`crate::lookup`], beside the
//! standard LogUp emission, so both modes share the single-source discipline.

use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::gkr::{DeepLayerOracle, GkrInstance, Layer, gen_layers};
use crate::lagrange_kernel::compute_lagrange_kernel;
use crate::lookup::{BusInteraction, LOGUP_CHALLENGE_ALPHA, PackingShifts, compute_alpha_powers};
use crate::table::Table;
use crate::trace::TraceTable;

// =============================================================================
// Challenge-vector layout (GKR mode)
// =============================================================================
// The per-table rap_challenges vector in GKR mode is the shared `[z, α]` prefix
// extended with the bridge parameters:
//   [0] = z, [1] = α                        (shared, sampled in Phase B)
//   [2] = γ                                 (shared, sampled after column claims)
//   [3] = bridge offset Δ = target/N        (derived, per table)
//   [4 .. 4+K+1] = γ⁰, γ¹, …, γᴷ           (derived; γᴷ backs the l² self-check)
//   [4+K+1 ..]   = r₀, …, r_{n−1}           (GKR random point, per table length)
// K = number of distinct main columns referenced by the table's interactions.

/// Index of the `z` challenge in the LogUp challenges vector.
pub const LOGUP_CHALLENGE_Z: usize = 0;

/// Index of the `γ` challenge (column-claim batching) in the per-table
/// rap_challenges vector. Sampled on the main transcript after every table's
/// column claims are absorbed (binding them before γ exists).
pub const LOGUP_CHALLENGE_GAMMA: usize = 2;

/// Index of the bridge offset `Δ = target/N` in the per-table rap_challenges
/// vector. A derived value, not a random challenge.
pub const LOGUP_BRIDGE_OFFSET_IDX: usize = 3;

/// Start index of the precomputed γ powers in the per-table rap_challenges
/// vector: `rap_challenges[LOGUP_GAMMA_POWERS_START + j] = γ^j` for
/// `j = 0..=K`. The extra `γ^K` power backs the Lagrange-kernel `l²`
/// self-check.
pub const LOGUP_GAMMA_POWERS_START: usize = 4;

/// Auxiliary column index of the Lagrange kernel `l`.
pub const GKR_AUX_KERNEL_COL: usize = 0;

/// Auxiliary column index of the bridge running sum `σ`.
pub const GKR_AUX_SIGMA_COL: usize = 1;

/// Number of auxiliary columns an interacting table commits in GKR mode.
pub const GKR_NUM_AUX_COLUMNS: usize = 2;

/// Start index of the GKR random-point coordinates in the per-table
/// rap_challenges vector, given the table's distinct-column count `K`.
pub const fn logup_random_point_start(num_columns: usize) -> usize {
    // +1 for the extra γ^K power (the l² self-check).
    LOGUP_GAMMA_POWERS_START + num_columns + 1
}

// =============================================================================
// Input-layer geometry (the linear input layer — see
// thoughts/logup-gkr/input-layer-design.md)
// =============================================================================
// Each table is ONE GKR instance over K̂·N leaves indexed `i·K̂ + k`
// (interaction bits LOW): leaf (i, k) = (±m_k(i), fp_k(i)) for k < K, the
// fraction identity (0, 1) for padding. Leaves are LINEAR in the trace
// columns, so the verifier reconstructs the input-layer claims exactly from
// the column claims (no fail-open branch). By pair-level associativity of
// fraction addition, the tree's layer at size N equals the cross-multiplied
// per-row fractions bit-for-bit, so every layer from N up — and the root —
// is unchanged.

/// Number of padded interaction variables for a table: `log2(K̂)` where
/// `K̂ = K.next_power_of_two()`. Zero for single-interaction tables (the
/// instance degenerates to the per-row tree).
pub fn gkr_input_num_vars(num_interactions: usize) -> usize {
    debug_assert!(num_interactions > 0);
    num_interactions.next_power_of_two().trailing_zeros() as usize
}

/// Split an instance's full evaluation point into `(κ, ρ)`: the low
/// `input_num_vars` coordinates κ index the interaction bits (they weight the
/// verifier's claim reconstruction), the remaining coordinates ρ index the
/// rows (they are THE random point for the column claims and the
/// kernel/bridge — same length `log2(N)` as a row-only point).
pub fn split_input_point<E: IsField>(
    point: &[FieldElement<E>],
    input_num_vars: usize,
) -> (&[FieldElement<E>], &[FieldElement<E>]) {
    point.split_at(input_num_vars)
}

// =============================================================================
// Column extraction
// =============================================================================

/// Extract the sorted distinct main-column indices referenced by any
/// interaction (bus values and multiplicities). This canonical order is shared
/// by the prover's `column_claims`, the bridge constraint's batched sum, and
/// the verifier's claim checks.
pub fn extract_column_indices(interactions: &[BusInteraction]) -> Vec<usize> {
    let mut cols = Vec::new();
    for inter in interactions {
        for val in &inter.values {
            cols.extend(val.column_indices());
        }
        inter.multiplicity.collect_columns(&mut cols);
    }
    cols.sort_unstable();
    cols.dedup();
    cols
}

// =============================================================================
// Leaf fractions and layer trees (prover side)
// =============================================================================

/// The fingerprint `z − (bus_id + Σ αⁱ·elementᵢ)` for one interaction at one
/// row (a borrowed row-major slice). `α⁰ = 1`: the bus-id term is embedded
/// directly, no multiply.
fn compute_fingerprint_at_row<F, E>(
    interaction: &BusInteraction,
    row: &[FieldElement<F>],
    z: &FieldElement<E>,
    alpha_powers: &[FieldElement<E>],
    shifts: &PackingShifts<F>,
) -> FieldElement<E>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let mut lc = FieldElement::<E>::from(interaction.bus_id);
    let mut alpha_offset = 1;
    for bv in &interaction.values {
        alpha_offset += bv.accumulate_fingerprint_row(
            |col| &row[col],
            alpha_powers,
            alpha_offset,
            &mut lc,
            shifts,
        );
    }
    z - &lc
}

/// Computes the leaf fractions of the GKR summation tree from a table's bus
/// interactions and main trace.
///
/// For each row `i`, all K interactions fold into a single fraction
/// `N(i)/D(i)` by cross-multiplication, starting from `0/1`:
///
/// - `n' = n·fp_k ± m_k·d` (`+` for senders, `−` for receivers)
/// - `d' = d·fp_k`
///
/// so `D(i) = Π_k fp_k(i)` and `N(i) = Σ_k ±m_k(i)·Π_{j≠k} fp_j(i)`, i.e. the
/// row's total LogUp contribution `Σ_k ±m_k/fp_k` as one fraction.
///
/// Returns `(numerators, denominators)`, each of length `trace_len`.
pub fn compute_logup_leaf_fractions<F, E>(
    interactions: &[BusInteraction],
    main: &Table<F>,
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> (Vec<FieldElement<E>>, Vec<FieldElement<E>>)
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    assert!(
        !interactions.is_empty(),
        "leaf fractions require at least one interaction"
    );

    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let max_bus_elements = interactions
        .iter()
        .map(|inter| inter.num_bus_elements())
        .max()
        .unwrap();
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);
    let shifts = PackingShifts::<F>::new();

    // Rows are read in place from the row-major main table (borrowed slices,
    // no column-major transpose/clone of the segment).
    let leaf_at_row = |row_idx: usize| {
        let row = main.get_row(row_idx);
        let mut running_n = FieldElement::<E>::zero();
        let mut running_d = FieldElement::<E>::one();
        for inter in interactions {
            let fp = compute_fingerprint_at_row(inter, row, z, &alpha_powers, &shifts);
            let m = inter.multiplicity.evaluate_from(|col| row[col].clone());
            // m·d is F×E (base operand LEFT); the sign resolves as add vs neg.
            let cross = &m * &running_d;
            let cross = if inter.is_sender { cross } else { -cross };
            running_n = &running_n * &fp + cross;
            running_d = &running_d * &fp;
        }
        (running_n, running_d)
    };

    #[cfg(feature = "parallel")]
    let (numerators, denominators): (Vec<_>, Vec<_>) =
        (0..trace_len).into_par_iter().map(leaf_at_row).unzip();
    #[cfg(not(feature = "parallel"))]
    let (numerators, denominators): (Vec<_>, Vec<_>) = (0..trace_len).map(leaf_at_row).unzip();

    (numerators, denominators)
}

/// Build the LINEAR input layer: `K̂·N` leaves indexed `i·K̂ + k`, leaf
/// `(i, k) = (sign_k·m_k(i), fp_k(i))` for `k < K` and the fraction identity
/// `(0, 1)` for padding. Row-parallel; rows are read in place from the
/// row-major main table.
fn compute_gkr_input_layer<F, E>(
    interactions: &[BusInteraction],
    main: &Table<F>,
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Layer<E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    assert!(
        !interactions.is_empty(),
        "the input layer requires at least one interaction"
    );

    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let max_bus_elements = interactions
        .iter()
        .map(|inter| inter.num_bus_elements())
        .max()
        .unwrap();
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);
    let shifts = PackingShifts::<F>::new();
    let k_hat = interactions.len().next_power_of_two();

    let row_leaves = |row_idx: usize| -> Vec<(FieldElement<E>, FieldElement<E>)> {
        let row = main.get_row(row_idx);
        let mut leaves = Vec::with_capacity(k_hat);
        for inter in interactions {
            let fp = compute_fingerprint_at_row(inter, row, z, &alpha_powers, &shifts);
            let m = inter.multiplicity.evaluate_from(|col| row[col].clone());
            let n = if inter.is_sender {
                m.to_extension()
            } else {
                (-m).to_extension()
            };
            leaves.push((n, fp));
        }
        // Padding: the fraction identity 0/1 (contributes nothing to any sum).
        leaves.resize_with(k_hat, || {
            (FieldElement::<E>::zero(), FieldElement::<E>::one())
        });
        leaves
    };

    #[cfg(feature = "parallel")]
    let per_row: Vec<Vec<(FieldElement<E>, FieldElement<E>)>> =
        (0..trace_len).into_par_iter().map(row_leaves).collect();
    #[cfg(not(feature = "parallel"))]
    let per_row: Vec<Vec<(FieldElement<E>, FieldElement<E>)>> =
        (0..trace_len).map(row_leaves).collect();

    let mut numerators = Vec::with_capacity(k_hat * trace_len);
    let mut denominators = Vec::with_capacity(k_hat * trace_len);
    for row in per_row {
        for (n, d) in row {
            numerators.push(n);
            denominators.push(d);
        }
    }

    Layer::LogUpGeneric {
        numerators,
        denominators,
    }
}

/// The K̂ leaf pairs of one row: `(sign_k·m_k, fp_k)` for `k < K`, the
/// fraction identity `(0, 1)` for padding. Shared by the Stage-1 materialized
/// input layer and the deep-layer oracle.
fn row_leaf_pairs<F, E>(
    interactions: &[BusInteraction],
    row: &[FieldElement<F>],
    z: &FieldElement<E>,
    alpha_powers: &[FieldElement<E>],
    shifts: &PackingShifts<F>,
    k_hat: usize,
) -> Vec<(FieldElement<E>, FieldElement<E>)>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let mut leaves = Vec::with_capacity(k_hat);
    for inter in interactions {
        let fp = compute_fingerprint_at_row(inter, row, z, alpha_powers, shifts);
        let m = inter.multiplicity.evaluate_from(|col| row[col].clone());
        let n = if inter.is_sender {
            m.to_extension()
        } else {
            (-m).to_extension()
        };
        leaves.push((n, fp));
    }
    leaves.resize_with(k_hat, || {
        (FieldElement::<E>::zero(), FieldElement::<E>::one())
    });
    leaves
}

/// [`DeepLayerOracle`] over a table's interactions and (borrowed) row-major
/// main trace: materializes one deep layer's four split tables at a time by
/// streaming rows — the K̂·N input layer is never resident.
struct LogUpDeepOracle<'a, F: IsField, E: IsField> {
    interactions: &'a [BusInteraction],
    main: &'a Table<F>,
    z: FieldElement<E>,
    alpha_powers: Vec<FieldElement<E>>,
    shifts: PackingShifts<F>,
    k_hat: usize,
    trace_len: usize,
    /// Row-major `[row·K + k]` cache of the K per-interaction fingerprints —
    /// the one expensive leaf component (numerators are 1–2 column reads,
    /// recomputed on the fly). Built once on first deep round, freed with the
    /// instance; K·N ext elements (the padded K̂−K tail is the constant 1 and
    /// never stored).
    fp_cache: std::sync::OnceLock<Vec<FieldElement<E>>>,
}

impl<F, E> LogUpDeepOracle<'_, F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    /// The fingerprint cache, built once (parallel) on first use.
    fn fps(&self) -> &[FieldElement<E>] {
        self.fp_cache.get_or_init(|| {
            let k = self.interactions.len();
            let fill_row = |row_idx: usize, out: &mut [FieldElement<E>]| {
                let row = self.main.get_row(row_idx);
                for (slot, inter) in out.iter_mut().zip(self.interactions) {
                    *slot = compute_fingerprint_at_row(
                        inter,
                        row,
                        &self.z,
                        &self.alpha_powers,
                        &self.shifts,
                    );
                }
            };
            let mut cache = vec![FieldElement::<E>::zero(); k * self.trace_len];
            #[cfg(feature = "parallel")]
            cache
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(row_idx, out)| fill_row(row_idx, out));
            #[cfg(not(feature = "parallel"))]
            for (row_idx, out) in cache.chunks_mut(k).enumerate() {
                fill_row(row_idx, out);
            }
            cache
        })
    }

    /// One row's four split arrays at child level `c`, folded by this layer's
    /// bound challenges, written into `scratch` (reused across a thread's
    /// rows): build the K leaf pairs from the fingerprint cache (numerators
    /// recomputed — 1–2 column reads each), fold `c` levels of balanced
    /// fraction addition padding-aware (`x + (0,1) = x`, and slots entirely
    /// in the padded tail are `(0,1)` with no work), split by slot parity,
    /// then fold each array `bound.len()` times with the sumcheck rule
    /// `v' = l + ch·(r − l)` — exactly what `fold_table` does per row.
    fn folded_row_arrays_into(
        &self,
        c: usize,
        bound: &[FieldElement<E>],
        row_idx: usize,
        scratch: &mut RowScratch<E>,
    ) {
        let k = self.interactions.len();
        let fps = &self.fps()[row_idx * k..(row_idx + 1) * k];
        let row = self.main.get_row(row_idx);

        let leaves = &mut scratch.leaves;
        leaves.clear();
        for (inter, fp) in self.interactions.iter().zip(fps) {
            let m = inter.multiplicity.evaluate_from(|col| row[col].clone());
            let n = if inter.is_sender {
                m.to_extension()
            } else {
                (-m).to_extension()
            };
            leaves.push((n, fp.clone()));
        }

        // Fold c levels, padding-aware: the tail beyond `active` is the
        // fraction identity (0, 1); an odd trailing real entry passes through
        // unchanged (x + identity = x).
        let mut active = k;
        for _ in 0..c {
            let m = active / 2;
            for t in 0..m {
                let (n0, d0) = leaves[2 * t].clone();
                let (n1, d1) = leaves[2 * t + 1].clone();
                leaves[t] = (&(&n0 * &d1) + &(&n1 * &d0), &d0 * &d1);
            }
            if active % 2 == 1 {
                leaves[m] = leaves[active - 1].clone();
            }
            active = active.div_ceil(2);
            leaves.truncate(active);
        }
        // Pad the slot array to the full K̂/2^c width with identities.
        let slots = self.k_hat >> c;
        leaves.resize_with(slots, || {
            (FieldElement::<E>::zero(), FieldElement::<E>::one())
        });

        let half = slots / 2;
        let (nl, nr, dl, dr) = (
            &mut scratch.nl,
            &mut scratch.nr,
            &mut scratch.dl,
            &mut scratch.dr,
        );
        nl.clear();
        nr.clear();
        dl.clear();
        dr.clear();
        for t in 0..half {
            let (n_even, d_even) = leaves[2 * t].clone();
            let (n_odd, d_odd) = leaves[2 * t + 1].clone();
            nl.push(n_even);
            dl.push(d_even);
            nr.push(n_odd);
            dr.push(d_odd);
        }

        for ch in bound {
            for table in [&mut *nl, &mut *nr, &mut *dl, &mut *dr] {
                let m = table.len() / 2;
                for j in 0..m {
                    let left = table[2 * j].clone();
                    let right = table[2 * j + 1].clone();
                    table[j] = &left + &(ch * &(&right - &left));
                }
                table.truncate(m);
            }
        }
    }
}

/// Per-thread scratch for the streamed deep rounds (reused across rows).
struct RowScratch<E: IsField> {
    leaves: Vec<(FieldElement<E>, FieldElement<E>)>,
    nl: Vec<FieldElement<E>>,
    nr: Vec<FieldElement<E>>,
    dl: Vec<FieldElement<E>>,
    dr: Vec<FieldElement<E>>,
}

impl<E: IsField> RowScratch<E> {
    fn new(k_hat: usize) -> Self {
        Self {
            leaves: Vec::with_capacity(k_hat),
            nl: Vec::with_capacity(k_hat / 2),
            nr: Vec::with_capacity(k_hat / 2),
            dl: Vec::with_capacity(k_hat / 2),
            dr: Vec::with_capacity(k_hat / 2),
        }
    }
}

impl<F, E> DeepLayerOracle<E> for LogUpDeepOracle<'_, F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn k_hat(&self) -> usize {
        self.k_hat
    }

    fn num_rows(&self) -> usize {
        self.trace_len
    }

    #[allow(clippy::type_complexity)]
    fn materialize_split(
        &self,
        c: usize,
    ) -> (
        Vec<FieldElement<E>>,
        Vec<FieldElement<E>>,
        Vec<FieldElement<E>>,
        Vec<FieldElement<E>>,
    ) {
        debug_assert!(c < self.k_hat.trailing_zeros() as usize);
        // Per-row slots at child level c, and per-row entries per split table.
        let slots = self.k_hat >> c;
        let half = slots / 2;
        let out_len = self.trace_len * half;

        let mut nl = vec![FieldElement::<E>::zero(); out_len];
        let mut nr = vec![FieldElement::<E>::zero(); out_len];
        let mut dl = vec![FieldElement::<E>::zero(); out_len];
        let mut dr = vec![FieldElement::<E>::zero(); out_len];

        // Each row owns the contiguous span [row·half, (row+1)·half) in every
        // table: zip per-row chunks so the fill is embarrassingly parallel.
        let fill_row = |row_idx: usize,
                        nl_c: &mut [FieldElement<E>],
                        nr_c: &mut [FieldElement<E>],
                        dl_c: &mut [FieldElement<E>],
                        dr_c: &mut [FieldElement<E>]| {
            let row = self.main.get_row(row_idx);
            let mut leaves = row_leaf_pairs(
                self.interactions,
                row,
                &self.z,
                &self.alpha_powers,
                &self.shifts,
                self.k_hat,
            );
            // Fold c levels of pairwise fraction addition (balanced, matching
            // the extended tree's deep layers bit-for-bit).
            for _ in 0..c {
                let m = leaves.len() / 2;
                for t in 0..m {
                    let (n0, d0) = leaves[2 * t].clone();
                    let (n1, d1) = leaves[2 * t + 1].clone();
                    leaves[t] = (&(&n0 * &d1) + &(&n1 * &d0), &d0 * &d1);
                }
                leaves.truncate(m);
            }
            for t in 0..half {
                let (n_even, d_even) = leaves[2 * t].clone();
                let (n_odd, d_odd) = leaves[2 * t + 1].clone();
                nl_c[t] = n_even;
                dl_c[t] = d_even;
                nr_c[t] = n_odd;
                dr_c[t] = d_odd;
            }
        };

        #[cfg(feature = "parallel")]
        nl.par_chunks_mut(half)
            .zip(nr.par_chunks_mut(half))
            .zip(dl.par_chunks_mut(half))
            .zip(dr.par_chunks_mut(half))
            .enumerate()
            .for_each(|(row_idx, (((nl_c, nr_c), dl_c), dr_c))| {
                fill_row(row_idx, nl_c, nr_c, dl_c, dr_c)
            });
        #[cfg(not(feature = "parallel"))]
        for row_idx in 0..self.trace_len {
            let span = row_idx * half..(row_idx + 1) * half;
            let (nl_c, nr_c, dl_c, dr_c) = (
                &mut nl[span.clone()],
                &mut nr[span.clone()],
                &mut dl[span.clone()],
                &mut dr[span.clone()],
            );
            // Split borrows: fall back to per-row temporary buffers.
            let mut nl_t = nl_c.to_vec();
            let mut nr_t = nr_c.to_vec();
            let mut dl_t = dl_c.to_vec();
            let mut dr_t = dr_c.to_vec();
            fill_row(row_idx, &mut nl_t, &mut nr_t, &mut dl_t, &mut dr_t);
            nl[span.clone()].clone_from_slice(&nl_t);
            nr[span.clone()].clone_from_slice(&nr_t);
            dl[span.clone()].clone_from_slice(&dl_t);
            dr[span].clone_from_slice(&dr_t);
        }

        (nl, nr, dl, dr)
    }

    fn deep_gate_sums(
        &self,
        c: usize,
        bound: &[FieldElement<E>],
        eq_k: &[FieldElement<E>],
        eq_row: &[FieldElement<E>],
        lambda: &FieldElement<E>,
    ) -> (FieldElement<E>, FieldElement<E>) {
        let row_sums =
            |scratch: &mut RowScratch<E>, row_idx: usize| -> (FieldElement<E>, FieldElement<E>) {
                self.folded_row_arrays_into(c, bound, row_idx, scratch);
                let (fnl, fnr, fdl, fdr) = (&scratch.nl, &scratch.nr, &scratch.dl, &scratch.dr);
                let pairs = fnl.len() / 2;
                debug_assert_eq!(eq_k.len(), pairs);
                let mut h0 = FieldElement::<E>::zero();
                let mut h2 = FieldElement::<E>::zero();
                for j in 0..pairs {
                    // EXACTLY gate_generic's formulas (crypto/stark/src/gkr.rs):
                    // g0 on the pair's left entries, g2 on the 2·r − l
                    // extrapolations. The stage-2 differential test pins parity.
                    let (nl_l, nl_r) = (&fnl[2 * j], &fnl[2 * j + 1]);
                    let (nr_l, nr_r) = (&fnr[2 * j], &fnr[2 * j + 1]);
                    let (dl_l, dl_r) = (&fdl[2 * j], &fdl[2 * j + 1]);
                    let (dr_l, dr_r) = (&fdr[2 * j], &fdr[2 * j + 1]);

                    let gate_0 = &(nl_l * dr_l) + &(dl_l * &(nr_l + &(lambda * dr_l)));
                    let nl_2 = &(nl_r + nl_r) - nl_l;
                    let nr_2 = &(nr_r + nr_r) - nr_l;
                    let dl_2 = &(dl_r + dl_r) - dl_l;
                    let dr_2 = &(dr_r + dr_r) - dr_l;
                    let gate_2 = &(&nl_2 * &dr_2) + &(&dl_2 * &(&nr_2 + &(lambda * &dr_2)));

                    h0 = &h0 + &(&eq_k[j] * &gate_0);
                    h2 = &h2 + &(&eq_k[j] * &gate_2);
                }
                (&eq_row[row_idx] * &h0, &eq_row[row_idx] * &h2)
            };

        #[cfg(feature = "parallel")]
        {
            (0..self.trace_len)
                .into_par_iter()
                .fold(
                    || {
                        (
                            RowScratch::new(self.k_hat),
                            FieldElement::<E>::zero(),
                            FieldElement::<E>::zero(),
                        )
                    },
                    |(mut scratch, h0, h2), row_idx| {
                        let (r0, r2) = row_sums(&mut scratch, row_idx);
                        (scratch, &h0 + &r0, &h2 + &r2)
                    },
                )
                .map(|(_, h0, h2)| (h0, h2))
                .reduce(
                    || (FieldElement::<E>::zero(), FieldElement::<E>::zero()),
                    |(a0, a2), (b0, b2)| (&a0 + &b0, &a2 + &b2),
                )
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = RowScratch::new(self.k_hat);
            let mut h0 = FieldElement::<E>::zero();
            let mut h2 = FieldElement::<E>::zero();
            for row_idx in 0..self.trace_len {
                let (r0, r2) = row_sums(&mut scratch, row_idx);
                h0 = &h0 + &r0;
                h2 = &h2 + &r2;
            }
            (h0, h2)
        }
    }

    #[allow(clippy::type_complexity)]
    fn materialize_folded_rows(
        &self,
        c: usize,
        bound: &[FieldElement<E>],
    ) -> (
        Vec<FieldElement<E>>,
        Vec<FieldElement<E>>,
        Vec<FieldElement<E>>,
        Vec<FieldElement<E>>,
    ) {
        let n = self.trace_len;
        let mut nl = vec![FieldElement::<E>::zero(); n];
        let mut nr = vec![FieldElement::<E>::zero(); n];
        let mut dl = vec![FieldElement::<E>::zero(); n];
        let mut dr = vec![FieldElement::<E>::zero(); n];

        #[cfg(feature = "parallel")]
        nl.par_iter_mut()
            .zip(nr.par_iter_mut())
            .zip(dl.par_iter_mut())
            .zip(dr.par_iter_mut())
            .enumerate()
            .for_each_init(
                || RowScratch::new(self.k_hat),
                |scratch, (row_idx, (((nl_e, nr_e), dl_e), dr_e))| {
                    self.folded_row_arrays_into(c, bound, row_idx, scratch);
                    debug_assert_eq!(scratch.nl.len(), 1, "all k-bits must be bound");
                    *nl_e = scratch.nl[0].clone();
                    *nr_e = scratch.nr[0].clone();
                    *dl_e = scratch.dl[0].clone();
                    *dr_e = scratch.dr[0].clone();
                },
            );
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = RowScratch::new(self.k_hat);
            for row_idx in 0..n {
                self.folded_row_arrays_into(c, bound, row_idx, &mut scratch);
                debug_assert_eq!(scratch.nl.len(), 1, "all k-bits must be bound");
                nl[row_idx] = scratch.nl[0].clone();
                nr[row_idx] = scratch.nr[0].clone();
                dl[row_idx] = scratch.dl[0].clone();
                dr[row_idx] = scratch.dr[0].clone();
            }
        }
        (nl, nr, dl, dr)
    }
}

/// Build a table's batch-GKR instance (Stage 2 of the input-layer design):
/// the materialized layers from the N-sized cross-multiplied fractions up to
/// the root, plus the deep-layer oracle streaming the below-N layers. Peak
/// memory is one deep layer's split tables at a time — the K̂·N input layer is
/// never resident. Transcript-identical to a fully materialized extended tree.
pub fn build_gkr_instance<'a, F, E>(
    interactions: &'a [BusInteraction],
    main: &'a Table<F>,
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> GkrInstance<'a, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'a,
    E: IsField + Send + Sync + 'a,
{
    let (numerators, denominators) =
        compute_logup_leaf_fractions(interactions, main, trace_len, challenges);
    let upper_layers = gen_layers(Layer::LogUpGeneric {
        numerators,
        denominators,
    });
    let input_num_vars = gkr_input_num_vars(interactions.len());
    let deep_oracle: Option<Box<dyn DeepLayerOracle<E> + 'a>> = if input_num_vars > 0 {
        let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
        let max_bus_elements = interactions
            .iter()
            .map(|inter| inter.num_bus_elements())
            .max()
            .unwrap();
        Some(Box::new(LogUpDeepOracle {
            interactions,
            main,
            z: challenges[LOGUP_CHALLENGE_Z].clone(),
            alpha_powers: compute_alpha_powers(alpha, max_bus_elements),
            shifts: PackingShifts::<F>::new(),
            k_hat: interactions.len().next_power_of_two(),
            trace_len,
            fp_cache: std::sync::OnceLock::new(),
        }))
    } else {
        None
    };
    GkrInstance {
        upper_layers,
        input_num_vars,
        deep_oracle,
    }
}

/// Compute the full GKR layer tree for one table's interactions, from the
/// LINEAR input layer (`K̂·N` per-interaction leaves) up to the root. The
/// first `log2(K̂)` summation layers absorb the interaction sum; by pair
/// associativity the layer at size N equals the cross-multiplied per-row
/// fractions ([`compute_logup_leaf_fractions`]) bit-for-bit, and everything
/// above — including the root — is unchanged from the row-only tree.
pub fn compute_logup_layers<F, E>(
    interactions: &[BusInteraction],
    main: &Table<F>,
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Vec<Layer<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    gen_layers(compute_gkr_input_layer(
        interactions,
        main,
        trace_len,
        challenges,
    ))
}

// =============================================================================
// Per-table GKR result (prover side)
// =============================================================================

/// One table's outcome of the batch GKR run: the transcript-bound claims that
/// feed the bridge parameters and the proof.
#[derive(Debug, Clone)]
pub struct LogUpGkrResult<E: IsField> {
    /// The table's total LogUp contribution (`root_n / root_d`).
    pub table_contribution: FieldElement<E>,
    /// The GKR random evaluation point for this instance
    /// (length = log2(trace_len)).
    pub random_point: Vec<FieldElement<E>>,
    /// Claimed MLE evaluation of the leaf numerator at `random_point`.
    pub n_claim: FieldElement<E>,
    /// Claimed MLE evaluation of the leaf denominator at `random_point`.
    pub d_claim: FieldElement<E>,
    /// MLE claim per distinct referenced main column, in
    /// [`extract_column_indices`] order: `(column_index, ⟨l, col⟩)`.
    pub column_claims: Vec<(usize, FieldElement<E>)>,
}

/// Finalize a table's GKR result: evaluate every referenced main column's MLE
/// at the instance random point (via one shared Lagrange kernel, dropped after
/// use — the aux-trace build recomputes it to keep this result small).
pub fn finalize_logup_gkr_result<F, E>(
    interactions: &[BusInteraction],
    main: &Table<F>,
    random_point: Vec<FieldElement<E>>,
    n_claim: FieldElement<E>,
    d_claim: FieldElement<E>,
    table_contribution: FieldElement<E>,
) -> LogUpGkrResult<E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let col_indices = extract_column_indices(interactions);
    let kernel = compute_lagrange_kernel(&random_point);
    let k = col_indices.len();
    let num_rows = main.height;

    // All K inner products ⟨kernel, col_j⟩ in ONE row-major pass (per-chunk
    // partial accumulators, reduced by field addition — value-identical to
    // per-column sums, no column-major transpose of the segment).
    let row_partial = |acc: &mut [FieldElement<E>], row_idx: usize| {
        let row = main.get_row(row_idx);
        let k_r = &kernel[row_idx];
        for (j, &col_idx) in col_indices.iter().enumerate() {
            // F×E (base operand LEFT).
            acc[j] = &acc[j] + &(&row[col_idx] * k_r);
        }
    };

    #[cfg(feature = "parallel")]
    let claims: Vec<FieldElement<E>> = (0..num_rows)
        .into_par_iter()
        .fold(
            || vec![FieldElement::<E>::zero(); k],
            |mut acc, row_idx| {
                row_partial(&mut acc, row_idx);
                acc
            },
        )
        .reduce(
            || vec![FieldElement::<E>::zero(); k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x = &*x + &y;
                }
                a
            },
        );
    #[cfg(not(feature = "parallel"))]
    let claims: Vec<FieldElement<E>> = {
        let mut acc = vec![FieldElement::<E>::zero(); k];
        for row_idx in 0..num_rows {
            row_partial(&mut acc, row_idx);
        }
        acc
    };

    let column_claims: Vec<(usize, FieldElement<E>)> =
        col_indices.into_iter().zip(claims).collect();

    LogUpGkrResult {
        table_contribution,
        random_point,
        n_claim,
        d_claim,
        column_claims,
    }
}

// =============================================================================
// Bridge parameters (shared prover/verifier derivation)
// =============================================================================

/// The expected boundary value of the Lagrange kernel column:
/// `l[0] = eq(bits(0), r) = ∏_j (1 − r_j)`.
pub fn lagrange_l0<E: IsField>(random_point: &[FieldElement<E>]) -> FieldElement<E> {
    let one = FieldElement::<E>::one();
    random_point
        .iter()
        .fold(one.clone(), |acc, r_j| acc * (&one - r_j))
}

/// The expected squared ℓ₂ norm of the true Lagrange kernel:
/// `Σ_i l[i]² = ∏_k (r_k² + (1 − r_k)²)`. Folding this into the bridge target
/// (via `γ^K`) forces the committed `l` column to be `eq(·, r)` — a forged
/// kernel would have to preserve this norm AND every `⟨l, col_j⟩` claim.
pub fn lagrange_kernel_norm_claim<E: IsField>(random_point: &[FieldElement<E>]) -> FieldElement<E> {
    let one = FieldElement::<E>::one();
    random_point.iter().fold(one.clone(), |acc, r_k| {
        let one_minus_r = &one - r_k;
        acc * (r_k.square() + one_minus_r.square())
    })
}

/// Compute the bridge parameters from column claims:
/// `Δ = (Σ_j γ^j·c_j + γ^K·norm_claim) / N` and the γ powers `γ^0..γ^K`.
///
/// Both prover and verifier derive these from transcript-bound values.
pub fn compute_bridge_params<E: IsField>(
    column_claims: &[(usize, FieldElement<E>)],
    gamma: &FieldElement<E>,
    trace_len: usize,
    kernel_norm_claim: &FieldElement<E>,
) -> (FieldElement<E>, Vec<FieldElement<E>>) {
    let k = column_claims.len();
    let gamma_powers = compute_alpha_powers(gamma, k + 1);

    let mut target = FieldElement::<E>::zero();
    for ((_, c_j), gp) in column_claims.iter().zip(gamma_powers.iter()) {
        target += c_j * gp;
    }
    target += kernel_norm_claim * &gamma_powers[k];

    let n_inv = FieldElement::<E>::from(trace_len as u64)
        .inv()
        .expect("trace length is nonzero");
    let bridge_offset = &target * &n_inv;

    (bridge_offset, gamma_powers)
}

/// Extend the shared `[z, α]` challenge prefix into a table's full GKR-mode
/// rap_challenges vector (see the module-top layout).
pub fn extend_rap_challenges_with_bridge<E: IsField>(
    rap_challenges: &mut Vec<FieldElement<E>>,
    column_claims: &[(usize, FieldElement<E>)],
    gamma: &FieldElement<E>,
    trace_len: usize,
    random_point: &[FieldElement<E>],
) {
    debug_assert_eq!(rap_challenges.len(), LOGUP_CHALLENGE_GAMMA);
    let norm_claim = lagrange_kernel_norm_claim(random_point);
    let (bridge_offset, gamma_powers) =
        compute_bridge_params(column_claims, gamma, trace_len, &norm_claim);
    rap_challenges.push(gamma.clone());
    rap_challenges.push(bridge_offset);
    rap_challenges.extend(gamma_powers);
    rap_challenges.extend_from_slice(random_point);
}

// =============================================================================
// Auxiliary-trace build (GKR mode)
// =============================================================================

/// Fill the two GKR auxiliary columns (Lagrange kernel `l`, bridge running sum
/// `σ`) from the extended challenge vector. The columns must already be
/// allocated. `challenges` must be the full extended vector — the bridge
/// parameters and random point are read from their layout positions.
pub(crate) fn build_gkr_aux_columns<F, E>(
    trace: &mut TraceTable<F, E>,
    column_indices: &[usize],
    challenges: &[FieldElement<E>],
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let trace_len = trace.num_rows();
    assert!(
        trace_len.is_power_of_two(),
        "GKR aux build requires a power-of-two trace length, got {trace_len}"
    );
    let n_vars = trace_len.trailing_zeros() as usize;
    let k = column_indices.len();
    let rp_start = logup_random_point_start(k);
    assert_eq!(
        challenges.len(),
        rp_start + n_vars,
        "GKR aux build requires the extended challenge vector \
         (got {} challenges, expected {} for K={k}, n_vars={n_vars})",
        challenges.len(),
        rp_start + n_vars,
    );

    let random_point = &challenges[rp_start..];
    let gamma_powers = &challenges[LOGUP_GAMMA_POWERS_START..LOGUP_GAMMA_POWERS_START + k + 1];
    let bridge_offset = &challenges[LOGUP_BRIDGE_OFFSET_IDX];

    let kernel = compute_lagrange_kernel(random_point);

    // batched[i] = Σ_j γ^j·col_j[i] + γ^K·l[i] — row-parallel over borrowed
    // row-major rows (no column-major clone of the main segment), matching
    // the bridge constraint's batched sum exactly.
    let batched_at_row = |row_idx: usize| {
        let row = trace.main_table.get_row(row_idx);
        let mut acc = &gamma_powers[k] * &kernel[row_idx];
        for (j, &col_idx) in column_indices.iter().enumerate() {
            // col·γ is F×E (base operand LEFT).
            acc += &row[col_idx] * &gamma_powers[j];
        }
        acc
    };
    #[cfg(feature = "parallel")]
    let batched: Vec<FieldElement<E>> =
        (0..trace_len).into_par_iter().map(batched_at_row).collect();
    #[cfg(not(feature = "parallel"))]
    let batched: Vec<FieldElement<E>> = (0..trace_len).map(batched_at_row).collect();

    // σ[0] = 0; σ[i+1] = σ[i] + l[i]·batched[i] − Δ. The circular constraint
    // σ_next − σ_curr − l·batched + Δ = 0 then holds on every row, including
    // the wraparound (Σ l·batched = target = N·Δ closes the cycle).
    let mut sigma = FieldElement::<E>::zero();
    for row in 0..trace_len {
        trace.set_aux(row, GKR_AUX_KERNEL_COL, kernel[row].clone());
        trace.set_aux(row, GKR_AUX_SIGMA_COL, sigma.clone());
        sigma = sigma + &kernel[row] * &batched[row] - bridge_offset;
    }
    debug_assert_eq!(
        sigma,
        FieldElement::<E>::zero(),
        "bridge running sum must wrap to zero (Σ l·batched = N·Δ)"
    );
}

// =============================================================================
// Verifier-side claim reconstruction
// =============================================================================

/// Sorted-slice column-claim lookup (guest-friendly: no hashing).
struct ClaimLookup<'a, E: IsField>(&'a [(usize, FieldElement<E>)]);

impl<E: IsField> ClaimLookup<'_, E> {
    /// The claimed MLE value for `col`. The caller has already checked the
    /// claim index set equals [`extract_column_indices`], so every referenced
    /// column is present; a miss returns zero (unreachable after that check).
    fn get(&self, col: usize) -> FieldElement<E> {
        match self.0.binary_search_by_key(&col, |(c, _)| *c) {
            Ok(i) => self.0[i].1.clone(),
            Err(_) => FieldElement::<E>::zero(),
        }
    }
}

/// Verify a table's `column_claims` against the batch-GKR input-layer claims
/// `(n_claim, d_claim)` — the EXACT reconstruction the linear input layer
/// affords (thoughts/logup-gkr/input-layer-design.md).
///
/// Always enforced: the claim index set must EQUAL the canonical sorted
/// distinct column set of the interactions (same indices, same order).
///
/// The input-layer leaves `(±m_k(i), fp_k(i))` are LINEAR in the trace
/// columns, so the leaf-vector MLEs at the instance point `(κ, ρ)` factor
/// through the column MLEs at ρ (the `column_claims`, bound to the committed
/// trace by the bridge constraint):
///
/// ```text
/// n̂ = Σ_{k<K} eq(κ, bits(k)) · sign_k · m_k(ĉ)
/// d̂ = Σ_{k<K} eq(κ, bits(k)) · fp_k(ĉ)  +  (1 − Σ_{k<K} eq(κ, bits(k)))
/// ```
///
/// (the trailing term is the padding leaves' `d = 1`, via the eq kernel's
/// partition of unity). Both are checked for EXACT equality against the
/// transcript-derived claims — every table, every size, no fail-open branch.
/// `kappa` is the κ part of the instance point (`split_input_point`).
pub fn reconstruct_and_verify_gkr_claims<E: IsField>(
    n_claim: &FieldElement<E>,
    d_claim: &FieldElement<E>,
    column_claims: &[(usize, FieldElement<E>)],
    interactions: &[BusInteraction],
    challenges: &[FieldElement<E>],
    kappa: &[FieldElement<E>],
) -> bool {
    // Structural binding: exact index-set (and order) equality with the
    // canonical column list. Subsumes presence checks and pins the transcript
    // absorption order of the claims.
    let expected = extract_column_indices(interactions);
    if column_claims.len() != expected.len() {
        return false;
    }
    if !column_claims
        .iter()
        .zip(expected.iter())
        .all(|((c, _), e)| c == e)
    {
        return false;
    }

    // κ must have exactly the padded-interaction bit count (a public function
    // of the AIR, never proof-derived — the caller slices it off the
    // transcript-derived instance point).
    if kappa.len() != gkr_input_num_vars(interactions.len()) {
        return false;
    }

    let claims = ClaimLookup(column_claims);
    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let max_bus_elements = interactions
        .iter()
        .map(|inter| inter.num_bus_elements())
        .max()
        .unwrap_or(0);
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);

    // eq(κ, bits(k)) weights over the padded interaction hypercube.
    let eq_kappa = compute_lagrange_kernel(kappa);

    // Per-interaction linear forms on the column claims, eq-weighted.
    let mut n_recon = FieldElement::<E>::zero();
    let mut d_recon = FieldElement::<E>::zero();
    let mut eq_sum = FieldElement::<E>::zero();
    for (k, inter) in interactions.iter().enumerate() {
        let mut lc = FieldElement::<E>::from(inter.bus_id);
        let mut alpha_offset = 1;
        for bv in &inter.values {
            for elem in bv.combine_from(|col| claims.get(col)) {
                lc += &elem * &alpha_powers[alpha_offset];
                alpha_offset += 1;
            }
        }
        let fp = z - &lc;
        let m = inter.multiplicity.evaluate_from(|col| claims.get(col));
        let m = if inter.is_sender { m } else { -m };

        let eq_k = &eq_kappa[k];
        n_recon += eq_k * &m;
        d_recon += eq_k * &fp;
        eq_sum += eq_k.clone();
    }
    // Padding leaves: n = 0 (nothing), d = 1 → Σ_{k≥K} eq(κ,k)·1, computed
    // via partition of unity (Σ_all eq = 1).
    d_recon += FieldElement::<E>::one() - eq_sum;

    n_recon == *n_claim && d_recon == *d_claim
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::{Multiplicity, Packing};
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    /// Single sender, multiplicity 1: N(i) = 1, D(i) = fp(i).
    #[test]
    fn leaf_fractions_single_sender() {
        let trace_len = 4;
        let col0: Vec<FE> = [10u64, 20, 30, 40].iter().map(|&v| FE::from(v)).collect();
        let main_segment_cols = Table::from_columns(vec![col0.clone()]);
        let interactions = vec![BusInteraction::sender(
            1u64,
            Multiplicity::One,
            Packing::Direct.columns(&[0]),
        )];
        let z = FE::from(100u64);
        let alpha = FE::from(3u64);
        let challenges = vec![z, alpha];

        let (numerators, denominators) = compute_logup_leaf_fractions::<F, F>(
            &interactions,
            &main_segment_cols,
            trace_len,
            &challenges,
        );

        let alpha_powers = compute_alpha_powers(&alpha, 2);
        for row in 0..trace_len {
            let lc = FE::from(1u64) + col0[row] * alpha_powers[1];
            let expected_fp = z - lc;
            assert_eq!(numerators[row], FE::one());
            assert_eq!(denominators[row], expected_fp);
        }
    }

    /// Single receiver with a column multiplicity: N(i) = −m(i), D(i) = fp(i).
    #[test]
    fn leaf_fractions_single_receiver_column_multiplicity() {
        let trace_len = 4;
        let col0: Vec<FE> = [5u64, 6, 7, 8].iter().map(|&v| FE::from(v)).collect();
        let col1: Vec<FE> = [2u64, 0, 1, 3].iter().map(|&v| FE::from(v)).collect();
        let main_segment_cols = Table::from_columns(vec![col0.clone(), col1.clone()]);
        let interactions = vec![BusInteraction::receiver(
            0u64,
            Multiplicity::Column(1),
            Packing::Direct.columns(&[0]),
        )];
        let z = FE::from(50u64);
        let alpha = FE::from(7u64);
        let challenges = vec![z, alpha];

        let (numerators, denominators) = compute_logup_leaf_fractions::<F, F>(
            &interactions,
            &main_segment_cols,
            trace_len,
            &challenges,
        );

        let alpha_powers = compute_alpha_powers(&alpha, 2);
        for row in 0..trace_len {
            let lc = col0[row] * alpha_powers[1];
            let expected_fp = z - lc;
            assert_eq!(numerators[row], -col1[row]);
            assert_eq!(denominators[row], expected_fp);
        }
    }

    /// Two interactions: n = fp₁ − fp₀ (sender then receiver, both m=1),
    /// d = fp₀·fp₁.
    #[test]
    fn leaf_fractions_two_interactions_cross_multiply() {
        let trace_len = 2;
        let col0: Vec<FE> = [10u64, 20].iter().map(|&v| FE::from(v)).collect();
        let col1: Vec<FE> = [30u64, 40].iter().map(|&v| FE::from(v)).collect();
        let main_segment_cols = Table::from_columns(vec![col0.clone(), col1.clone()]);
        let interactions = vec![
            BusInteraction::sender(0u64, Multiplicity::One, Packing::Direct.columns(&[0])),
            BusInteraction::receiver(1u64, Multiplicity::One, Packing::Direct.columns(&[1])),
        ];
        let z = FE::from(200u64);
        let alpha = FE::from(5u64);
        let challenges = vec![z, alpha];

        let (numerators, denominators) = compute_logup_leaf_fractions::<F, F>(
            &interactions,
            &main_segment_cols,
            trace_len,
            &challenges,
        );

        let alpha_powers = compute_alpha_powers(&alpha, 2);
        for row in 0..trace_len {
            let fp_0 = z - (col0[row] * alpha_powers[1]);
            let fp_1 = z - (FE::from(1u64) + col1[row] * alpha_powers[1]);
            assert_eq!(numerators[row], fp_1 - fp_0);
            assert_eq!(denominators[row], fp_0 * fp_1);
        }
    }

    /// Column extraction covers bus values AND multiplicity columns, sorted
    /// and deduplicated.
    #[test]
    fn extract_column_indices_covers_values_and_multiplicities() {
        let interactions = vec![
            BusInteraction::sender(0u64, Multiplicity::Sum(7, 2), Packing::Word2L.columns(&[4])),
            BusInteraction::receiver(1u64, Multiplicity::Column(2), Packing::Direct.columns(&[5])),
        ];
        assert_eq!(extract_column_indices(&interactions), vec![2, 4, 5, 7]);
    }

    /// The verifier-side reconstruction accepts honest single-interaction
    /// claims and rejects a tampered one; the exact index-set check rejects
    /// missing or extra claims.
    #[test]
    fn reconstruct_single_interaction_roundtrip_and_tamper() {
        // Single-row table (n_vars = 0): the MLE at the empty point IS the row
        // value, so honest claims are the row values themselves.
        let col0 = FE::from(10u64);
        let col1 = FE::from(3u64);
        let interactions = vec![BusInteraction::sender(
            2u64,
            Multiplicity::Column(1),
            Packing::Direct.columns(&[0]),
        )];
        let z = FE::from(1000u64);
        let alpha = FE::from(11u64);
        let challenges = vec![z, alpha];
        let alpha_powers = compute_alpha_powers(&alpha, 2);

        let fp = z - (FE::from(2u64) + col0 * alpha_powers[1]);
        let n_claim = col1;
        let d_claim = fp;
        let column_claims = vec![(0usize, col0), (1usize, col1)];

        assert!(reconstruct_and_verify_gkr_claims(
            &n_claim,
            &d_claim,
            &column_claims,
            &interactions,
            &challenges,
            &[],
        ));

        let tampered = FE::from(999u64);
        assert!(!reconstruct_and_verify_gkr_claims(
            &tampered,
            &d_claim,
            &column_claims,
            &interactions,
            &challenges,
            &[],
        ));

        // Missing claim → reject.
        assert!(!reconstruct_and_verify_gkr_claims(
            &n_claim,
            &d_claim,
            &column_claims[..1],
            &interactions,
            &challenges,
            &[],
        ));
        // Extra claim → reject.
        let mut extra = column_claims.clone();
        extra.push((9, FE::from(1u64)));
        assert!(!reconstruct_and_verify_gkr_claims(
            &n_claim,
            &d_claim,
            &extra,
            &interactions,
            &challenges,
            &[],
        ));
    }

    /// Fact 1 of the input-layer design: by pair-level associativity of
    /// fraction addition, the extended tree's layer at size N (after the
    /// `log2(K̂)` interaction-summing layers) equals the cross-multiplied
    /// per-row fractions of [`compute_logup_leaf_fractions`] BIT-FOR-BIT —
    /// including with padding (K = 3 → K̂ = 4 exercises the (0,1) identity).
    #[test]
    fn extended_tree_n_layer_matches_cross_multiplied_fractions() {
        let trace_len = 8usize;
        let col0: Vec<FE> = (1..=8).map(|v| FE::from(v as u64)).collect();
        let col1: Vec<FE> = (21..=28).map(|v| FE::from(v as u64)).collect();
        let col2: Vec<FE> = [1u64, 0, 2, 1, 0, 3, 1, 1]
            .iter()
            .map(|&v| FE::from(v))
            .collect();
        let main = Table::from_columns(vec![col0, col1, col2]);

        let interactions = vec![
            BusInteraction::sender(3u64, Multiplicity::One, Packing::Direct.columns(&[0])),
            BusInteraction::receiver(3u64, Multiplicity::Column(2), Packing::Direct.columns(&[1])),
            BusInteraction::sender(5u64, Multiplicity::Column(2), Packing::Word2L.columns(&[0])),
        ];
        let challenges = vec![FE::from(0xDEAD_BEEFu64), FE::from(0x1234_5678u64)];

        let (expected_n, expected_d) =
            compute_logup_leaf_fractions::<F, F>(&interactions, &main, trace_len, &challenges);

        let layers = compute_logup_layers::<F, F>(&interactions, &main, trace_len, &challenges);
        let input_vars = gkr_input_num_vars(interactions.len());
        assert_eq!(input_vars, 2, "K = 3 pads to K̂ = 4");
        let n_layer = &layers[input_vars];
        match n_layer {
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                assert_eq!(numerators.len(), trace_len);
                assert_eq!(numerators, &expected_n, "numerators diverge at the N layer");
                assert_eq!(
                    denominators, &expected_d,
                    "denominators diverge at the N layer"
                );
            }
            other => panic!("unexpected layer variant: {other:?}"),
        }
    }

    /// THE Stage-2 gate: the streamed deep-layer path
    /// ([`build_gkr_instance`], oracle-backed) must be TRANSCRIPT-IDENTICAL
    /// to a fully materialized extended tree ([`compute_logup_layers`]) —
    /// same seed, byte-identical proof, same point, same claims. Mixed sizes
    /// and padding included (K = 3 → K̂ = 4, two trace lengths).
    #[test]
    fn streamed_deep_layers_match_materialized_extended_tree() {
        use crate::gkr::{GkrInstance, gkr_prove_batch};
        use crypto::fiat_shamir::default_transcript::DefaultTranscript;
        use crypto::fiat_shamir::is_transcript::IsTranscript;

        let mk_table = |rows: usize, seed: u64| {
            let col0: Vec<FE> = (0..rows).map(|v| FE::from(seed + v as u64)).collect();
            let col1: Vec<FE> = (0..rows).map(|v| FE::from(seed + 100 + v as u64)).collect();
            let col2: Vec<FE> = (0..rows).map(|v| FE::from((v as u64) % 3)).collect();
            Table::from_columns(vec![col0, col1, col2])
        };
        let interactions = vec![
            BusInteraction::sender(3u64, Multiplicity::One, Packing::Direct.columns(&[0])),
            BusInteraction::receiver(3u64, Multiplicity::Column(2), Packing::Direct.columns(&[1])),
            BusInteraction::sender(5u64, Multiplicity::Column(2), Packing::Word2L.columns(&[0])),
        ];
        let challenges = vec![FE::from(0xDEAD_BEEFu64), FE::from(0x1234_5678u64)];

        let table_a = mk_table(16, 7);
        let table_b = mk_table(8, 1_000);
        // A wide instance (K = 20 → K̂ = 32) exercises the STREAMED deep
        // rounds (slots 32 and 16 stream; slots ≤ 8 materialize).
        let wide_interactions: Vec<BusInteraction> = (0..20)
            .map(|k| {
                if k % 2 == 0 {
                    BusInteraction::sender(
                        k as u64,
                        Multiplicity::One,
                        Packing::Direct.columns(&[k % 3]),
                    )
                } else {
                    BusInteraction::receiver(
                        k as u64,
                        Multiplicity::Column(2),
                        Packing::Direct.columns(&[(k + 1) % 3]),
                    )
                }
            })
            .collect();
        let table_c = mk_table(16, 55);

        // Materialized extended trees (the Stage-1 shape).
        let mat_a = compute_logup_layers::<F, F>(&interactions, &table_a, 16, &challenges);
        let mat_b = compute_logup_layers::<F, F>(&interactions, &table_b, 8, &challenges);
        let mat_c = compute_logup_layers::<F, F>(&wide_interactions, &table_c, 16, &challenges);
        let mut t1 = DefaultTranscript::<F>::new(&[42]);
        let (proof_m, point_m, claims_m) = gkr_prove_batch(
            vec![
                GkrInstance::materialized(mat_a),
                GkrInstance::materialized(mat_b),
                GkrInstance::materialized(mat_c),
            ],
            &mut t1,
        );

        // Streamed instances (the Stage-2 shape).
        let inst_a = build_gkr_instance::<F, F>(&interactions, &table_a, 16, &challenges);
        let inst_b = build_gkr_instance::<F, F>(&interactions, &table_b, 8, &challenges);
        let inst_c = build_gkr_instance::<F, F>(&wide_interactions, &table_c, 16, &challenges);
        let mut t2 = DefaultTranscript::<F>::new(&[42]);
        let (proof_s, point_s, claims_s) = gkr_prove_batch(vec![inst_a, inst_b, inst_c], &mut t2);

        assert_eq!(point_m, point_s, "shared random points diverge");
        assert_eq!(claims_m, claims_s, "instance claims diverge");
        let bytes_m = rkyv::to_bytes::<rkyv::rancor::Error>(&proof_m).unwrap();
        let bytes_s = rkyv::to_bytes::<rkyv::rancor::Error>(&proof_s).unwrap();
        assert_eq!(
            bytes_m.as_slice(),
            bytes_s.as_slice(),
            "streamed and materialized proofs are not byte-identical"
        );
        // The transcripts must have absorbed identical data too.
        assert_eq!(t1.state(), t2.state(), "transcript states diverge");
    }

    /// The cross-mode oracle: the GKR summation-tree ROOT must equal the
    /// standard-mode table contribution `L = Σ_rows Σ_k ±m_k/fp_k` (computed
    /// via the standard aux path's per-interaction term columns). This ties
    /// leaf fractions AND `gen_layers` to the standard LogUp semantics — the
    /// two modes must claim the same bus balance.
    #[test]
    fn gkr_root_matches_standard_table_contribution() {
        let trace_len = 8usize;
        let col0: Vec<FE> = (1..=8).map(|v| FE::from(v as u64)).collect();
        let col1: Vec<FE> = (21..=28).map(|v| FE::from(v as u64)).collect();
        let col2: Vec<FE> = [1u64, 0, 2, 1, 0, 3, 1, 1]
            .iter()
            .map(|&v| FE::from(v))
            .collect();
        // The standard term-column path takes column-major vectors; the GKR
        // path reads the row-major table — same data, both representations.
        let columns = vec![col0, col1, col2];
        let main_segment_cols = Table::from_columns(columns.clone());

        let interactions = vec![
            BusInteraction::sender(3u64, Multiplicity::One, Packing::Direct.columns(&[0])),
            BusInteraction::receiver(3u64, Multiplicity::Column(2), Packing::Direct.columns(&[1])),
            BusInteraction::sender(5u64, Multiplicity::Column(2), Packing::Word2L.columns(&[0])),
        ];
        let challenges = vec![FE::from(0xDEAD_BEEFu64), FE::from(0x1234_5678u64)];

        // Standard-mode L: sum every interaction's term column.
        let mut expected_l = FE::zero();
        for inter in &interactions {
            let terms = crate::lookup::compute_logup_term_column::<F, F>(
                &[inter],
                &columns,
                trace_len,
                &challenges,
                "oracle",
            );
            for t in &terms {
                expected_l = expected_l + t;
            }
        }

        // GKR: leaf fractions → summation tree → root claim n/d.
        let layers =
            compute_logup_layers::<F, F>(&interactions, &main_segment_cols, trace_len, &challenges);
        let (root_n, root_d) = match layers.last().expect("root layer") {
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                assert_eq!(numerators.len(), 1, "root layer has one element");
                (numerators[0], denominators[0])
            }
            other => panic!("unexpected root layer variant: {other:?}"),
        };
        let root_value = root_n * root_d.inv().expect("nonzero root denominator");
        assert_eq!(root_value, expected_l, "GKR root != standard-mode L");
    }

    /// The leaf-binding check (formerly the KNOWN SOUNDNESS GAP of
    /// port-plan.md §6, closed by the linear input layer): fabricated
    /// multi-interaction leaf claims are REJECTED, and the honest
    /// reconstruction is accepted — exact equality, both n̂ and d̂.
    #[test]
    fn reconstruct_multi_interaction_rejects_fabricated_leaf_claims() {
        let interactions = vec![
            BusInteraction::sender(1u64, Multiplicity::One, Packing::Direct.columns(&[0])),
            BusInteraction::receiver(2u64, Multiplicity::One, Packing::Direct.columns(&[1])),
        ];
        let challenges = vec![FE::from(1000u64), FE::from(7u64)];
        let alpha_powers = compute_alpha_powers(&challenges[1], 2);
        let column_claims = vec![(0usize, FE::from(5u64)), (1usize, FE::from(9u64))];
        // K = 2 → K̂ = 2 → one κ coordinate.
        let kappa = vec![FE::from(13u64)];

        // Honest input-layer claims: eq-weighted linear forms on the claims.
        let eq = compute_lagrange_kernel(&kappa);
        let fp0 = challenges[0] - (FE::from(1u64) + FE::from(5u64) * alpha_powers[1]);
        let fp1 = challenges[0] - (FE::from(2u64) + FE::from(9u64) * alpha_powers[1]);
        let honest_n = eq[0] * FE::one() + eq[1] * (-FE::one());
        let honest_d = eq[0] * fp0 + eq[1] * fp1;
        assert!(
            reconstruct_and_verify_gkr_claims(
                &honest_n,
                &honest_d,
                &column_claims,
                &interactions,
                &challenges,
                &kappa,
            ),
            "honest multi-interaction leaf claims must be accepted"
        );

        // Fabricated leaf claims that no leaf vector consistent with the
        // columns could produce — must be rejected (this was the fail-open
        // hole before the linear input layer).
        let fabricated_n = FE::from(0xBADu64);
        let fabricated_d = FE::from(0xC0DEu64);
        assert!(
            !reconstruct_and_verify_gkr_claims(
                &fabricated_n,
                &fabricated_d,
                &column_claims,
                &interactions,
                &challenges,
                &kappa,
            ),
            "fabricated multi-interaction leaf claims must be rejected"
        );

        // A wrong-length κ (proof-shape confusion) must be rejected.
        assert!(
            !reconstruct_and_verify_gkr_claims(
                &honest_n,
                &honest_d,
                &column_claims,
                &interactions,
                &challenges,
                &[],
            ),
            "wrong κ length must be rejected"
        );
    }

    /// Bridge parameters: Δ·N == Σ γʲ·cⱼ + γᴷ·norm_claim, and the kernel norm
    /// claim matches the actual kernel's Σ l².
    #[test]
    fn bridge_params_and_kernel_norm() {
        let r = vec![FE::from(3u64), FE::from(7u64)];
        let kernel = compute_lagrange_kernel(&r);
        let norm: FE = kernel.iter().fold(FE::zero(), |acc, l| acc + l * l);
        assert_eq!(norm, lagrange_kernel_norm_claim(&r));

        let column_claims = vec![(0usize, FE::from(5u64)), (3usize, FE::from(8u64))];
        let gamma = FE::from(13u64);
        let trace_len = 4usize;
        let (delta, gamma_powers) = compute_bridge_params(&column_claims, &gamma, trace_len, &norm);
        assert_eq!(gamma_powers.len(), 3);
        let target = FE::from(5u64) * gamma_powers[0]
            + FE::from(8u64) * gamma_powers[1]
            + norm * gamma_powers[2];
        assert_eq!(delta * FE::from(trace_len as u64), target);
    }

    /// The GKR aux columns satisfy the bridge recurrence and boundary values,
    /// and honest column claims telescope: Σ l·batched = N·Δ.
    #[test]
    fn gkr_aux_columns_satisfy_bridge_recurrence() {
        use crate::trace::TraceTable;

        let trace_len = 8usize;
        let n_vars = 3usize;
        let col0: Vec<FE> = (1..=8).map(|v| FE::from(v as u64)).collect();
        let col1: Vec<FE> = (11..=18).map(|v| FE::from(v as u64)).collect();
        let column_indices = vec![0usize, 1usize];
        let k = column_indices.len();

        let mut trace = TraceTable::<F, F>::from_columns_main(vec![col0.clone(), col1.clone()], 1);
        trace.allocate_aux_table(GKR_NUM_AUX_COLUMNS);

        // Honest bridge parameters from honest column claims.
        let random_point: Vec<FE> = [3u64, 7, 5].iter().map(|&v| FE::from(v)).collect();
        assert_eq!(random_point.len(), n_vars);
        let kernel = compute_lagrange_kernel(&random_point);
        let column_claims: Vec<(usize, FE)> = column_indices
            .iter()
            .map(|&c| {
                let cols = [&col0, &col1];
                let claim = cols[c]
                    .iter()
                    .zip(kernel.iter())
                    .fold(FE::zero(), |acc, (v, l)| acc + v * l);
                (c, claim)
            })
            .collect();
        let gamma = FE::from(17u64);

        let mut challenges = vec![FE::from(1u64), FE::from(2u64)]; // z, α (unused here)
        extend_rap_challenges_with_bridge(
            &mut challenges,
            &column_claims,
            &gamma,
            trace_len,
            &random_point,
        );
        assert_eq!(challenges.len(), logup_random_point_start(k) + n_vars);

        build_gkr_aux_columns(&mut trace, &column_indices, &challenges);

        // Boundary values.
        let l0 = trace.get_aux(0, GKR_AUX_KERNEL_COL);
        assert_eq!(*l0, lagrange_l0(&random_point));
        let sigma0 = trace.get_aux(0, GKR_AUX_SIGMA_COL);
        assert_eq!(*sigma0, FE::zero());

        // The circular recurrence on every row (including wraparound).
        let delta = &challenges[LOGUP_BRIDGE_OFFSET_IDX];
        let gp = &challenges[LOGUP_GAMMA_POWERS_START..LOGUP_GAMMA_POWERS_START + k + 1];
        for row in 0..trace_len {
            let next = (row + 1) % trace_len;
            let l = *trace.get_aux(row, GKR_AUX_KERNEL_COL);
            let sigma_curr = *trace.get_aux(row, GKR_AUX_SIGMA_COL);
            let sigma_next = *trace.get_aux(next, GKR_AUX_SIGMA_COL);
            let batched = col0[row] * gp[0] + col1[row] * gp[1] + gp[k] * l;
            assert_eq!(
                sigma_next - sigma_curr - l * batched + delta,
                FE::zero(),
                "bridge recurrence failed at row {row}"
            );
        }
    }
}
