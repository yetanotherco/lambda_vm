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
    traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::gkr::{Layer, gen_layers};
use crate::lagrange_kernel::{compute_lagrange_kernel, eval_mle_base_with_kernel};
use crate::lookup::{BusInteraction, LOGUP_CHALLENGE_ALPHA, PackingShifts, compute_alpha_powers};
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
/// row. `α⁰ = 1`: the bus-id term is embedded directly, no multiply.
fn compute_fingerprint_at_row<F, E>(
    interaction: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    row: usize,
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
        alpha_offset += bv.accumulate_fingerprint(
            main_segment_cols,
            row,
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
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> (Vec<FieldElement<E>>, Vec<FieldElement<E>>)
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
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

    let leaf_at_row = |row: usize| {
        let mut running_n = FieldElement::<E>::zero();
        let mut running_d = FieldElement::<E>::one();
        for inter in interactions {
            let fp = compute_fingerprint_at_row(
                inter,
                main_segment_cols,
                row,
                z,
                &alpha_powers,
                &shifts,
            );
            let m = inter.multiplicity.evaluate_at_row(main_segment_cols, row);
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

/// Compute the full GKR layer tree for one table's interactions: leaf
/// fractions, then pairwise fraction-summation layers up to the root.
pub fn compute_logup_layers<F, E>(
    interactions: &[BusInteraction],
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Vec<Layer<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    let (numerators, denominators) =
        compute_logup_leaf_fractions(interactions, main_segment_cols, trace_len, challenges);
    gen_layers(Layer::LogUpGeneric {
        numerators,
        denominators,
    })
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
    main_segment_cols: &[Vec<FieldElement<F>>],
    random_point: Vec<FieldElement<E>>,
    n_claim: FieldElement<E>,
    d_claim: FieldElement<E>,
    table_contribution: FieldElement<E>,
) -> LogUpGkrResult<E>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    let col_indices = extract_column_indices(interactions);
    let kernel = compute_lagrange_kernel(&random_point);

    #[cfg(feature = "parallel")]
    let col_iter = col_indices.into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let col_iter = col_indices.into_iter();

    let column_claims: Vec<(usize, FieldElement<E>)> = col_iter
        .map(|col_idx| {
            let claim = eval_mle_base_with_kernel(&main_segment_cols[col_idx], &kernel);
            (col_idx, claim)
        })
        .collect();

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
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
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
    let main_segment_cols = trace.columns_main();

    // batched[i] = Σ_j γ^j·col_j[i] + γ^K·l[i] — row-parallel, matches the
    // bridge constraint's batched sum exactly.
    let batched_at_row = |row: usize| {
        let mut acc = &gamma_powers[k] * &kernel[row];
        for (j, &col_idx) in column_indices.iter().enumerate() {
            // col·γ is F×E (base operand LEFT).
            acc += &main_segment_cols[col_idx][row] * &gamma_powers[j];
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

/// Verify a table's `column_claims` against the batch-GKR leaf claims
/// `(n_claim, d_claim)`.
///
/// Always enforced: the claim index set must EQUAL the canonical sorted
/// distinct column set of the interactions (same indices, same order).
///
/// For single-interaction tables and 0-layer (single-row) instances the leaf
/// fraction is reconstructible from the column MLEs — single-interaction
/// leaves are linear in the columns, and at the empty point the MLE is the row
/// value itself — so the rational cross-check
/// `n_recon·d_claim == n_claim·d_recon` is exact and enforced.
///
/// # KNOWN SOUNDNESS GAP (multi-interaction tables)
///
/// For `interactions.len() > 1` with `n_layers > 0` this check is FAIL-OPEN:
/// the leaf fraction is a nonlinear (cross-multiplied) function of the
/// columns, MLE does not commute with products, and nothing else binds
/// `(n_claim, d_claim)` to the committed trace — the bridge constraint binds
/// only `column_claims`. A malicious prover can therefore run an honest GKR
/// over fabricated leaves. Every production table is multi-interaction. The
/// fix (a linear input layer or an input-layer sumcheck) is designed as the
/// immediate follow-up — see `thoughts/logup-gkr/port-plan.md` §6. Do NOT
/// treat GKR mode as production-sound until it lands.
pub fn reconstruct_and_verify_gkr_claims<E: IsField>(
    n_claim: &FieldElement<E>,
    d_claim: &FieldElement<E>,
    column_claims: &[(usize, FieldElement<E>)],
    interactions: &[BusInteraction],
    challenges: &[FieldElement<E>],
    n_layers: usize,
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

    let claims = ClaimLookup(column_claims);
    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let max_bus_elements = interactions
        .iter()
        .map(|inter| inter.num_bus_elements())
        .max()
        .unwrap_or(0);
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);

    // Reconstruct the leaf fraction from the column claims with the SAME
    // cross-multiplication recurrence as `compute_logup_leaf_fractions`.
    let mut running_n = FieldElement::<E>::zero();
    let mut running_d = FieldElement::<E>::one();
    for inter in interactions {
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
        let cross = &m * &running_d;
        let cross = if inter.is_sender { cross } else { -cross };
        running_n = &running_n * &fp + cross;
        running_d = &running_d * &fp;
    }

    if interactions.len() == 1 || n_layers == 0 {
        // Rational cross-check: n_recon/d_recon == n_claim/d_claim.
        &running_n * d_claim == n_claim * &running_d
    } else {
        // FAIL-OPEN — see the doc comment's KNOWN SOUNDNESS GAP.
        true
    }
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
        let main_segment_cols = vec![col0.clone()];
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
        let main_segment_cols = vec![col0.clone(), col1.clone()];
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
        let main_segment_cols = vec![col0.clone(), col1.clone()];
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
            0,
        ));

        let tampered = FE::from(999u64);
        assert!(!reconstruct_and_verify_gkr_claims(
            &tampered,
            &d_claim,
            &column_claims,
            &interactions,
            &challenges,
            0,
        ));

        // Missing claim → reject.
        assert!(!reconstruct_and_verify_gkr_claims(
            &n_claim,
            &d_claim,
            &column_claims[..1],
            &interactions,
            &challenges,
            0,
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
            0,
        ));
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
        let main_segment_cols = vec![col0, col1, col2];

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
                &main_segment_cols,
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
