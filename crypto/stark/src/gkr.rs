use crate::sumcheck::{RoundPoly, SumcheckProof};
use core::fmt;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Minimum parent_num_vars for enabling Split-Value Optimization (SVO).
/// Below this threshold, the standard flat eq_table approach is used.
const SVO_THRESHOLD: usize = 8;

// =============================================================================
// Layer enum for gate-specialized GKR
// =============================================================================

/// A layer in the GKR binary-tree circuit.
///
/// Each layer has half the size of the layer below it. The leaves (input layer)
/// are at the bottom, and the root (output layer) has a single element.
///
/// Different gate types allow specialized inner loops:
/// - `LogUpGeneric`: explicit numerators and denominators (6 muls/pair in sumcheck)
/// - `LogUpSingles`: numerators are implicitly 1 (2 muls/pair — ~50% savings at leaf)
#[derive(Debug, Clone)]
pub enum Layer<E: IsField> {
    /// LogUp with explicit numerators and denominators.
    LogUpGeneric {
        numerators: Vec<FieldElement<E>>,
        denominators: Vec<FieldElement<E>>,
    },
    /// LogUp where all numerators are implicitly 1.
    /// Saves ~50% muls in the sumcheck inner loop (no nl/nr tables needed).
    LogUpSingles { denominators: Vec<FieldElement<E>> },
}

impl<E: IsField> Layer<E> {
    /// Number of variables: log2 of the layer size.
    pub fn n_variables(&self) -> usize {
        let len = match self {
            Self::LogUpGeneric { denominators, .. } => denominators.len(),
            Self::LogUpSingles { denominators } => denominators.len(),
        };
        debug_assert!(len.is_power_of_two());
        len.trailing_zeros() as usize
    }

    /// Whether this is the root layer (single value, 0 variables).
    pub fn is_output_layer(&self) -> bool {
        self.n_variables() == 0
    }

    /// Returns the root (numerator, denominator) for a single-element output layer.
    pub fn try_into_output_values(&self) -> Option<(FieldElement<E>, FieldElement<E>)> {
        if !self.is_output_layer() {
            return None;
        }
        Some(match self {
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => (numerators[0].clone(), denominators[0].clone()),
            Layer::LogUpSingles { denominators } => (FieldElement::one(), denominators[0].clone()),
        })
    }

    /// Computes the next (parent) layer by pairwise fraction addition.
    /// Returns `None` if already at the output layer.
    ///
    /// Both Singles and Generic produce Generic output (since 1/a + 1/b = (a+b)/(a*b)
    /// requires an explicit numerator).
    pub fn next_layer(&self) -> Option<Self> {
        if self.is_output_layer() {
            return None;
        }
        Some(match self {
            Self::LogUpGeneric {
                numerators,
                denominators,
            } => next_logup_layer(Some(numerators), denominators),
            Self::LogUpSingles { denominators } => next_logup_layer(None, denominators),
        })
    }
}

/// Pairwise fraction addition for LogUp layers.
///
/// If `numerators` is `None`, all numerators are implicitly 1 (singles case):
///   1/d[2j] + 1/d[2j+1] = (d[2j+1] + d[2j]) / (d[2j] * d[2j+1])
///
/// Otherwise: n[2j]/d[2j] + n[2j+1]/d[2j+1] = cross-multiply.
fn next_logup_layer<E: IsField>(
    numerators: Option<&[FieldElement<E>]>,
    denominators: &[FieldElement<E>],
) -> Layer<E> {
    let half_n = denominators.len() / 2;
    let mut next_numerators = Vec::with_capacity(half_n);
    let mut next_denominators = Vec::with_capacity(half_n);

    for j in 0..half_n {
        let dl = &denominators[2 * j];
        let dr = &denominators[2 * j + 1];
        let (num, den) = match numerators {
            Some(nums) => {
                let nl = &nums[2 * j];
                let nr = &nums[2 * j + 1];
                // nl/dl + nr/dr = (nl*dr + nr*dl) / (dl*dr)
                (&(nl * dr) + &(nr * dl), dl * dr)
            }
            None => {
                // 1/dl + 1/dr = (dr + dl) / (dl*dr)
                (dl + dr, dl * dr)
            }
        };
        next_numerators.push(num);
        next_denominators.push(den);
    }

    Layer::LogUpGeneric {
        numerators: next_numerators,
        denominators: next_denominators,
    }
}

/// Generates all layers from the input (leaves) to the output (root).
///
/// Returns layers[0] = input (leaves), layers[last] = output (root, 1 element).
pub fn gen_layers<E: IsField>(input_layer: Layer<E>) -> Vec<Layer<E>> {
    let n_variables = input_layer.n_variables();
    let mut layers = vec![input_layer];
    while let Some(next) = layers.last().unwrap().next_layer() {
        layers.push(next);
    }
    assert_eq!(layers.len(), n_variables + 1);
    layers
}

// =============================================================================
// Batch GKR proof types
// =============================================================================

/// Proof for a single layer in a batch GKR reduction.
///
/// Contains the shared sumcheck proof and per-instance child claims (masks).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct BatchGkrLayerProof<E: IsField> {
    /// Shared sumcheck proof for this layer (combined across all active instances).
    pub sumcheck_proof: SumcheckProof<E>,
    /// Per-active-instance child claims: [n_left, n_right, d_left, d_right].
    /// Order matches the order instances became active.
    pub child_claims_by_instance: Vec<[FieldElement<E>; 4]>,
}

/// Complete batch GKR proof for multiple fractional summation trees.
///
/// All instances share one sumcheck per layer via random linear combination.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct BatchGkrProof<E: IsField> {
    /// Per-instance root claims as (numerator, denominator) pairs.
    /// The claimed sum for instance i is root_claims[i].0 / root_claims[i].1.
    pub root_claims: Vec<(FieldElement<E>, FieldElement<E>)>,
    /// One layer proof per reduction step (from root towards leaves).
    /// The number of layers equals max(n_variables) across all instances.
    pub layer_proofs: Vec<BatchGkrLayerProof<E>>,
}

/// A rational number `numerator / denominator` in a field.
///
/// Used in the GKR protocol to represent LogUp contributions before they are
/// reduced to a single field element via batch inversion. Fraction addition
/// uses cross-multiplication to avoid per-addition inversions.
pub struct Fraction<E: IsField> {
    pub numerator: FieldElement<E>,
    pub denominator: FieldElement<E>,
}

impl<E: IsField> Fraction<E> {
    /// Create a new fraction `numerator / denominator`.
    pub fn new(numerator: FieldElement<E>, denominator: FieldElement<E>) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Add two fractions via cross-multiplication:
    ///   a/b + c/d = (a*d + c*b) / (b*d)
    ///
    /// No reduction or normalization is performed.
    pub fn add(&self, other: &Fraction<E>) -> Fraction<E> {
        let numerator =
            &(&self.numerator * &other.denominator) + &(&other.numerator * &self.denominator);
        let denominator = &self.denominator * &other.denominator;
        Fraction {
            numerator,
            denominator,
        }
    }
}

/// One layer of the summation tree used in the GKR protocol.
///
/// Each layer stores parallel arrays of numerators and denominators representing
/// fractions at that level. The leaf layer has N fractions; each subsequent layer
/// halves the count by pairwise fraction addition, until the root layer has 1.
pub struct SummationLayer<E: IsField> {
    pub numerators: Vec<FieldElement<E>>,
    pub denominators: Vec<FieldElement<E>>,
}

/// Build a summation tree from leaf fractions.
///
/// Takes N leaf fractions (as parallel numerator/denominator vectors) and
/// returns layers from leaves (index 0, size N) to root (last index, size 1).
/// Each layer is built by pairwise fraction addition:
///   parent_n = left_n * right_d + right_n * left_d
///   parent_d = left_d * right_d
///
/// # Panics
/// Panics if `leaf_numerators` and `leaf_denominators` have different lengths,
/// or if the length is not a power of 2.
pub fn build_summation_tree<E: IsField>(
    leaf_numerators: Vec<FieldElement<E>>,
    leaf_denominators: Vec<FieldElement<E>>,
) -> Vec<SummationLayer<E>> {
    let n = leaf_numerators.len();
    assert_eq!(
        n,
        leaf_denominators.len(),
        "numerators and denominators must have the same length"
    );
    assert!(n.is_power_of_two(), "number of leaves must be a power of 2");

    // Number of layers: log2(n) + 1 (leaves + intermediate + root)
    let num_layers = n.trailing_zeros() as usize + 1;
    let mut layers = Vec::with_capacity(num_layers);

    // Layer 0: the leaves themselves
    layers.push(SummationLayer {
        numerators: leaf_numerators,
        denominators: leaf_denominators,
    });

    // Build each subsequent layer by pairwise fraction addition
    for layer_idx in 1..num_layers {
        let prev = &layers[layer_idx - 1];
        let prev_len = prev.numerators.len();
        let new_len = prev_len / 2;

        let compute_pair = |i: usize| -> (FieldElement<E>, FieldElement<E>) {
            let left_n = &prev.numerators[2 * i];
            let left_d = &prev.denominators[2 * i];
            let right_n = &prev.numerators[2 * i + 1];
            let right_d = &prev.denominators[2 * i + 1];

            // Cross-multiply: (left_n * right_d + right_n * left_d) / (left_d * right_d)
            let parent_n = &(left_n * right_d) + &(right_n * left_d);
            let parent_d = left_d * right_d;
            (parent_n, parent_d)
        };

        #[cfg(feature = "parallel")]
        let (numerators, denominators): (Vec<_>, Vec<_>) = if new_len >= 256 {
            (0..new_len).into_par_iter().map(compute_pair).unzip()
        } else {
            (0..new_len).map(compute_pair).unzip()
        };

        #[cfg(not(feature = "parallel"))]
        let (numerators, denominators): (Vec<_>, Vec<_>) = (0..new_len).map(compute_pair).unzip();

        layers.push(SummationLayer {
            numerators,
            denominators,
        });
    }

    layers
}

/// Proof for a single GKR layer reduction.
///
/// Contains the sumcheck proof that reduces claims about a parent layer's MLEs
/// to claims about the children layer's MLEs, plus the four claimed evaluations
/// of the children's numerator/denominator MLEs at the reduced point.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct GkrLayerProof<E: IsField> {
    /// Sumcheck proof for the layer reduction (degree-3 round polynomials).
    pub sumcheck_proof: SumcheckProof<E>,
    /// Claimed evaluations at the children layer: [n_left, n_right, d_left, d_right].
    /// These are the children MLE values at the point (r', 0) and (r', 1) where
    /// r' is the sumcheck challenge point and the last coordinate selects left/right.
    pub child_claims: [FieldElement<E>; 4],
}

/// Complete GKR proof for a fractional summation tree.
///
/// Proves that the root of the summation tree has a specific value by
/// layer-by-layer reduction from the root to the leaves via fractional sumcheck.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct GkrProof<E: IsField> {
    /// The claimed sum at the root: numerator / denominator as a field element.
    pub claimed_sum: FieldElement<E>,
    /// One layer proof per reduction step (from root towards leaves).
    pub layer_proofs: Vec<GkrLayerProof<E>>,
}

/// Compute the equality polynomial evaluations eq(point, b) for all b in {0,1}^n.
///
/// The equality polynomial is defined as:
///   eq(x, y) = prod_{i=0}^{n-1} (x_i * y_i + (1 - x_i) * (1 - y_i))
///
/// For a fixed point r = (r_0, ..., r_{n-1}), this computes eq(r, b) for every
/// Boolean vector b, yielding 2^n values. Uses the standard butterfly/tensor
/// product construction:
///   - Start with [1]
///   - For each coordinate r_i, double the table:
///     existing entries are scaled by (1 - r_i), and new entries by r_i
///
/// Returns a vector of length 2^n in little-endian bit order.
pub fn compute_eq_evals<E: IsField>(point: &[FieldElement<E>]) -> Vec<FieldElement<E>> {
    let n = point.len();
    let size = 1 << n;
    let mut evals = Vec::with_capacity(size);
    evals.push(FieldElement::one());

    for r_i in point.iter() {
        let one_minus_ri = &FieldElement::<E>::one() - r_i;
        let prev_len = evals.len();
        // Extend: for each existing entry e, push e * r_i, then scale e by (1 - r_i)
        for j in 0..prev_len {
            let new_val = &evals[j] * r_i;
            evals.push(new_val);
        }
        // Scale existing entries by (1 - r_i)
        for eval in evals[..prev_len].iter_mut() {
            *eval = &*eval * &one_minus_ri;
        }
    }

    evals
}

/// Evaluate the MLE (multilinear extension) of a table at a given point.
///
/// Given evaluations `table[b]` for b in {0,1}^n, computes:
///   MLE(point) = sum_{b in {0,1}^n} table[b] * eq(point, b)
///
/// This is equivalent to multilinear interpolation of the table at the point.
#[cfg(test)]
fn evaluate_mle<E: IsField>(
    table: &[FieldElement<E>],
    point: &[FieldElement<E>],
) -> FieldElement<E> {
    let eq_evals = compute_eq_evals(point);
    assert_eq!(table.len(), eq_evals.len());
    table
        .iter()
        .zip(eq_evals.iter())
        .fold(FieldElement::zero(), |acc, (t, e)| &acc + &(t * e))
}

/// Run the GKR prover on a fractional summation tree.
///
/// Proves that the root of the tree has a specific numerator/denominator by
/// layer-by-layer reduction from the root to the leaves. At each layer, a
/// fractional sumcheck proves that the parent layer's claims are consistent
/// with the children layer via the gate equation:
///   parent_n(j) = child_n(2j) * child_d(2j+1) + child_n(2j+1) * child_d(2j)
///   parent_d(j) = child_d(2j) * child_d(2j+1)
///
/// The sumcheck at each layer operates on a degree-3 function (product of eq
/// weight with two child values), so round polynomials have degree 3 with
/// 4 evaluation points each.
///
/// # Arguments
/// - `tree`: the summation tree layers from leaves (index 0) to root (last index)
/// - `transcript`: Fiat-Shamir transcript for challenge sampling
///
/// # Returns
/// A `GkrProof` containing the claimed sum and layer proofs, plus the final
/// evaluation point and claims at the leaf layer.
///
/// # Panics
/// Panics if the tree is empty or has inconsistent layer sizes.
#[allow(clippy::type_complexity)]
pub fn gkr_prove<E: IsField>(
    tree: &[SummationLayer<E>],
    transcript: &mut impl IsTranscript<E>,
) -> Result<
    (
        GkrProof<E>,
        Vec<FieldElement<E>>,
        FieldElement<E>,
        FieldElement<E>,
    ),
    GkrError,
> {
    assert!(!tree.is_empty(), "tree must have at least one layer");

    let num_layers = tree.len(); // layers 0..num_layers-1, root is at num_layers-1
    let root = &tree[num_layers - 1];
    assert_eq!(
        root.numerators.len(),
        1,
        "root layer must have exactly 1 element"
    );

    let root_n = &root.numerators[0];
    let root_d = &root.denominators[0];

    // Compute claimed_sum = root_n / root_d. root_d = 0 means the LogUp challenge z
    // collided with a fingerprint denominator (probability ~1/p ≈ 2^{-64}); return
    // an error rather than panicking so the caller can retry with a fresh transcript.
    let root_d_inv = root_d.inv().map_err(|_| GkrError::ZeroDenominator)?;
    let claimed_sum = root_n * &root_d_inv;

    // Append the claimed sum to the transcript
    transcript.append_field_element(&claimed_sum);

    // If the tree has only 1 layer (just leaves = root), no reductions needed
    if num_layers == 1 {
        return Ok((
            GkrProof {
                claimed_sum,
                layer_proofs: vec![],
            },
            vec![],
            root_n.clone(),
            root_d.clone(),
        ));
    }

    let mut layer_proofs = Vec::with_capacity(num_layers - 1);
    let mut n_claim = root_n.clone();
    let mut d_claim = root_d.clone();
    let mut current_point: Vec<FieldElement<E>> = vec![];

    // Reduce from root towards leaves: for each layer l from (num_layers-2) down to 0,
    // the children layer is tree[l] and the parent is tree[l+1].
    for l in (0..num_layers - 1).rev() {
        let child_n = &tree[l].numerators;
        let child_d = &tree[l].denominators;
        let parent_size = child_n.len() / 2; // = tree[l+1].numerators.len()
        let parent_num_vars = parent_size.trailing_zeros() as usize;

        // Sample lambda to combine numerator and denominator claims
        let lambda: FieldElement<E> = transcript.sample_field_element();
        let combined_claim = &n_claim + &(&lambda * &d_claim);

        if parent_num_vars == 0 {
            // Trivial case: parent has 1 element (0 variables), no sumcheck needed.
            // The "sumcheck" is just a direct check: combined_claim must equal
            // child_n[0]*child_d[1] + child_n[1]*child_d[0] + lambda * child_d[0]*child_d[1]
            let nl = &child_n[0];
            let nr = &child_n[1];
            let dl = &child_d[0];
            let dr = &child_d[1];

            // Provide the 4 child claims (they are just the raw values since there's
            // no random point yet)
            let child_claims = [nl.clone(), nr.clone(), dl.clone(), dr.clone()];

            // Append child claims to transcript
            for claim in &child_claims {
                transcript.append_field_element(claim);
            }

            // Sample eta to combine left/right into new claims for next layer
            let eta: FieldElement<E> = transcript.sample_field_element();

            // New claims for the next layer down: fold left and right using eta
            // The children MLE at point (eta) for a 2-element table [a, b] is:
            //   a*(1-eta) + b*eta
            n_claim = &child_n[0] + &(&eta * &(&child_n[1] - &child_n[0]));
            d_claim = &child_d[0] + &(&eta * &(&child_d[1] - &child_d[0]));
            current_point = vec![eta];

            layer_proofs.push(GkrLayerProof {
                sumcheck_proof: SumcheckProof {
                    round_polys: vec![],
                },
                child_claims,
            });
        } else {
            // Non-trivial case: run the fractional sumcheck over parent_num_vars variables.
            //
            // The function to sum over {0,1}^parent_num_vars is:
            //   f(b) = eq(current_point, b) * [n_left(b)*d_right(b) + n_right(b)*d_left(b) + lambda*d_left(b)*d_right(b)]
            // where left/right are the even/odd children indexed by b.
            //
            // This function has degree 3 in each variable (product of eq * two child values),
            // so the round polynomial needs 4 evaluation points (degree 3).

            // Build the four gate "bookkeeping" tables for the sumcheck:
            // - nl_table: child_n[2b] (left numerators)
            // - nr_table: child_n[2b+1] (right numerators)
            // - dl_table: child_d[2b] (left denominators)
            // - dr_table: child_d[2b+1] (right denominators)
            let mut nl_table: Vec<FieldElement<E>> =
                (0..parent_size).map(|j| child_n[2 * j].clone()).collect();
            let mut nr_table: Vec<FieldElement<E>> = (0..parent_size)
                .map(|j| child_n[2 * j + 1].clone())
                .collect();
            let mut dl_table: Vec<FieldElement<E>> =
                (0..parent_size).map(|j| child_d[2 * j].clone()).collect();
            let mut dr_table: Vec<FieldElement<E>> = (0..parent_size)
                .map(|j| child_d[2 * j + 1].clone())
                .collect();

            let mut round_polys = Vec::with_capacity(parent_num_vars);
            let mut challenges = Vec::with_capacity(parent_num_vars);
            let mut round_combined_claim = combined_claim.clone();
            // Eq correction factor: accumulates eq(r_k, c_k) from previous rounds.
            // Instead of folding eq_table (N/2 multiplications per round), we halve
            // it (N/2 additions) and track the missing fold factors in this scalar.
            let mut eq_correction = FieldElement::<E>::one();

            // SVO (Split-Value Optimization, ePrint 2025/1117 Algorithm 5):
            // For large tables, split eq(w, x) into prefix and suffix halves to
            // reduce memory from 2^l to 2 * 2^{l/2}.
            //
            // eq(w, b) = eq_prefix(w_prefix, b_prefix) * eq_suffix(w_suffix, b_suffix)
            //
            // During the first suffix_len rounds, eq_suffix is halved each round
            // while eq_prefix stays fixed. The inner loop restructures as:
            //   h_raw(t) = Σ_s eq_suffix[s] * Σ_p eq_prefix[p] * gate(t, p*suffix_half + s)
            //
            // After suffix rounds, eq_suffix is absorbed into eq_correction, and
            // eq_prefix becomes the eq_table for the remaining prefix rounds.
            let use_svo = parent_num_vars >= SVO_THRESHOLD;

            if use_svo {
                // --- SVO path: split eq into prefix + suffix ---
                let suffix_len = parent_num_vars / 2;
                let _prefix_len = parent_num_vars - suffix_len;
                let mut eq_suffix = compute_eq_evals(&current_point[..suffix_len]);
                let eq_prefix = compute_eq_evals(&current_point[suffix_len..parent_num_vars]);
                let prefix_size = eq_prefix.len(); // = 2^prefix_len, constant during suffix rounds

                // Verify initial consistency using the split eq tables
                debug_assert!({
                    let mut check_sum = FieldElement::<E>::zero();
                    for j in 0..parent_size {
                        let suffix_idx = j & (eq_suffix.len() - 1);
                        let prefix_idx = j >> suffix_len;
                        let eq_val = &eq_suffix[suffix_idx] * &eq_prefix[prefix_idx];
                        let gate_val = &(&nl_table[j] * &dr_table[j])
                            + &(&nr_table[j] * &dl_table[j])
                            + &(&lambda * &(&dl_table[j] * &dr_table[j]));
                        check_sum = &check_sum + &(&eq_val * &gate_val);
                    }
                    check_sum == combined_claim
                });

                // Phase 1: Suffix rounds (first suffix_len rounds)
                // Process variables current_point[0..suffix_len].
                // eq_suffix is halved each round; eq_prefix stays fixed.
                #[allow(clippy::needless_range_loop)]
                for round_idx in 0..suffix_len {
                    let r_round = &current_point[round_idx];
                    let half = nl_table.len() / 2;
                    let suffix_half = eq_suffix.len() / 2;

                    // Pre-halve eq_suffix (same as Dao-Thaler halving for eq_table)
                    for j in 0..suffix_half {
                        eq_suffix[j] = &eq_suffix[2 * j] + &eq_suffix[2 * j + 1];
                    }
                    eq_suffix.truncate(suffix_half);

                    let one = FieldElement::<E>::one();

                    // Inner loop: for each suffix index, accumulate gate contributions
                    // weighted by eq_prefix, then weight by eq_suffix.
                    //
                    // h_raw(t) = Σ_s eq_suffix[s] * Σ_p eq_prefix[p] * gate(t, p*suffix_half + s)
                    let compute_suffix_contribution = |suffix_idx: usize| -> [FieldElement<E>; 2] {
                        let eq_s = &eq_suffix[suffix_idx];
                        let mut contrib_h0 = FieldElement::<E>::zero();
                        let mut contrib_h2 = FieldElement::<E>::zero();

                        #[allow(clippy::needless_range_loop)]
                        for prefix_idx in 0..prefix_size {
                            let j = prefix_idx * suffix_half + suffix_idx;
                            let eq_p = &eq_prefix[prefix_idx];

                            let nl_l = &nl_table[2 * j];
                            let nl_r = &nl_table[2 * j + 1];
                            let nr_l = &nr_table[2 * j];
                            let nr_r = &nr_table[2 * j + 1];
                            let dl_l = &dl_table[2 * j];
                            let dl_r = &dl_table[2 * j + 1];
                            let dr_l = &dr_table[2 * j];
                            let dr_r = &dr_table[2 * j + 1];

                            // t=0: gate = nl*dr + dl*(nr + lambda*dr)
                            let gate_0 = &(nl_l * dr_l) + &(dl_l * &(nr_l + &(&lambda * dr_l)));
                            contrib_h0 = &contrib_h0 + &(eq_p * &gate_0);

                            // t=2: val = 2*right - left
                            let nl_2 = &(nl_r + nl_r) - nl_l;
                            let nr_2 = &(nr_r + nr_r) - nr_l;
                            let dl_2 = &(dl_r + dl_r) - dl_l;
                            let dr_2 = &(dr_r + dr_r) - dr_l;
                            let gate_2 =
                                &(&nl_2 * &dr_2) + &(&dl_2 * &(&nr_2 + &(&lambda * &dr_2)));
                            contrib_h2 = &contrib_h2 + &(eq_p * &gate_2);
                        }

                        [eq_s * &contrib_h0, eq_s * &contrib_h2]
                    };

                    let zero2 = || [FieldElement::<E>::zero(), FieldElement::<E>::zero()];
                    let add2 = |a: [FieldElement<E>; 2], b: [FieldElement<E>; 2]| {
                        [&a[0] + &b[0], &a[1] + &b[1]]
                    };

                    #[cfg(feature = "parallel")]
                    let totals: [FieldElement<E>; 2] = if suffix_half >= 256 {
                        (0..suffix_half)
                            .into_par_iter()
                            .fold(zero2, |acc, s| add2(acc, compute_suffix_contribution(s)))
                            .reduce(zero2, add2)
                    } else {
                        (0..suffix_half)
                            .fold(zero2(), |acc, s| add2(acc, compute_suffix_contribution(s)))
                    };

                    #[cfg(not(feature = "parallel"))]
                    let totals: [FieldElement<E>; 2] = (0..suffix_half)
                        .fold(zero2(), |acc, s| add2(acc, compute_suffix_contribution(s)));

                    // Phase 2: Recover S(t) from h(t) and eq_round(r, t).
                    let [raw_h0, raw_h2] = totals;
                    let total_h0 = &eq_correction * &raw_h0;
                    let total_h2 = &eq_correction * &raw_h2;

                    let one_minus_r = &one - r_round;
                    let s0 = &one_minus_r * &total_h0;
                    let s1 = &round_combined_claim - &s0;

                    let r_inv = r_round
                        .inv()
                        .expect("r_round = 0 is probability 2^{-64} for random challenges");
                    let h1 = &s1 * &r_inv;

                    let three = FieldElement::<E>::from(3u64);
                    let h3 = &(&(&three * &total_h2) - &(&three * &h1)) + &total_h0;

                    let eq_at_2 = &(&three * r_round) - &one;
                    let s2 = &eq_at_2 * &total_h2;

                    let eq_at_3 = &(&FieldElement::<E>::from(5u64) * r_round)
                        - &FieldElement::<E>::from(2u64);
                    let s3 = &eq_at_3 * &h3;

                    let poly_evals = vec![s0, s1, s2, s3];
                    let round_poly = RoundPoly::new(poly_evals);

                    for eval in round_poly.evals() {
                        transcript.append_field_element(eval);
                    }

                    let challenge: FieldElement<E> = transcript.sample_field_element();
                    round_combined_claim = round_poly.evaluate(&challenge);

                    let eq_update =
                        &(r_round * &challenge) + &(&one_minus_r * &(&one - &challenge));
                    eq_correction = &eq_correction * &eq_update;

                    // Fold the four gate tables
                    let fold_table = |table: &mut Vec<FieldElement<E>>| {
                        #[cfg(feature = "parallel")]
                        if half >= 256 {
                            let folded: Vec<FieldElement<E>> = table
                                .par_chunks(2)
                                .map(|pair| &pair[0] + &(&challenge * &(&pair[1] - &pair[0])))
                                .collect();
                            *table = folded;
                            return;
                        }
                        for j in 0..half {
                            let left = &table[2 * j];
                            let right = &table[2 * j + 1];
                            table[j] = left + &(&challenge * &(right - left));
                        }
                        table.truncate(half);
                    };

                    fold_table(&mut nl_table);
                    fold_table(&mut nr_table);
                    fold_table(&mut dl_table);
                    fold_table(&mut dr_table);

                    round_polys.push(round_poly);
                    challenges.push(challenge);
                }

                // Transition: absorb the remaining eq_suffix scalar into eq_correction.
                // After suffix_len halvings, eq_suffix has been reduced to a single entry.
                debug_assert_eq!(eq_suffix.len(), 1);
                eq_correction = &eq_correction * &eq_suffix[0];

                // Phase 2: Prefix rounds (remaining prefix_len rounds)
                // Use eq_prefix as the eq_table for standard Dao-Thaler halving.
                let mut eq_table = eq_prefix;

                for r_round in current_point.iter().take(parent_num_vars).skip(suffix_len) {
                    let half = nl_table.len() / 2;
                    let one = FieldElement::<E>::one();

                    // Pre-halve eq_table
                    for j in 0..half {
                        eq_table[j] = &eq_table[2 * j] + &eq_table[2 * j + 1];
                    }
                    eq_table.truncate(half);

                    let compute_pair_sums = |j: usize| -> [FieldElement<E>; 2] {
                        let eq_rem = &eq_table[j];

                        let nl_l = &nl_table[2 * j];
                        let nl_r = &nl_table[2 * j + 1];
                        let nr_l = &nr_table[2 * j];
                        let nr_r = &nr_table[2 * j + 1];
                        let dl_l = &dl_table[2 * j];
                        let dl_r = &dl_table[2 * j + 1];
                        let dr_l = &dr_table[2 * j];
                        let dr_r = &dr_table[2 * j + 1];

                        let gate_0 = &(nl_l * dr_l) + &(dl_l * &(nr_l + &(&lambda * dr_l)));
                        let h0 = eq_rem * &gate_0;

                        let nl_2 = &(nl_r + nl_r) - nl_l;
                        let nr_2 = &(nr_r + nr_r) - nr_l;
                        let dl_2 = &(dl_r + dl_r) - dl_l;
                        let dr_2 = &(dr_r + dr_r) - dr_l;
                        let gate_2 = &(&nl_2 * &dr_2) + &(&dl_2 * &(&nr_2 + &(&lambda * &dr_2)));
                        let h2 = eq_rem * &gate_2;

                        [h0, h2]
                    };

                    let zero2 = || [FieldElement::<E>::zero(), FieldElement::<E>::zero()];
                    let add2 = |a: [FieldElement<E>; 2], b: [FieldElement<E>; 2]| {
                        [&a[0] + &b[0], &a[1] + &b[1]]
                    };

                    #[cfg(feature = "parallel")]
                    let totals: [FieldElement<E>; 2] = if half >= 256 {
                        (0..half)
                            .into_par_iter()
                            .fold(zero2, |acc, j| add2(acc, compute_pair_sums(j)))
                            .reduce(zero2, add2)
                    } else {
                        (0..half).fold(zero2(), |acc, j| add2(acc, compute_pair_sums(j)))
                    };

                    #[cfg(not(feature = "parallel"))]
                    let totals: [FieldElement<E>; 2] =
                        (0..half).fold(zero2(), |acc, j| add2(acc, compute_pair_sums(j)));

                    // Phase 2: Recover S(t) from h(t) and eq_round(r, t).
                    let [raw_h0, raw_h2] = totals;
                    let total_h0 = &eq_correction * &raw_h0;
                    let total_h2 = &eq_correction * &raw_h2;

                    let one_minus_r = &one - r_round;
                    let s0 = &one_minus_r * &total_h0;
                    let s1 = &round_combined_claim - &s0;

                    let r_inv = r_round
                        .inv()
                        .expect("r_round = 0 is probability 2^{-64} for random challenges");
                    let h1 = &s1 * &r_inv;

                    let three = FieldElement::<E>::from(3u64);
                    let h3 = &(&(&three * &total_h2) - &(&three * &h1)) + &total_h0;

                    let eq_at_2 = &(&three * r_round) - &one;
                    let s2 = &eq_at_2 * &total_h2;

                    let eq_at_3 = &(&FieldElement::<E>::from(5u64) * r_round)
                        - &FieldElement::<E>::from(2u64);
                    let s3 = &eq_at_3 * &h3;

                    let poly_evals = vec![s0, s1, s2, s3];
                    let round_poly = RoundPoly::new(poly_evals);

                    for eval in round_poly.evals() {
                        transcript.append_field_element(eval);
                    }

                    let challenge: FieldElement<E> = transcript.sample_field_element();
                    round_combined_claim = round_poly.evaluate(&challenge);

                    let eq_update =
                        &(r_round * &challenge) + &(&one_minus_r * &(&one - &challenge));
                    eq_correction = &eq_correction * &eq_update;

                    let fold_table = |table: &mut Vec<FieldElement<E>>| {
                        #[cfg(feature = "parallel")]
                        if half >= 256 {
                            let folded: Vec<FieldElement<E>> = table
                                .par_chunks(2)
                                .map(|pair| &pair[0] + &(&challenge * &(&pair[1] - &pair[0])))
                                .collect();
                            *table = folded;
                            return;
                        }
                        for j in 0..half {
                            let left = &table[2 * j];
                            let right = &table[2 * j + 1];
                            table[j] = left + &(&challenge * &(right - left));
                        }
                        table.truncate(half);
                    };

                    fold_table(&mut nl_table);
                    fold_table(&mut nr_table);
                    fold_table(&mut dl_table);
                    fold_table(&mut dr_table);

                    round_polys.push(round_poly);
                    challenges.push(challenge);
                }
            } else {
                // --- Standard path for small tables (parent_num_vars < SVO_THRESHOLD) ---
                let mut eq_table = compute_eq_evals(&current_point);

                // Verify initial consistency
                debug_assert!({
                    let mut check_sum = FieldElement::<E>::zero();
                    for j in 0..parent_size {
                        let gate_val = &(&nl_table[j] * &dr_table[j])
                            + &(&nr_table[j] * &dl_table[j])
                            + &(&lambda * &(&dl_table[j] * &dr_table[j]));
                        check_sum = &check_sum + &(&eq_table[j] * &gate_val);
                    }
                    check_sum == combined_claim
                });

                for r_round in current_point.iter().take(parent_num_vars) {
                    let half = nl_table.len() / 2;
                    let one = FieldElement::<E>::one();

                    // Pre-halve eq_table
                    for j in 0..half {
                        eq_table[j] = &eq_table[2 * j] + &eq_table[2 * j + 1];
                    }
                    eq_table.truncate(half);

                    let compute_pair_sums = |j: usize| -> [FieldElement<E>; 2] {
                        let eq_rem = &eq_table[j];

                        let nl_l = &nl_table[2 * j];
                        let nl_r = &nl_table[2 * j + 1];
                        let nr_l = &nr_table[2 * j];
                        let nr_r = &nr_table[2 * j + 1];
                        let dl_l = &dl_table[2 * j];
                        let dl_r = &dl_table[2 * j + 1];
                        let dr_l = &dr_table[2 * j];
                        let dr_r = &dr_table[2 * j + 1];

                        let gate_0 = &(nl_l * dr_l) + &(dl_l * &(nr_l + &(&lambda * dr_l)));
                        let h0 = eq_rem * &gate_0;

                        let nl_2 = &(nl_r + nl_r) - nl_l;
                        let nr_2 = &(nr_r + nr_r) - nr_l;
                        let dl_2 = &(dl_r + dl_r) - dl_l;
                        let dr_2 = &(dr_r + dr_r) - dr_l;
                        let gate_2 = &(&nl_2 * &dr_2) + &(&dl_2 * &(&nr_2 + &(&lambda * &dr_2)));
                        let h2 = eq_rem * &gate_2;

                        [h0, h2]
                    };

                    let zero2 = || [FieldElement::<E>::zero(), FieldElement::<E>::zero()];
                    let add2 = |a: [FieldElement<E>; 2], b: [FieldElement<E>; 2]| {
                        [&a[0] + &b[0], &a[1] + &b[1]]
                    };

                    #[cfg(feature = "parallel")]
                    let totals: [FieldElement<E>; 2] = if half >= 256 {
                        (0..half)
                            .into_par_iter()
                            .fold(zero2, |acc, j| add2(acc, compute_pair_sums(j)))
                            .reduce(zero2, add2)
                    } else {
                        (0..half).fold(zero2(), |acc, j| add2(acc, compute_pair_sums(j)))
                    };

                    #[cfg(not(feature = "parallel"))]
                    let totals: [FieldElement<E>; 2] =
                        (0..half).fold(zero2(), |acc, j| add2(acc, compute_pair_sums(j)));

                    let [raw_h0, raw_h2] = totals;
                    let total_h0 = &eq_correction * &raw_h0;
                    let total_h2 = &eq_correction * &raw_h2;

                    let one_minus_r = &one - r_round;
                    let s0 = &one_minus_r * &total_h0;
                    let s1 = &round_combined_claim - &s0;

                    let r_inv = r_round
                        .inv()
                        .expect("r_round = 0 is probability 2^{-64} for random challenges");
                    let h1 = &s1 * &r_inv;

                    let three = FieldElement::<E>::from(3u64);
                    let h3 = &(&(&three * &total_h2) - &(&three * &h1)) + &total_h0;

                    let eq_at_2 = &(&three * r_round) - &one;
                    let s2 = &eq_at_2 * &total_h2;

                    let eq_at_3 = &(&FieldElement::<E>::from(5u64) * r_round)
                        - &FieldElement::<E>::from(2u64);
                    let s3 = &eq_at_3 * &h3;

                    let poly_evals = vec![s0, s1, s2, s3];
                    let round_poly = RoundPoly::new(poly_evals);

                    for eval in round_poly.evals() {
                        transcript.append_field_element(eval);
                    }

                    let challenge: FieldElement<E> = transcript.sample_field_element();
                    round_combined_claim = round_poly.evaluate(&challenge);

                    let eq_update =
                        &(r_round * &challenge) + &(&one_minus_r * &(&one - &challenge));
                    eq_correction = &eq_correction * &eq_update;

                    let fold_table = |table: &mut Vec<FieldElement<E>>| {
                        #[cfg(feature = "parallel")]
                        if half >= 256 {
                            let folded: Vec<FieldElement<E>> = table
                                .par_chunks(2)
                                .map(|pair| &pair[0] + &(&challenge * &(&pair[1] - &pair[0])))
                                .collect();
                            *table = folded;
                            return;
                        }
                        for j in 0..half {
                            let left = &table[2 * j];
                            let right = &table[2 * j + 1];
                            table[j] = left + &(&challenge * &(right - left));
                        }
                        table.truncate(half);
                    };

                    fold_table(&mut nl_table);
                    fold_table(&mut nr_table);
                    fold_table(&mut dl_table);
                    fold_table(&mut dr_table);

                    round_polys.push(round_poly);
                    challenges.push(challenge);
                }
            }

            // After all rounds, each table has a single entry: the MLE evaluated
            // at the sumcheck challenge point.
            let child_claims = [
                nl_table[0].clone(),
                nr_table[0].clone(),
                dl_table[0].clone(),
                dr_table[0].clone(),
            ];

            // Append child claims to transcript
            for claim in &child_claims {
                transcript.append_field_element(claim);
            }

            // Sample eta to fold left/right children into new claims
            let eta: FieldElement<E> = transcript.sample_field_element();

            // New claims: fold left and right using eta
            // n_claim = nl + eta*(nr - nl), d_claim = dl + eta*(dr - dl)
            n_claim = &child_claims[0] + &(&eta * &(&child_claims[1] - &child_claims[0]));
            d_claim = &child_claims[2] + &(&eta * &(&child_claims[3] - &child_claims[2]));

            // Update current_point: eta corresponds to x_0 (the even/odd selector),
            // followed by the sumcheck challenges for the remaining parent variables.
            // In little-endian convention: point[i] = value of x_i.
            let mut new_point = Vec::with_capacity(challenges.len() + 1);
            new_point.push(eta);
            new_point.extend(challenges);
            current_point = new_point;

            layer_proofs.push(GkrLayerProof {
                sumcheck_proof: SumcheckProof { round_polys },
                child_claims,
            });
        }
    }

    Ok((
        GkrProof {
            claimed_sum,
            layer_proofs,
        },
        current_point,
        n_claim,
        d_claim,
    ))
}

/// Errors that can occur during GKR proving or verification.
#[derive(Debug, Clone)]
pub enum GkrError {
    /// The summation tree structure is invalid.
    InvalidTree { reason: String },
    /// A sumcheck round failed verification.
    SumcheckFailed { layer: usize, reason: String },
    /// The gate equation check failed at a layer.
    GateCheckFailed { layer: usize },
    /// The claimed sum does not match (unused in the verifier itself,
    /// but available for callers that compare the claimed sum to an external value).
    ClaimedSumMismatch,
    /// The root denominator is zero (LogUp challenge z collided with a fingerprint
    /// denominator). Probability ~1/p ≈ 2^{-64}; prover should sample a new transcript.
    ZeroDenominator,
}

impl fmt::Display for GkrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GkrError::InvalidTree { reason } => {
                write!(f, "invalid GKR tree: {}", reason)
            }
            GkrError::SumcheckFailed { layer, reason } => {
                write!(f, "sumcheck failed at layer {}: {}", layer, reason)
            }
            GkrError::GateCheckFailed { layer } => {
                write!(f, "gate check failed at layer {}", layer)
            }
            GkrError::ClaimedSumMismatch => {
                write!(f, "claimed sum mismatch")
            }
            GkrError::ZeroDenominator => {
                write!(
                    f,
                    "GKR root denominator is zero (LogUp challenge z collided with a fingerprint denominator; probability ~1/p ≈ 2^{{-64}})"
                )
            }
        }
    }
}

/// Verify a GKR proof for a fractional summation tree.
///
/// Replays the Fiat-Shamir transcript identically to `gkr_prove` and checks:
/// - At each non-trivial layer: sumcheck round consistency (p(0)+p(1) = current_sum)
///   and the gate equation at the final evaluation point.
/// - At trivial layers (0-variable parent): no sumcheck check; soundness is
///   enforced by subsequent layers.
///
/// # Arguments
/// - `proof`: the GKR proof produced by `gkr_prove`
/// - `transcript`: Fiat-Shamir transcript (must use the same seed as the prover)
///
/// # Returns
/// `Ok((final_point, n_claim, d_claim))` where `final_point` is the random
/// evaluation point at the leaf layer, and `n_claim`/`d_claim` are the claimed
/// MLE evaluations of the leaf numerators/denominators at that point.
#[allow(clippy::type_complexity)]
pub fn gkr_verify<E: IsField>(
    proof: &GkrProof<E>,
    transcript: &mut impl IsTranscript<E>,
) -> Result<(Vec<FieldElement<E>>, FieldElement<E>, FieldElement<E>), GkrError> {
    // Step 1: Append claimed_sum to transcript (mirrors prover line 239)
    transcript.append_field_element(&proof.claimed_sum);

    // If there are no layer proofs, the tree had a single leaf (root = leaf).
    // Return empty point and the claimed_sum as n_claim with d_claim = 1.
    if proof.layer_proofs.is_empty() {
        return Ok((vec![], proof.claimed_sum.clone(), FieldElement::one()));
    }

    // Step 2: Initialize claims.
    // The verifier sets n_claim = claimed_sum, d_claim = 1.
    // This represents the same rational value as root_n/root_d.
    // Trivial layers (0 sumcheck rounds) are gate-checked directly in the loop below.
    let mut n_claim = proof.claimed_sum.clone();
    let mut d_claim = FieldElement::<E>::one();
    let mut current_point: Vec<FieldElement<E>> = vec![];

    for (layer_idx, layer_proof) in proof.layer_proofs.iter().enumerate() {
        // Step 4: Sample lambda (mirrors prover line 268)
        let lambda: FieldElement<E> = transcript.sample_field_element();

        // Step 5: combined_claim = n_claim + lambda * d_claim
        let combined_claim = &n_claim + &(&lambda * &d_claim);

        let round_polys = &layer_proof.sumcheck_proof.round_polys;

        if round_polys.is_empty() {
            // Trivial layer (0 variables in parent): no sumcheck rounds.
            // Gate check: verify that n_claim * (dl·dr) = nl·dr + nr·dl.
            //
            // The verifier works in normalized form: n_claim = root_n/root_d, d_claim = 1.
            // The prover's gate equation root_n + λ·root_d = nl·dr + nr·dl + λ·dl·dr,
            // divided by root_d (= dl·dr), becomes n_claim·(dl·dr) = nl·dr + nr·dl
            // (the λ terms cancel). This binds claimed_sum to the actual tree structure.
            let [ref nl, ref nr, ref dl, ref dr] = layer_proof.child_claims;
            let lhs = &n_claim * &(dl * dr);
            let rhs = &(nl * dr) + &(nr * dl);
            if lhs != rhs {
                return Err(GkrError::GateCheckFailed { layer: layer_idx });
            }
        } else {
            // Non-trivial layer: verify sumcheck inline.
            let num_rounds = round_polys.len();
            let mut current_sum = combined_claim;
            let mut challenges = Vec::with_capacity(num_rounds);

            for (round, round_poly) in round_polys.iter().enumerate() {
                // Check p(0) + p(1) == current_sum
                if round_poly.sum_at_binary() != current_sum {
                    return Err(GkrError::SumcheckFailed {
                        layer: layer_idx,
                        reason: format!("round {} sum mismatch: p(0)+p(1) != expected sum", round),
                    });
                }

                // Append round poly evals to transcript (mirrors prover lines 388-389)
                for eval in round_poly.evals() {
                    transcript.append_field_element(eval);
                }

                // Sample challenge (mirrors prover line 393)
                let challenge: FieldElement<E> = transcript.sample_field_element();

                // Update current_sum to p(challenge)
                current_sum = round_poly.evaluate(&challenge);

                challenges.push(challenge);
            }

            // Gate check: verify that the final sumcheck evaluation equals
            // eq(current_point, challenges) * gate(child_claims, lambda)
            let [ref nl, ref nr, ref dl, ref dr] = layer_proof.child_claims;

            // Compute eq(current_point, challenges) as a single field element.
            // eq(a, b) = prod_i (a_i*b_i + (1-a_i)*(1-b_i))
            let eq_val = compute_eq_at_point(&current_point, &challenges);

            // gate_combined = nl*dr + nr*dl + lambda*dl*dr
            let gate_combined = &(&(nl * dr) + &(nr * dl)) + &(&lambda * &(dl * dr));

            let expected = &eq_val * &gate_combined;

            if current_sum != expected {
                return Err(GkrError::GateCheckFailed { layer: layer_idx });
            }

            // Build the new current_point from eta (below) and sumcheck challenges.
            // We need to store challenges for constructing current_point after eta is sampled.
            // Store them temporarily.
            // (We'll construct the point after sampling eta below.)

            // Append child claims to transcript (mirrors prover lines 431-433)
            for claim in &layer_proof.child_claims {
                transcript.append_field_element(claim);
            }

            // Sample eta (mirrors prover line 436)
            let eta: FieldElement<E> = transcript.sample_field_element();

            // Update claims: fold left/right using eta
            n_claim = nl + &(&eta * &(nr - nl));
            d_claim = dl + &(&eta * &(dr - dl));

            // Update current_point = [eta] ++ challenges (mirrors prover lines 447-450)
            let mut new_point = Vec::with_capacity(challenges.len() + 1);
            new_point.push(eta);
            new_point.extend(challenges);
            current_point = new_point;

            continue;
        }

        // Trivial layer path: append child claims and sample eta
        // (mirrors prover lines 285-290)
        for claim in &layer_proof.child_claims {
            transcript.append_field_element(claim);
        }
        let eta: FieldElement<E> = transcript.sample_field_element();

        let [ref nl, ref nr, ref dl, ref dr] = layer_proof.child_claims;
        n_claim = nl + &(&eta * &(nr - nl));
        d_claim = dl + &(&eta * &(dr - dl));
        current_point = vec![eta.clone()];
    }

    Ok((current_point, n_claim, d_claim))
}

// =============================================================================
// Batch GKR prover
// =============================================================================

/// Run the batch GKR prover on multiple instances (each a Vec<Layer>).
///
/// All instances share one sumcheck per layer via random linear combination
/// (sumcheck_alpha). Instances can have different numbers of layers (different
/// trace lengths). Smaller instances start participating at later layers.
///
/// # Arguments
/// - `layers_per_instance`: for each instance, its layer tree from leaves (index 0)
///   to root (last index), as produced by `gen_layers`.
/// - `transcript`: Fiat-Shamir transcript for challenge sampling
///
/// # Returns
/// A `BatchGkrProof`, the shared `random_point`, and per-instance `(n_claim, d_claim)`.
#[allow(clippy::type_complexity)]
pub fn gkr_prove_batch<E: IsField>(
    layers_per_instance: Vec<Vec<Layer<E>>>,
    transcript: &mut impl IsTranscript<E>,
) -> (
    BatchGkrProof<E>,
    Vec<FieldElement<E>>,
    Vec<(FieldElement<E>, FieldElement<E>)>,
) {
    let n_instances = layers_per_instance.len();

    // Domain separation
    transcript.append_bytes(b"gkr_batch");
    transcript.append_bytes(&(n_instances as u64).to_le_bytes());

    if n_instances == 0 {
        return (
            BatchGkrProof {
                root_claims: vec![],
                layer_proofs: vec![],
            },
            vec![],
            vec![],
        );
    }

    // n_layers_by_instance[i] = number of tree layers - 1 = number of reduction steps
    let n_layers_by_instance: Vec<usize> = layers_per_instance
        .iter()
        .map(|layers| layers.len() - 1)
        .collect();
    let max_layers = *n_layers_by_instance.iter().max().unwrap();

    // Extract root (numerator, denominator) for each instance
    let root_claims: Vec<(FieldElement<E>, FieldElement<E>)> = layers_per_instance
        .iter()
        .map(|layers| {
            let root = layers.last().unwrap();
            root.try_into_output_values()
                .expect("root layer must be output")
        })
        .collect();

    // Track per-instance state
    // n_claim[i], d_claim[i] start as root values and get updated each layer
    let mut n_claims: Vec<Option<FieldElement<E>>> = vec![None; n_instances];
    let mut d_claims: Vec<Option<FieldElement<E>>> = vec![None; n_instances];

    let mut current_point: Vec<FieldElement<E>> = vec![];
    let mut layer_proofs = Vec::with_capacity(max_layers);

    for layer in 0..max_layers {
        let n_remaining = max_layers - layer;

        // Detect output layers: instances whose tree has exactly n_remaining reduction steps
        // start participating now (their root is the output layer).
        // Use actual (root_n, root_d) so the gate check at the end of the sumcheck
        // matches: gate(children) = nl*dr + nr*dl + lambda*dl*dr uses the true root values.
        for i in 0..n_instances {
            if n_layers_by_instance[i] == n_remaining {
                n_claims[i] = Some(root_claims[i].0.clone());
                d_claims[i] = Some(root_claims[i].1.clone());
            }
        }

        // Append active claims to transcript (matches verifier)
        for i in 0..n_instances {
            if let (Some(n), Some(d)) = (&n_claims[i], &d_claims[i]) {
                transcript.append_field_element(n);
                transcript.append_field_element(d);
            }
        }

        // Sample randomness
        let sumcheck_alpha: FieldElement<E> = transcript.sample_field_element();
        let lambda: FieldElement<E> = transcript.sample_field_element();

        // Collect active instances (those that have claims and have reduction steps remaining)
        let mut active_instances: Vec<usize> = Vec::new();
        let mut combined_claims: Vec<FieldElement<E>> = Vec::new();

        for i in 0..n_instances {
            if n_claims[i].is_some() && n_layers_by_instance[i] > 0 {
                active_instances.push(i);
                let n = n_claims[i].as_ref().unwrap();
                let d = d_claims[i].as_ref().unwrap();
                let claim = n + &(&lambda * d);

                // Apply doubling factor for instances with fewer layers
                let n_unused = max_layers - n_layers_by_instance[i];
                if n_unused > 0 {
                    let doubling = FieldElement::<E>::from(1u64 << n_unused);
                    combined_claims.push(&claim * &doubling);
                } else {
                    combined_claims.push(claim);
                }
            }
        }

        if active_instances.is_empty() {
            break;
        }

        // The child layer index for instance i: layers[tree_layer_idx] where
        // tree_layer_idx = n_layers_by_instance[i] - 1 - (layer - n_unused)
        // Equivalently: layers[n_remaining - 1 - n_unused] but let's compute directly.

        // Compute the max parent_num_vars across active instances
        // (this determines the sumcheck dimension for this batch layer)
        let parent_num_vars_by_instance: Vec<usize> = active_instances
            .iter()
            .map(|&i| {
                let tree_layer_idx = n_remaining - 1;
                // tree_layer_idx is the child layer; the parent has half the size
                // child has 2^k elements → parent has 2^{k-1} → parent_num_vars = k-1
                let child_n_vars = layers_per_instance[i][tree_layer_idx].n_variables();
                debug_assert!(
                    child_n_vars >= 1,
                    "child of a non-output layer must have >= 2 elements"
                );
                child_n_vars - 1
            })
            .collect();
        let max_parent_vars = *parent_num_vars_by_instance.iter().max().unwrap();

        if max_parent_vars == 0 {
            // All active instances have trivial layers (0 variables in parent).
            // No sumcheck needed — provide child claims directly.
            let mut child_claims_by_instance = Vec::new();

            for (idx, &i) in active_instances.iter().enumerate() {
                let tree_layer_idx = n_remaining - 1;
                let child = &layers_per_instance[i][tree_layer_idx];

                let (child_n, child_d) = match child {
                    Layer::LogUpGeneric {
                        numerators,
                        denominators,
                    } => (numerators.as_slice(), denominators.as_slice()),
                    Layer::LogUpSingles { denominators } => {
                        // For singles at size 2: numerators are [1, 1]
                        // We need to handle this specially. Since the child has 2 elements,
                        // and numerators are implicit 1s, we need explicit values.
                        // Actually, if the leaf is Singles with 2 elements, after next_layer
                        // it becomes Generic. But at the leaf level with 2 elements,
                        // the tree has 2 layers (leaf + root). So the "child layer" could
                        // be Singles with 2 elements. Let's handle it.
                        // For the trivial case, we just provide the claims directly.
                        // n_left=1, n_right=1, d_left=d[0], d_right=d[1]
                        child_claims_by_instance.push([
                            FieldElement::one(),
                            FieldElement::one(),
                            denominators[0].clone(),
                            denominators[1].clone(),
                        ]);
                        continue;
                    }
                };

                let _ = idx; // suppress unused warning
                child_claims_by_instance.push([
                    child_n[0].clone(),
                    child_n[1].clone(),
                    child_d[0].clone(),
                    child_d[1].clone(),
                ]);
            }

            // Append child claims to transcript
            for claims in &child_claims_by_instance {
                for claim in claims {
                    transcript.append_field_element(claim);
                }
            }

            // Sample eta to fold left/right
            let eta: FieldElement<E> = transcript.sample_field_element();

            // Update per-instance claims
            for (idx, &i) in active_instances.iter().enumerate() {
                let [ref nl, ref nr, ref dl, ref dr] = child_claims_by_instance[idx];
                n_claims[i] = Some(nl + &(&eta * &(nr - nl)));
                d_claims[i] = Some(dl + &(&eta * &(dr - dl)));
            }

            current_point = vec![eta];

            layer_proofs.push(BatchGkrLayerProof {
                sumcheck_proof: SumcheckProof {
                    round_polys: vec![],
                },
                child_claims_by_instance,
            });
        } else {
            // Non-trivial case: run shared sumcheck over max_parent_vars variables.
            // For each active instance, build bookkeeping tables.

            // We run the sumcheck manually, combining round polynomials across instances.
            let mut per_instance_tables: Vec<PerInstanceTables<E>> = active_instances
                .iter()
                .map(|&i| {
                    let tree_layer_idx = n_remaining - 1;
                    let child = &layers_per_instance[i][tree_layer_idx];

                    let parent_size = match child {
                        Layer::LogUpGeneric { denominators, .. }
                        | Layer::LogUpSingles { denominators } => denominators.len() / 2,
                    };

                    let (nl_table, nr_table, is_singles) = match child {
                        Layer::LogUpGeneric { numerators, .. } => {
                            let nl: Vec<_> = (0..parent_size)
                                .map(|j| numerators[2 * j].clone())
                                .collect();
                            let nr: Vec<_> = (0..parent_size)
                                .map(|j| numerators[2 * j + 1].clone())
                                .collect();
                            (nl, nr, false)
                        }
                        Layer::LogUpSingles { .. } => {
                            // Singles: numerators are all 1
                            (vec![], vec![], true)
                        }
                    };

                    let denominators = match child {
                        Layer::LogUpGeneric { denominators, .. }
                        | Layer::LogUpSingles { denominators } => denominators,
                    };

                    let dl_table: Vec<_> = (0..parent_size)
                        .map(|j| denominators[2 * j].clone())
                        .collect();
                    let dr_table: Vec<_> = (0..parent_size)
                        .map(|j| denominators[2 * j + 1].clone())
                        .collect();

                    let my_parent_num_vars = parent_num_vars_by_instance
                        [active_instances.iter().position(|&x| x == i).unwrap()];
                    // current_point must have at least parent_num_vars coordinates
                    // (it grows by max_parent_vars+1 each layer, matching the tree doubling).
                    debug_assert!(
                        current_point.len() >= my_parent_num_vars,
                        "current_point.len()={} < parent_num_vars={}",
                        current_point.len(),
                        my_parent_num_vars
                    );
                    let inst_point = instance_eval_point(&current_point, my_parent_num_vars);
                    let use_svo = my_parent_num_vars >= SVO_THRESHOLD;
                    let svo_suffix_len = if use_svo { my_parent_num_vars / 2 } else { 0 };

                    let (eq_table, eq_prefix, eq_suffix) = if use_svo {
                        let suffix = compute_eq_evals(&inst_point[..svo_suffix_len]);
                        let prefix =
                            compute_eq_evals(&inst_point[svo_suffix_len..my_parent_num_vars]);
                        (Vec::new(), prefix, suffix)
                    } else {
                        (compute_eq_evals(&inst_point), Vec::new(), Vec::new())
                    };

                    PerInstanceTables {
                        nl_table,
                        nr_table,
                        dl_table,
                        dr_table,
                        eq_table,
                        eq_correction: FieldElement::one(),
                        is_singles,
                        parent_num_vars: my_parent_num_vars,
                        instance_point: inst_point,
                        use_svo,
                        svo_suffix_len,
                        eq_prefix,
                        eq_suffix,
                    }
                })
                .collect();

            let mut round_polys = Vec::with_capacity(max_parent_vars);
            let mut challenges = Vec::with_capacity(max_parent_vars);
            // Combined claim across instances (via sumcheck_alpha)
            let mut _round_combined_claim = {
                let mut sum = FieldElement::<E>::zero();
                let mut alpha_pow = FieldElement::<E>::one();
                for claim in &combined_claims {
                    sum = &sum + &(&alpha_pow * claim);
                    alpha_pow = &alpha_pow * &sumcheck_alpha;
                }
                sum
            };

            // Per-instance round polynomial evals, used to update combined_claims
            // via O(1) interpolation instead of O(N) table scan.
            let mut per_instance_evals: Vec<[FieldElement<E>; 4]> = vec![
                [
                    FieldElement::zero(),
                    FieldElement::zero(),
                    FieldElement::zero(),
                    FieldElement::zero()
                ];
                per_instance_tables.len()
            ];

            for round_idx in 0..max_parent_vars {
                // For each active instance, compute the round poly contribution
                let mut batch_s0 = FieldElement::<E>::zero();
                let mut batch_s1 = FieldElement::<E>::zero();
                let mut batch_s2 = FieldElement::<E>::zero();
                let mut batch_s3 = FieldElement::<E>::zero();
                let mut alpha_pow = FieldElement::<E>::one();

                for (idx, tables) in per_instance_tables.iter_mut().enumerate() {
                    let n_unused = max_parent_vars - tables.parent_num_vars;

                    if round_idx < n_unused {
                        // This instance hasn't started yet — constant polynomial = claim/2
                        let half_claim =
                            &combined_claims[idx] * &FieldElement::<E>::from(2u64).inv().unwrap();
                        // S(0) = S(1) = half_claim, S(2) = S(3) = half_claim (constant)
                        per_instance_evals[idx] = [
                            half_claim.clone(),
                            half_claim.clone(),
                            half_claim.clone(),
                            half_claim.clone(),
                        ];
                        batch_s0 = &batch_s0 + &(&alpha_pow * &half_claim);
                        batch_s1 = &batch_s1 + &(&alpha_pow * &half_claim);
                        batch_s2 = &batch_s2 + &(&alpha_pow * &half_claim);
                        batch_s3 = &batch_s3 + &(&alpha_pow * &half_claim);
                        alpha_pow = &alpha_pow * &sumcheck_alpha;
                        continue;
                    }

                    let instance_round = round_idx - n_unused;
                    let half = tables.nl_table.len().max(tables.dl_table.len()) / 2;

                    if half == 0 {
                        // Already reduced to constant
                        let half_claim =
                            &combined_claims[idx] * &FieldElement::<E>::from(2u64).inv().unwrap();
                        per_instance_evals[idx] = [
                            half_claim.clone(),
                            half_claim.clone(),
                            half_claim.clone(),
                            half_claim.clone(),
                        ];
                        batch_s0 = &batch_s0 + &(&alpha_pow * &half_claim);
                        batch_s1 = &batch_s1 + &(&alpha_pow * &half_claim);
                        batch_s2 = &batch_s2 + &(&alpha_pow * &half_claim);
                        batch_s3 = &batch_s3 + &(&alpha_pow * &half_claim);
                        alpha_pow = &alpha_pow * &sumcheck_alpha;
                        continue;
                    }

                    // Eq polynomial factoring (same as single-instance prover).
                    let r_round = tables.instance_point[instance_round].clone();
                    let one = FieldElement::<E>::one();

                    // Helper: compute gate h(0) and h(2) for a generic pair at index j.
                    let gate_generic = |tables: &PerInstanceTables<E>,
                                        j: usize|
                     -> [FieldElement<E>; 2] {
                        let nl_l = &tables.nl_table[2 * j];
                        let nl_r = &tables.nl_table[2 * j + 1];
                        let nr_l = &tables.nr_table[2 * j];
                        let nr_r = &tables.nr_table[2 * j + 1];
                        let dl_l = &tables.dl_table[2 * j];
                        let dl_r = &tables.dl_table[2 * j + 1];
                        let dr_l = &tables.dr_table[2 * j];
                        let dr_r = &tables.dr_table[2 * j + 1];

                        let gate_0 = &(nl_l * dr_l) + &(dl_l * &(nr_l + &(&lambda * dr_l)));
                        let nl_2 = &(nl_r + nl_r) - nl_l;
                        let nr_2 = &(nr_r + nr_r) - nr_l;
                        let dl_2 = &(dl_r + dl_r) - dl_l;
                        let dr_2 = &(dr_r + dr_r) - dr_l;
                        let gate_2 = &(&nl_2 * &dr_2) + &(&dl_2 * &(&nr_2 + &(&lambda * &dr_2)));
                        [gate_0, gate_2]
                    };

                    let gate_singles =
                        |tables: &PerInstanceTables<E>, j: usize| -> [FieldElement<E>; 2] {
                            let dl_l = &tables.dl_table[2 * j];
                            let dl_r = &tables.dl_table[2 * j + 1];
                            let dr_l = &tables.dr_table[2 * j];
                            let dr_r = &tables.dr_table[2 * j + 1];

                            let gate_0 = &(dl_l + dr_l) + &(&lambda * &(dl_l * dr_l));
                            let dl_2 = &(dl_r + dl_r) - dl_l;
                            let dr_2 = &(dr_r + dr_r) - dr_l;
                            let gate_2 = &(&dl_2 + &dr_2) + &(&lambda * &(&dl_2 * &dr_2));
                            [gate_0, gate_2]
                        };

                    // Compute h(0) and h(2) using SVO or standard path.
                    let (raw_h0, raw_h2) =
                        if tables.use_svo && instance_round < tables.svo_suffix_len {
                            // SVO suffix round: nested eq_suffix × (eq_prefix × gate) loop
                            let suffix_half = tables.eq_suffix.len() / 2;
                            let prefix_size = tables.eq_prefix.len();

                            // Pre-halve eq_suffix
                            for j in 0..suffix_half {
                                tables.eq_suffix[j] =
                                    &tables.eq_suffix[2 * j] + &tables.eq_suffix[2 * j + 1];
                            }
                            tables.eq_suffix.truncate(suffix_half);

                            let mut h0 = FieldElement::<E>::zero();
                            let mut h2 = FieldElement::<E>::zero();
                            for suffix_idx in 0..suffix_half {
                                let eq_s = &tables.eq_suffix[suffix_idx];
                                let mut ch0 = FieldElement::<E>::zero();
                                let mut ch2 = FieldElement::<E>::zero();
                                #[allow(clippy::needless_range_loop)]
                                for prefix_idx in 0..prefix_size {
                                    let j = prefix_idx * suffix_half + suffix_idx;
                                    let eq_p = &tables.eq_prefix[prefix_idx];
                                    let [g0, g2] = if tables.is_singles {
                                        gate_singles(tables, j)
                                    } else {
                                        gate_generic(tables, j)
                                    };
                                    ch0 = &ch0 + &(eq_p * &g0);
                                    ch2 = &ch2 + &(eq_p * &g2);
                                }
                                h0 = &h0 + &(eq_s * &ch0);
                                h2 = &h2 + &(eq_s * &ch2);
                            }
                            (h0, h2)
                        } else {
                            // Standard path (non-SVO, or SVO prefix rounds after suffix is exhausted)
                            if tables.use_svo && instance_round == tables.svo_suffix_len {
                                // Transition: absorb remaining eq_suffix into eq_correction
                                // and switch to eq_prefix as eq_table.
                                debug_assert_eq!(tables.eq_suffix.len(), 1);
                                tables.eq_correction = &tables.eq_correction * &tables.eq_suffix[0];
                                tables.eq_table = std::mem::take(&mut tables.eq_prefix);
                                tables.use_svo = false;
                            }

                            // Pre-halve eq_table
                            for j in 0..half {
                                tables.eq_table[j] =
                                    &tables.eq_table[2 * j] + &tables.eq_table[2 * j + 1];
                            }
                            tables.eq_table.truncate(half);

                            let mut h0 = FieldElement::<E>::zero();
                            let mut h2 = FieldElement::<E>::zero();
                            for j in 0..half {
                                let eq_rem = &tables.eq_table[j];
                                let [g0, g2] = if tables.is_singles {
                                    gate_singles(tables, j)
                                } else {
                                    gate_generic(tables, j)
                                };
                                h0 = &h0 + &(eq_rem * &g0);
                                h2 = &h2 + &(eq_rem * &g2);
                            }
                            (h0, h2)
                        };

                    // Apply eq_correction
                    let total_h0 = &tables.eq_correction * &raw_h0;
                    let total_h2 = &tables.eq_correction * &raw_h2;

                    // Recover S(t) from h(t) and eq_round(r, t)
                    let one_minus_r = &one - &r_round;
                    let s0 = &one_minus_r * &total_h0;
                    let s1 = &combined_claims[idx] - &s0;

                    let r_inv = r_round.inv().expect("r_round = 0 is probability 2^{-64}");
                    let h1 = &s1 * &r_inv;

                    let three = FieldElement::<E>::from(3u64);
                    let h3 = &(&(&three * &total_h2) - &(&three * &h1)) + &total_h0;

                    let eq_at_2 = &(&three * &r_round) - &one;
                    let s2 = &eq_at_2 * &total_h2;

                    let eq_at_3 = &(&FieldElement::<E>::from(5u64) * &r_round)
                        - &FieldElement::<E>::from(2u64);
                    let s3 = &eq_at_3 * &h3;

                    per_instance_evals[idx] = [s0.clone(), s1.clone(), s2.clone(), s3.clone()];
                    batch_s0 = &batch_s0 + &(&alpha_pow * &s0);
                    batch_s1 = &batch_s1 + &(&alpha_pow * &s1);
                    batch_s2 = &batch_s2 + &(&alpha_pow * &s2);
                    batch_s3 = &batch_s3 + &(&alpha_pow * &s3);
                    alpha_pow = &alpha_pow * &sumcheck_alpha;
                }

                let round_poly = RoundPoly::new(vec![batch_s0, batch_s1, batch_s2, batch_s3]);

                // Append to transcript
                for eval in round_poly.evals() {
                    transcript.append_field_element(eval);
                }

                // Sample challenge
                let challenge: FieldElement<E> = transcript.sample_field_element();
                _round_combined_claim = round_poly.evaluate(&challenge);

                // Update per-instance: fold tables, update eq_correction, and
                // update combined_claims via O(1) polynomial evaluation (not O(N) table scan).
                for (idx, tables) in per_instance_tables.iter_mut().enumerate() {
                    // Update combined_claims from saved per-instance round poly evals.
                    // S_i(challenge) via degree-3 Lagrange interpolation at {0,1,2,3}.
                    let [ref si0, ref si1, ref si2, ref si3] = per_instance_evals[idx];
                    let instance_poly =
                        RoundPoly::new(vec![si0.clone(), si1.clone(), si2.clone(), si3.clone()]);
                    combined_claims[idx] = instance_poly.evaluate(&challenge);

                    let n_unused = max_parent_vars - tables.parent_num_vars;
                    if round_idx < n_unused || tables.dl_table.len() / 2 == 0 {
                        continue;
                    }

                    let instance_round = round_idx - n_unused;
                    let half = tables.dl_table.len() / 2;

                    // Update eq_correction using instance-specific eval point
                    let r_round = &tables.instance_point[instance_round];
                    let one = FieldElement::<E>::one();
                    let eq_update =
                        &(r_round * &challenge) + &(&(&one - r_round) * &(&one - &challenge));
                    tables.eq_correction = &tables.eq_correction * &eq_update;

                    // Fold gate tables
                    let fold_table = |table: &mut Vec<FieldElement<E>>| {
                        #[cfg(feature = "parallel")]
                        if half >= 256 {
                            let folded: Vec<FieldElement<E>> = table
                                .par_chunks(2)
                                .map(|pair| &pair[0] + &(&challenge * &(&pair[1] - &pair[0])))
                                .collect();
                            *table = folded;
                            return;
                        }
                        for j in 0..half {
                            let left = &table[2 * j];
                            let right = &table[2 * j + 1];
                            table[j] = left + &(&challenge * &(right - left));
                        }
                        table.truncate(half);
                    };

                    if !tables.is_singles {
                        fold_table(&mut tables.nl_table);
                        fold_table(&mut tables.nr_table);
                    }
                    fold_table(&mut tables.dl_table);
                    fold_table(&mut tables.dr_table);
                }

                round_polys.push(round_poly);
                challenges.push(challenge);
            }

            // After all sumcheck rounds, extract child claims from each instance's tables
            let mut child_claims_by_instance = Vec::new();

            for tables in &per_instance_tables {
                if tables.is_singles {
                    child_claims_by_instance.push([
                        FieldElement::one(),
                        FieldElement::one(),
                        tables.dl_table[0].clone(),
                        tables.dr_table[0].clone(),
                    ]);
                } else {
                    child_claims_by_instance.push([
                        tables.nl_table[0].clone(),
                        tables.nr_table[0].clone(),
                        tables.dl_table[0].clone(),
                        tables.dr_table[0].clone(),
                    ]);
                }
            }

            // Append child claims to transcript
            for claims in &child_claims_by_instance {
                for claim in claims {
                    transcript.append_field_element(claim);
                }
            }

            // Sample eta to fold left/right
            let eta: FieldElement<E> = transcript.sample_field_element();

            // Update per-instance claims
            for (idx, &i) in active_instances.iter().enumerate() {
                let [ref nl, ref nr, ref dl, ref dr] = child_claims_by_instance[idx];
                n_claims[i] = Some(nl + &(&eta * &(nr - nl)));
                d_claims[i] = Some(dl + &(&eta * &(dr - dl)));
            }

            let mut new_point = Vec::with_capacity(challenges.len() + 1);
            new_point.push(eta);
            new_point.extend(challenges);
            current_point = new_point;

            layer_proofs.push(BatchGkrLayerProof {
                sumcheck_proof: SumcheckProof { round_polys },
                child_claims_by_instance,
            });
        }
    }

    let final_claims: Vec<(FieldElement<E>, FieldElement<E>)> = (0..n_instances)
        .map(|i| {
            (
                n_claims[i]
                    .clone()
                    .unwrap_or_else(|| root_claims[i].0.clone()),
                d_claims[i]
                    .clone()
                    .unwrap_or_else(|| root_claims[i].1.clone()),
            )
        })
        .collect();

    (
        BatchGkrProof {
            root_claims,
            layer_proofs,
        },
        current_point,
        final_claims,
    )
}

/// Compute the per-instance evaluation point from the shared point.
///
/// In batch GKR with mixed-size instances, smaller instances skip the first
/// rounds of each shared sumcheck. Their variables bind to the **last**
/// challenges, not the first. The eval point for instance i is:
///   `[shared_point[0] (eta)] ++ shared_point[len - (n_vars - 1)..]`
///
/// Returns an empty vec for n_vars == 0 (trivial/output layers).
pub(crate) fn instance_eval_point<E: IsField>(
    shared_point: &[FieldElement<E>],
    n_vars: usize,
) -> Vec<FieldElement<E>> {
    if n_vars == 0 {
        return vec![];
    }
    if n_vars >= shared_point.len() {
        return shared_point.to_vec();
    }
    let mut point = Vec::with_capacity(n_vars);
    point.push(shared_point[0].clone()); // eta (shared left/right selector)
    let start = shared_point.len() - (n_vars - 1);
    point.extend_from_slice(&shared_point[start..]);
    point
}

/// Per-instance bookkeeping tables for the batch sumcheck inner loop.
struct PerInstanceTables<E: IsField> {
    nl_table: Vec<FieldElement<E>>,
    nr_table: Vec<FieldElement<E>>,
    dl_table: Vec<FieldElement<E>>,
    dr_table: Vec<FieldElement<E>>,
    eq_table: Vec<FieldElement<E>>,
    eq_correction: FieldElement<E>,
    is_singles: bool,
    parent_num_vars: usize,
    /// The instance-specific evaluation point derived from the shared current_point.
    /// Used for r_round lookups in the Dao-Thaler eq factoring.
    instance_point: Vec<FieldElement<E>>,
    // SVO (Split-Value Optimization) fields.
    // When use_svo is true, eq is split into prefix × suffix for sqrt memory.
    use_svo: bool,
    svo_suffix_len: usize,
    eq_prefix: Vec<FieldElement<E>>,
    eq_suffix: Vec<FieldElement<E>>,
}

// =============================================================================
// Batch GKR verifier
// =============================================================================

/// Verify a batch GKR proof.
///
/// Replays the Fiat-Shamir transcript and checks sumcheck consistency and gate
/// equations for all instances simultaneously.
///
/// # Returns
/// `Ok((shared_random_point, per_instance_claims))` where per_instance_claims[i] = (n_claim, d_claim).
#[allow(clippy::type_complexity)]
pub fn gkr_verify_batch<E: IsField>(
    proof: &BatchGkrProof<E>,
    n_layers_by_instance: &[usize],
    transcript: &mut impl IsTranscript<E>,
) -> Result<
    (
        Vec<FieldElement<E>>,
        Vec<(FieldElement<E>, FieldElement<E>)>,
    ),
    GkrError,
> {
    let n_instances = proof.root_claims.len();
    if n_layers_by_instance.len() != n_instances {
        return Err(GkrError::InvalidTree {
            reason: "n_layers_by_instance length mismatch".to_string(),
        });
    }

    // Domain separation (must match prover)
    transcript.append_bytes(b"gkr_batch");
    transcript.append_bytes(&(n_instances as u64).to_le_bytes());

    if n_instances == 0 {
        return Ok((vec![], vec![]));
    }

    let max_layers = *n_layers_by_instance.iter().max().unwrap();

    if proof.layer_proofs.len() != max_layers {
        return Err(GkrError::InvalidTree {
            reason: format!(
                "expected {} layer proofs but got {}",
                max_layers,
                proof.layer_proofs.len(),
            ),
        });
    }

    // Track per-instance state
    let mut n_claims: Vec<Option<FieldElement<E>>> = vec![None; n_instances];
    let mut d_claims: Vec<Option<FieldElement<E>>> = vec![None; n_instances];
    let mut current_point: Vec<FieldElement<E>> = vec![];

    for (layer_idx, layer_proof) in proof.layer_proofs.iter().enumerate() {
        let n_remaining = max_layers - layer_idx;

        // Detect output layers — use actual (root_n, root_d) from the proof
        for i in 0..n_instances {
            if n_layers_by_instance[i] == n_remaining {
                n_claims[i] = Some(proof.root_claims[i].0.clone());
                d_claims[i] = Some(proof.root_claims[i].1.clone());
            }
        }

        // Append active claims to transcript
        for i in 0..n_instances {
            if let (Some(n), Some(d)) = (&n_claims[i], &d_claims[i]) {
                transcript.append_field_element(n);
                transcript.append_field_element(d);
            }
        }

        // Sample randomness
        let sumcheck_alpha: FieldElement<E> = transcript.sample_field_element();
        let lambda: FieldElement<E> = transcript.sample_field_element();

        // Collect active instances
        let mut active_instances: Vec<usize> = Vec::new();
        let mut combined_claims: Vec<FieldElement<E>> = Vec::new();

        for i in 0..n_instances {
            if n_claims[i].is_some() && n_layers_by_instance[i] > 0 {
                active_instances.push(i);
                let n = n_claims[i].as_ref().unwrap();
                let d = d_claims[i].as_ref().unwrap();
                let claim = n + &(&lambda * d);
                let n_unused = max_layers - n_layers_by_instance[i];
                if n_unused > 0 {
                    let doubling = FieldElement::<E>::from(1u64 << n_unused);
                    combined_claims.push(&claim * &doubling);
                } else {
                    combined_claims.push(claim);
                }
            }
        }

        let round_polys = &layer_proof.sumcheck_proof.round_polys;

        if layer_proof.child_claims_by_instance.len() < active_instances.len() {
            return Err(GkrError::InvalidTree {
                reason: format!(
                    "layer {}: child_claims_by_instance has {} entries but {} active instances",
                    layer_idx,
                    layer_proof.child_claims_by_instance.len(),
                    active_instances.len(),
                ),
            });
        }

        if round_polys.is_empty() {
            // Trivial layer: no sumcheck rounds.
            // Gate check: verify that the alpha-batched combined claims match the
            // alpha-batched gate evaluations (gate_i = nl·dr + nr·dl + λ·dl·dr, scaled
            // by 2^n_unused_i to match combined_claims).
            {
                let mut actual_sum = FieldElement::<E>::zero();
                let mut expected_sum = FieldElement::<E>::zero();
                let mut alpha_pow = FieldElement::<E>::one();
                for (idx, &i) in active_instances.iter().enumerate() {
                    let [ref nl, ref nr, ref dl, ref dr] =
                        layer_proof.child_claims_by_instance[idx];
                    let gate = &(&(nl * dr) + &(nr * dl)) + &(&lambda * &(dl * dr));
                    let n_unused = max_layers - n_layers_by_instance[i];
                    let gate_scaled = if n_unused > 0 {
                        &gate * &FieldElement::<E>::from(1u64 << n_unused)
                    } else {
                        gate
                    };
                    actual_sum = &actual_sum + &(&alpha_pow * &combined_claims[idx]);
                    expected_sum = &expected_sum + &(&alpha_pow * &gate_scaled);
                    alpha_pow = &alpha_pow * &sumcheck_alpha;
                }
                if actual_sum != expected_sum {
                    return Err(GkrError::GateCheckFailed { layer: layer_idx });
                }
            }

            // Append child claims and sample eta (same as prover)
            for claims in &layer_proof.child_claims_by_instance {
                for claim in claims {
                    transcript.append_field_element(claim);
                }
            }

            let eta: FieldElement<E> = transcript.sample_field_element();

            for (idx, &i) in active_instances.iter().enumerate() {
                let [ref nl, ref nr, ref dl, ref dr] = layer_proof.child_claims_by_instance[idx];
                n_claims[i] = Some(nl + &(&eta * &(nr - nl)));
                d_claims[i] = Some(dl + &(&eta * &(dr - dl)));
            }

            current_point = vec![eta];
        } else {
            // Non-trivial: verify sumcheck
            let num_rounds = round_polys.len();

            // Compute combined claim across instances via sumcheck_alpha
            let mut current_sum = {
                let mut sum = FieldElement::<E>::zero();
                let mut alpha_pow = FieldElement::<E>::one();
                for claim in &combined_claims {
                    sum = &sum + &(&alpha_pow * claim);
                    alpha_pow = &alpha_pow * &sumcheck_alpha;
                }
                sum
            };

            let mut challenges = Vec::with_capacity(num_rounds);

            for (round, round_poly) in round_polys.iter().enumerate() {
                if round_poly.sum_at_binary() != current_sum {
                    return Err(GkrError::SumcheckFailed {
                        layer: layer_idx,
                        reason: format!("round {} sum mismatch: p(0)+p(1) != expected sum", round),
                    });
                }

                for eval in round_poly.evals() {
                    transcript.append_field_element(eval);
                }

                let challenge: FieldElement<E> = transcript.sample_field_element();
                current_sum = round_poly.evaluate(&challenge);
                challenges.push(challenge);
            }

            // Gate check: for each active instance, verify the gate equation
            let mut expected_sum = FieldElement::<E>::zero();
            let mut alpha_pow = FieldElement::<E>::one();

            for (idx, &i) in active_instances.iter().enumerate() {
                let [ref nl, ref nr, ref dl, ref dr] = layer_proof.child_claims_by_instance[idx];

                // parent_num_vars for this instance at this layer:
                // instance i's child layer has (n_layers[i] - n_remaining + 1) variables,
                // so the parent has (n_layers[i] - n_remaining) variables.
                let parent_num_vars_i = n_layers_by_instance[i] - n_remaining;
                if num_rounds < parent_num_vars_i {
                    return Err(GkrError::InvalidTree {
                        reason: format!(
                            "layer {}: num_rounds ({}) < parent_num_vars for instance {} ({})",
                            layer_idx, num_rounds, i, parent_num_vars_i,
                        ),
                    });
                }
                let sumcheck_n_unused = num_rounds - parent_num_vars_i;

                // eq evaluation: the prover builds eq from the instance-specific eval point
                // (eta + last challenges), and the active sumcheck challenges are the last ones.
                let eq_val = if parent_num_vars_i == 0 {
                    FieldElement::<E>::one()
                } else {
                    let inst_point = instance_eval_point(&current_point, parent_num_vars_i);
                    compute_eq_at_point(&inst_point, &challenges[sumcheck_n_unused..])
                };

                // gate_combined = nl*dr + nr*dl + lambda*dl*dr
                let gate_combined = &(&(nl * dr) + &(nr * dl)) + &(&lambda * &(dl * dr));

                // No doubling factor here: the 2^n_unused doubling is in the initial
                // combined claim and gets cancelled by the unused rounds' halving.
                // The point evaluation at the final challenge is just eq * gate.
                let instance_eval = &eq_val * &gate_combined;

                expected_sum = &expected_sum + &(&alpha_pow * &instance_eval);
                alpha_pow = &alpha_pow * &sumcheck_alpha;
            }

            if current_sum != expected_sum {
                return Err(GkrError::GateCheckFailed { layer: layer_idx });
            }

            // Append child claims to transcript
            for claims in &layer_proof.child_claims_by_instance {
                for claim in claims {
                    transcript.append_field_element(claim);
                }
            }

            let eta: FieldElement<E> = transcript.sample_field_element();

            for (idx, &i) in active_instances.iter().enumerate() {
                let [ref nl, ref nr, ref dl, ref dr] = layer_proof.child_claims_by_instance[idx];
                n_claims[i] = Some(nl + &(&eta * &(nr - nl)));
                d_claims[i] = Some(dl + &(&eta * &(dr - dl)));
            }

            let mut new_point = Vec::with_capacity(challenges.len() + 1);
            new_point.push(eta);
            new_point.extend(challenges);
            current_point = new_point;
        }
    }

    let final_claims: Vec<(FieldElement<E>, FieldElement<E>)> = (0..n_instances)
        .map(|i| {
            (
                n_claims[i]
                    .clone()
                    .unwrap_or_else(|| proof.root_claims[i].0.clone()),
                d_claims[i]
                    .clone()
                    .unwrap_or_else(|| proof.root_claims[i].1.clone()),
            )
        })
        .collect();

    Ok((current_point, final_claims))
}

/// Compute eq(a, b) for two points of equal length.
///
/// eq(a, b) = prod_i (a_i * b_i + (1 - a_i) * (1 - b_i))
///
/// This is a single field element, NOT the full eq table.
fn compute_eq_at_point<E: IsField>(
    a: &[FieldElement<E>],
    b: &[FieldElement<E>],
) -> FieldElement<E> {
    assert_eq!(a.len(), b.len(), "eq points must have equal length");
    let one = FieldElement::<E>::one();
    a.iter()
        .zip(b.iter())
        .fold(FieldElement::one(), |acc, (ai, bi)| {
            let term = &(ai * bi) + &(&(&one - ai) * &(&one - bi));
            &acc * &term
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn test_fraction_add() {
        // (3/5) + (7/11) = (3*11 + 7*5) / (5*11) = (33 + 35) / 55 = 68/55
        let a = Fraction::new(FE::from(3u64), FE::from(5u64));
        let b = Fraction::new(FE::from(7u64), FE::from(11u64));
        let result = a.add(&b);

        assert_eq!(result.numerator, FE::from(68u64));
        assert_eq!(result.denominator, FE::from(55u64));
    }

    #[test]
    fn test_build_summation_tree_4_leaves() {
        // 4 leaves: 1/2, 3/4, 5/6, 7/8
        // Layer 0 (4 fractions): [1/2, 3/4, 5/6, 7/8]
        // Layer 1 (2 fractions):
        //   pair 0: 1/2 + 3/4 = (1*4 + 3*2)/(2*4) = 10/8
        //   pair 1: 5/6 + 7/8 = (5*8 + 7*6)/(6*8) = 82/48
        // Layer 2 (1 fraction, root):
        //   10/8 + 82/48 = (10*48 + 82*8)/(8*48) = (480 + 656)/384 = 1136/384
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];

        let tree = build_summation_tree(nums, dens);

        // Should have 3 layers: 4 -> 2 -> 1
        assert_eq!(tree.len(), 3);
        assert_eq!(tree[0].numerators.len(), 4);
        assert_eq!(tree[1].numerators.len(), 2);
        assert_eq!(tree[2].numerators.len(), 1);

        // Layer 1 checks
        assert_eq!(tree[1].numerators[0], FE::from(10u64)); // 1*4 + 3*2
        assert_eq!(tree[1].denominators[0], FE::from(8u64)); // 2*4
        assert_eq!(tree[1].numerators[1], FE::from(82u64)); // 5*8 + 7*6
        assert_eq!(tree[1].denominators[1], FE::from(48u64)); // 6*8

        // Root checks
        assert_eq!(tree[2].numerators[0], FE::from(1136u64)); // 10*48 + 82*8
        assert_eq!(tree[2].denominators[0], FE::from(384u64)); // 8*48

        // Verify the root fraction equals the actual sum:
        // 1/2 + 3/4 + 5/6 + 7/8 = 12/24 + 18/24 + 20/24 + 21/24 = 71/24
        // Check: 1136/384 = 71/24 (both sides: 1136*24 = 27264, 71*384 = 27264)
        let root_n = &tree[2].numerators[0];
        let root_d = &tree[2].denominators[0];
        assert_eq!(root_n * &FE::from(24u64), &FE::from(71u64) * root_d);
    }

    #[test]
    fn test_build_summation_tree_8_leaves() {
        // 8 leaves: i/(i+1) for i in 1..=8, i.e., 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8, 8/9
        let nums: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (2..=9).map(|i| FE::from(i as u64)).collect();

        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Should have 4 layers: 8 -> 4 -> 2 -> 1
        assert_eq!(tree.len(), 4);
        assert_eq!(tree[0].numerators.len(), 8);
        assert_eq!(tree[1].numerators.len(), 4);
        assert_eq!(tree[2].numerators.len(), 2);
        assert_eq!(tree[3].numerators.len(), 1);

        // Compute expected root by sequential fraction addition
        let mut acc = Fraction::new(nums[0], dens[0]);
        for i in 1..8 {
            acc = acc.add(&Fraction::new(nums[i], dens[i]));
        }

        // The tree root and the sequential sum should represent the same rational number:
        // tree_n / tree_d == acc_n / acc_d  <=>  tree_n * acc_d == acc_n * tree_d
        let root_n = &tree[3].numerators[0];
        let root_d = &tree[3].denominators[0];
        assert_eq!(
            root_n * &acc.denominator,
            &acc.numerator * root_d,
            "Tree root must equal sequential sum as a fraction"
        );
    }

    #[test]
    fn test_build_summation_tree_single_leaf() {
        // Edge case: 1 leaf (2^0 = 1)
        let nums = vec![FE::from(42u64)];
        let dens = vec![FE::from(7u64)];

        let tree = build_summation_tree(nums, dens);

        // Should have 1 layer (just the leaf)
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].numerators.len(), 1);
        assert_eq!(tree[0].numerators[0], FE::from(42u64));
        assert_eq!(tree[0].denominators[0], FE::from(7u64));
    }

    #[test]
    #[should_panic(expected = "number of leaves must be a power of 2")]
    fn test_build_summation_tree_non_power_of_2_panics() {
        let nums = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let dens = vec![FE::from(1u64), FE::from(1u64), FE::from(1u64)];
        let _ = build_summation_tree(nums, dens);
    }

    #[test]
    #[should_panic(expected = "numerators and denominators must have the same length")]
    fn test_build_summation_tree_mismatched_lengths_panics() {
        let nums = vec![FE::from(1u64), FE::from(2u64)];
        let dens = vec![FE::from(1u64)];
        let _ = build_summation_tree(nums, dens);
    }

    // ==================== compute_eq_evals tests ====================

    #[test]
    fn test_compute_eq_evals_empty_point() {
        // eq with 0 variables: single entry = 1
        let evals = compute_eq_evals::<GoldilocksField>(&[]);
        assert_eq!(evals.len(), 1);
        assert_eq!(evals[0], FE::one());
    }

    #[test]
    fn test_compute_eq_evals_1var() {
        // eq((r,), (b,)) = r*b + (1-r)*(1-b)
        // For r = 3: eq(3, 0) = (1-3) = -2, eq(3, 1) = 3
        let r = FE::from(3u64);
        let evals = compute_eq_evals(std::slice::from_ref(&r));
        assert_eq!(evals.len(), 2);
        assert_eq!(evals[0], FE::one() - r); // eq(r, 0) = 1-r = -2
        assert_eq!(evals[1], r); // eq(r, 1) = r = 3
    }

    #[test]
    fn test_compute_eq_evals_2var() {
        // eq((r0, r1), (b0, b1)) = [r0*b0 + (1-r0)*(1-b0)] * [r1*b1 + (1-r1)*(1-b1)]
        let r0 = FE::from(2u64);
        let r1 = FE::from(5u64);
        let evals = compute_eq_evals(&[r0, r1]);
        assert_eq!(evals.len(), 4);

        let one = FE::one();
        let one_minus_r0 = &one - &r0;
        let one_minus_r1 = &one - &r1;

        // Index 0 = (b0=0, b1=0): (1-r0)*(1-r1)
        assert_eq!(evals[0], &one_minus_r0 * &one_minus_r1);
        // Index 1 = (b0=1, b1=0): r0*(1-r1)
        assert_eq!(evals[1], &r0 * &one_minus_r1);
        // Index 2 = (b0=0, b1=1): (1-r0)*r1
        assert_eq!(evals[2], &one_minus_r0 * &r1);
        // Index 3 = (b0=1, b1=1): r0*r1
        assert_eq!(evals[3], &r0 * &r1);
    }

    #[test]
    fn test_compute_eq_evals_sum_to_one_on_booleans() {
        // When point is Boolean, eq_evals should have exactly one 1 and rest 0
        let point = vec![FE::one(), FE::zero(), FE::one()]; // b = (1, 0, 1) = index 5
        let evals = compute_eq_evals(&point);
        assert_eq!(evals.len(), 8);
        for (i, e) in evals.iter().enumerate() {
            if i == 5 {
                assert_eq!(*e, FE::one());
            } else {
                assert_eq!(*e, FE::zero());
            }
        }
    }

    // ==================== evaluate_mle tests ====================

    #[test]
    fn test_evaluate_mle_linear() {
        // MLE of [3, 7] at point r:
        // f(x) = 3*(1-x) + 7*x = 3 + 4x
        // f(5) = 3 + 20 = 23
        let table = vec![FE::from(3u64), FE::from(7u64)];
        let result = evaluate_mle(&table, &[FE::from(5u64)]);
        assert_eq!(result, FE::from(23u64));
    }

    #[test]
    fn test_evaluate_mle_at_boolean() {
        // MLE at a Boolean point should return the table entry
        let table = vec![
            FE::from(10u64),
            FE::from(20u64),
            FE::from(30u64),
            FE::from(40u64),
        ];
        // Index 2 = (b0=0, b1=1)
        let result = evaluate_mle(&table, &[FE::zero(), FE::one()]);
        assert_eq!(result, FE::from(30u64));
    }

    // ==================== GKR prover tests ====================

    #[test]
    fn test_gkr_prove_2_leaves() {
        // Simplest non-trivial case: 2 leaves
        // Tree: layer 0 (2 fractions), layer 1 (root, 1 fraction)
        // Leaves: 3/5, 7/11
        // Root: (3*11 + 7*5) / (5*11) = 68/55
        let nums = vec![FE::from(3u64), FE::from(7u64)];
        let dens = vec![FE::from(5u64), FE::from(11u64)];
        let tree = build_summation_tree(nums, dens);

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, final_point, final_n_claim, final_d_claim) =
            gkr_prove(&tree, &mut transcript).unwrap();

        // claimed_sum = 68/55
        let expected_sum = &FE::from(68u64) * &FE::from(55u64).inv().unwrap();
        assert_eq!(proof.claimed_sum, expected_sum);

        // Should have 1 layer proof (root -> leaves)
        assert_eq!(proof.layer_proofs.len(), 1);

        // The first (and only) layer proof has a trivial sumcheck (0 rounds)
        assert_eq!(proof.layer_proofs[0].sumcheck_proof.round_polys.len(), 0);

        // The child claims should be the raw leaf values
        assert_eq!(proof.layer_proofs[0].child_claims[0], FE::from(3u64)); // n_left
        assert_eq!(proof.layer_proofs[0].child_claims[1], FE::from(7u64)); // n_right
        assert_eq!(proof.layer_proofs[0].child_claims[2], FE::from(5u64)); // d_left
        assert_eq!(proof.layer_proofs[0].child_claims[3], FE::from(11u64)); // d_right

        // final_point should have 1 element (eta)
        assert_eq!(final_point.len(), 1);

        // Final claims should be the leaf MLE evaluated at final_point
        // n_MLE(eta) = 3*(1-eta) + 7*eta = 3 + 4*eta
        // d_MLE(eta) = 5*(1-eta) + 11*eta = 5 + 6*eta
        let eta = &final_point[0];
        let expected_n = &FE::from(3u64) * &(&FE::one() - eta) + &(&FE::from(7u64) * eta);
        let expected_d = &FE::from(5u64) * &(&FE::one() - eta) + &(&FE::from(11u64) * eta);
        assert_eq!(final_n_claim, expected_n);
        assert_eq!(final_d_claim, expected_d);
    }

    #[test]
    fn test_gkr_prove_4_leaves() {
        // 4 leaves: 1/2, 3/4, 5/6, 7/8
        // Tree has 3 layers (0=leaves size 4, 1=size 2, 2=root size 1)
        // GKR reduces: root -> layer 1 (trivial, 0 vars) -> layer 0 (1-var sumcheck)
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums.clone(), dens.clone());

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, final_point, final_n_claim, final_d_claim) =
            gkr_prove(&tree, &mut transcript).unwrap();

        // claimed_sum = root_n / root_d = 1136 / 384
        let expected_sum = &FE::from(1136u64) * &FE::from(384u64).inv().unwrap();
        assert_eq!(proof.claimed_sum, expected_sum);

        // Should have 2 layer proofs
        assert_eq!(proof.layer_proofs.len(), 2);

        // First layer proof: root (1 elem) -> layer 1 (2 elems), trivial (0 rounds)
        assert_eq!(proof.layer_proofs[0].sumcheck_proof.round_polys.len(), 0);

        // Second layer proof: layer 1 (2 elems) -> layer 0 (4 elems)
        // This is a 1-variable sumcheck, so should have 1 round polynomial
        assert_eq!(proof.layer_proofs[1].sumcheck_proof.round_polys.len(), 1);

        // The round polynomial should have 4 evaluations (degree 3)
        assert_eq!(
            proof.layer_proofs[1].sumcheck_proof.round_polys[0].num_evals(),
            4
        );

        // final_point should have 2 elements (challenge from sumcheck + eta)
        assert_eq!(final_point.len(), 2);

        // Verify final claims match the leaf MLEs at final_point
        let expected_n = evaluate_mle(&nums, &final_point);
        let expected_d = evaluate_mle(&dens, &final_point);
        assert_eq!(final_n_claim, expected_n);
        assert_eq!(final_d_claim, expected_d);
    }

    #[test]
    fn test_gkr_prove_claimed_sum() {
        // Verify that proof.claimed_sum equals root_n * root_d.inv()
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums, dens);

        let root_n = &tree[2].numerators[0];
        let root_d = &tree[2].denominators[0];
        let expected_sum = root_n * &root_d.inv().unwrap();

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, _, _, _) = gkr_prove(&tree, &mut transcript).unwrap();

        assert_eq!(proof.claimed_sum, expected_sum);
    }

    #[test]
    fn test_gkr_prove_8_leaves() {
        // 8 leaves: i/(i+1) for i in 1..=8
        // Tree: 4 layers (sizes 8, 4, 2, 1)
        // GKR reductions: root->layer2 (trivial), layer2->layer1 (1-var), layer1->layer0 (2-var)
        let nums: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (2..=9).map(|i| FE::from(i as u64)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, final_point, final_n_claim, final_d_claim) =
            gkr_prove(&tree, &mut transcript).unwrap();

        // Should have 3 layer proofs
        assert_eq!(proof.layer_proofs.len(), 3);

        // Layer 0: root (1) -> layer 2 (2): trivial
        assert_eq!(proof.layer_proofs[0].sumcheck_proof.round_polys.len(), 0);

        // Layer 1: layer 2 (2) -> layer 1 (4): 1-variable sumcheck
        assert_eq!(proof.layer_proofs[1].sumcheck_proof.round_polys.len(), 1);

        // Layer 2: layer 1 (4) -> layer 0 (8): 2-variable sumcheck
        assert_eq!(proof.layer_proofs[2].sumcheck_proof.round_polys.len(), 2);

        // final_point should have 3 elements
        assert_eq!(final_point.len(), 3);

        // Verify final claims match the leaf MLEs at final_point
        let expected_n = evaluate_mle(&nums, &final_point);
        let expected_d = evaluate_mle(&dens, &final_point);
        assert_eq!(final_n_claim, expected_n);
        assert_eq!(final_d_claim, expected_d);
    }

    #[test]
    fn test_gkr_prove_16_leaves() {
        // 16 leaves with various fractions
        // Tree: 5 layers (sizes 16, 8, 4, 2, 1)
        let nums: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (17..=32).map(|i| FE::from(i as u64)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAB]);
        let (proof, final_point, final_n_claim, final_d_claim) =
            gkr_prove(&tree, &mut transcript).unwrap();

        // Should have 4 layer proofs
        assert_eq!(proof.layer_proofs.len(), 4);

        // final_point should have 4 elements (log2(16))
        assert_eq!(final_point.len(), 4);

        // Verify final claims match the leaf MLEs at final_point
        let expected_n = evaluate_mle(&nums, &final_point);
        let expected_d = evaluate_mle(&dens, &final_point);
        assert_eq!(final_n_claim, expected_n);
        assert_eq!(final_d_claim, expected_d);
    }

    #[test]
    fn test_gkr_prove_deterministic() {
        // Same inputs and transcript seed should produce identical proofs
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums, dens);

        let mut t1 = DefaultTranscript::<GoldilocksField>::new(&[0x42]);
        let (proof1, point1, n1, d1) = gkr_prove(&tree, &mut t1).unwrap();

        let mut t2 = DefaultTranscript::<GoldilocksField>::new(&[0x42]);
        let (proof2, point2, n2, d2) = gkr_prove(&tree, &mut t2).unwrap();

        assert_eq!(proof1.claimed_sum, proof2.claimed_sum);
        assert_eq!(point1, point2);
        assert_eq!(n1, n2);
        assert_eq!(d1, d2);
        assert_eq!(proof1.layer_proofs.len(), proof2.layer_proofs.len());

        for (lp1, lp2) in proof1.layer_proofs.iter().zip(proof2.layer_proofs.iter()) {
            assert_eq!(lp1.child_claims, lp2.child_claims);
            assert_eq!(
                lp1.sumcheck_proof.round_polys.len(),
                lp2.sumcheck_proof.round_polys.len()
            );
            for (rp1, rp2) in lp1
                .sumcheck_proof
                .round_polys
                .iter()
                .zip(lp2.sumcheck_proof.round_polys.iter())
            {
                assert_eq!(rp1.evals(), rp2.evals());
            }
        }
    }

    #[test]
    fn test_gkr_prove_sumcheck_consistency() {
        // Verify that the round polynomial p(0) + p(1) matches the combined claim
        // for the non-trivial sumcheck layers
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Re-run the protocol manually to extract the combined_claim at each layer
        // and verify the sumcheck round poly sum
        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, _, _, _) = gkr_prove(&tree, &mut transcript).unwrap();

        // Replay transcript to get the same challenges
        let mut replay = DefaultTranscript::<GoldilocksField>::new(&[]);
        replay.append_field_element(&proof.claimed_sum);

        let mut n_claim = tree[2].numerators[0];
        let mut d_claim = tree[2].denominators[0];

        for (layer_idx, lp) in proof.layer_proofs.iter().enumerate() {
            let lambda: FE = replay.sample_field_element();
            let combined_claim = &n_claim + &(&lambda * &d_claim);

            if lp.sumcheck_proof.round_polys.is_empty() {
                // Trivial layer: verify gate equation directly
                let nl = &lp.child_claims[0];
                let nr = &lp.child_claims[1];
                let dl = &lp.child_claims[2];
                let dr = &lp.child_claims[3];
                let gate_val = &(nl * dr) + &(nr * dl) + &(&lambda * &(dl * dr));
                assert_eq!(
                    combined_claim, gate_val,
                    "Gate equation failed at trivial layer {}",
                    layer_idx
                );
            } else {
                // Non-trivial layer: verify p(0) + p(1) = combined_claim
                let first_round = &lp.sumcheck_proof.round_polys[0];
                assert_eq!(
                    first_round.sum_at_binary(),
                    combined_claim,
                    "Sumcheck round sum mismatch at layer {}",
                    layer_idx
                );
            }

            // Replay transcript operations from the sumcheck
            for rp in &lp.sumcheck_proof.round_polys {
                for eval in rp.evals() {
                    replay.append_field_element(eval);
                }
                let _challenge: FE = replay.sample_field_element();
            }

            // Replay child claims and eta
            for claim in &lp.child_claims {
                replay.append_field_element(claim);
            }
            let eta: FE = replay.sample_field_element();

            // Update claims for next layer
            let one_minus_eta = &FE::one() - &eta;
            n_claim = &(&lp.child_claims[0] * &one_minus_eta) + &(&lp.child_claims[1] * &eta);
            d_claim = &(&lp.child_claims[2] * &one_minus_eta) + &(&lp.child_claims[3] * &eta);
        }
    }

    #[test]
    fn test_gkr_prove_single_leaf() {
        // Edge case: single leaf (tree has 1 layer, no reductions)
        let nums = vec![FE::from(42u64)];
        let dens = vec![FE::from(7u64)];
        let tree = build_summation_tree(nums, dens);

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, final_point, final_n_claim, final_d_claim) =
            gkr_prove(&tree, &mut transcript).unwrap();

        assert_eq!(proof.claimed_sum, FE::from(6u64)); // 42/7 = 6
        assert_eq!(proof.layer_proofs.len(), 0);
        assert!(final_point.is_empty());
        assert_eq!(final_n_claim, FE::from(42u64));
        assert_eq!(final_d_claim, FE::from(7u64));
    }

    // ==================== GKR verifier tests ====================

    #[test]
    fn test_gkr_prove_verify_roundtrip_4() {
        // 4 leaves: 1/2, 3/4, 5/6, 7/8
        // Tree has 3 layers (0=leaves size 4, 1=size 2, 2=root size 1)
        // GKR reduces: root -> layer 1 (trivial) -> layer 0 (1-var sumcheck)
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let (proof, prover_point, prover_n, prover_d) =
            gkr_prove(&tree, &mut prover_transcript).unwrap();

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "GKR verification should succeed for 4 leaves"
        );
        let (verifier_point, verifier_n, verifier_d) = result.unwrap();

        // The verifier's final point must match the prover's
        assert_eq!(
            verifier_point, prover_point,
            "Verifier and prover must derive the same final point"
        );

        // The verifier's leaf claims must match the prover's
        assert_eq!(verifier_n, prover_n, "n_claim must match");
        assert_eq!(verifier_d, prover_d, "d_claim must match");

        // Additionally verify that the claims are consistent with the leaf MLEs
        let expected_n = evaluate_mle(&nums, &verifier_point);
        let expected_d = evaluate_mle(&dens, &verifier_point);
        assert_eq!(verifier_n, expected_n, "n_claim must match leaf MLE");
        assert_eq!(verifier_d, expected_d, "d_claim must match leaf MLE");
    }

    #[test]
    fn test_gkr_prove_verify_roundtrip_8() {
        // 8 leaves: i/(i+1) for i in 1..=8
        // Tree: 4 layers (sizes 8, 4, 2, 1)
        // GKR: root->layer2 (trivial), layer2->layer1 (1-var), layer1->layer0 (2-var)
        let nums: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (2..=9).map(|i| FE::from(i as u64)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let (proof, prover_point, prover_n, prover_d) =
            gkr_prove(&tree, &mut prover_transcript).unwrap();

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "GKR verification should succeed for 8 leaves"
        );
        let (verifier_point, verifier_n, verifier_d) = result.unwrap();

        // The verifier's final point must match the prover's
        assert_eq!(verifier_point, prover_point);

        // The verifier's leaf claims must match the prover's
        assert_eq!(verifier_n, prover_n);
        assert_eq!(verifier_d, prover_d);

        // Verify consistency with leaf MLEs
        let expected_n = evaluate_mle(&nums, &verifier_point);
        let expected_d = evaluate_mle(&dens, &verifier_point);
        assert_eq!(verifier_n, expected_n);
        assert_eq!(verifier_d, expected_d);
    }

    #[test]
    fn test_gkr_verify_wrong_claimed_sum() {
        // Create a valid proof and then tamper with the claimed_sum.
        // The verifier should fail (either at sumcheck or gate check).
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums, dens);

        // Prove with correct claimed_sum
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let (mut proof, _, _, _) = gkr_prove(&tree, &mut prover_transcript).unwrap();

        // Tamper with the claimed_sum
        proof.claimed_sum = &proof.claimed_sum + &FE::one();

        // Verify with the tampered proof
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        // The verification should fail. The tampered claimed_sum changes the
        // transcript state, which changes lambda and eta, leading to a sumcheck
        // or gate check failure at the non-trivial layer.
        assert!(
            result.is_err(),
            "GKR verification should fail with tampered claimed_sum"
        );
    }

    #[test]
    fn test_gkr_verify_trivial_layer_gate_check_rejected() {
        // A 4-leaf tree produces:
        //   layer_proofs[0]: trivial (root→layer1, parent_num_vars=0, round_polys=[])
        //   layer_proofs[1]: non-trivial (layer1→leaves, parent_num_vars=1)
        //
        // Tamper with child_claims of the trivial layer_proofs[0] and assert
        // that gkr_verify returns GkrError::GateCheckFailed.
        let nums = vec![
            FE::from(1u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let dens = vec![
            FE::from(2u64),
            FE::from(4u64),
            FE::from(6u64),
            FE::from(8u64),
        ];
        let tree = build_summation_tree(nums, dens);

        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xC0]);
        let (mut proof, _, _, _) = gkr_prove(&tree, &mut prover_transcript).unwrap();

        // layer_proofs[0] is the trivial layer: round_polys must be empty
        assert!(
            proof.layer_proofs[0].sumcheck_proof.round_polys.is_empty(),
            "layer_proofs[0] must be the trivial layer for a 4-leaf tree"
        );

        // Corrupt nl (child_claims[0]) by adding 1 — this breaks the gate equation
        // n_claim*(dl*dr) = nl*dr + nr*dl without altering the transcript ordering
        proof.layer_proofs[0].child_claims[0] += FE::one();

        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xC0]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(
            matches!(result, Err(GkrError::GateCheckFailed { layer: 0 })),
            "Tampered trivial-layer child_claims must be rejected with GateCheckFailed {{ layer: 0 }}, got: {:?}",
            result
        );
    }

    #[test]
    fn test_gkr_prove_verify_roundtrip_16() {
        // 16 leaves with various fractions
        // Tree: 5 layers (sizes 16, 8, 4, 2, 1)
        let nums: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (17..=32).map(|i| FE::from(i as u64)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let (proof, prover_point, prover_n, prover_d) =
            gkr_prove(&tree, &mut prover_transcript).unwrap();

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "GKR verification should succeed for 16 leaves"
        );
        let (verifier_point, verifier_n, verifier_d) = result.unwrap();

        // The verifier's final point must match the prover's
        assert_eq!(verifier_point, prover_point);

        // The verifier's leaf claims must match the prover's
        assert_eq!(verifier_n, prover_n);
        assert_eq!(verifier_d, prover_d);

        // Verify consistency with leaf MLEs
        let expected_n = evaluate_mle(&nums, &verifier_point);
        let expected_d = evaluate_mle(&dens, &verifier_point);
        assert_eq!(verifier_n, expected_n);
        assert_eq!(verifier_d, expected_d);
    }

    // ==================== SVO (Split-Value Optimization) tests ====================

    #[test]
    fn test_gkr_prove_verify_roundtrip_512_svo() {
        // 512 leaves: exercises the SVO path (parent_num_vars = 8 >= SVO_THRESHOLD)
        // Tree: 10 layers (sizes 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
        let n = 512;
        let nums: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (1..=n).map(|i| FE::from(i as u64 + 1000)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0x5F, 0x00]);
        let (proof, prover_point, prover_n, prover_d) =
            gkr_prove(&tree, &mut prover_transcript).unwrap();

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0x5F, 0x00]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "GKR verification should succeed for 512 leaves (SVO path): {:?}",
            result.err()
        );
        let (verifier_point, verifier_n, verifier_d) = result.unwrap();

        // The verifier's final point must match the prover's
        assert_eq!(verifier_point, prover_point);

        // The verifier's leaf claims must match the prover's
        assert_eq!(verifier_n, prover_n);
        assert_eq!(verifier_d, prover_d);

        // Verify consistency with leaf MLEs
        let expected_n = evaluate_mle(&nums, &verifier_point);
        let expected_d = evaluate_mle(&dens, &verifier_point);
        assert_eq!(verifier_n, expected_n);
        assert_eq!(verifier_d, expected_d);
    }

    #[test]
    fn test_gkr_prove_verify_roundtrip_1024_svo() {
        // 1024 leaves: ensures SVO path is exercised at multiple layers
        // parent_num_vars = 9 at the bottom layer
        let n = 1024;
        let nums: Vec<FE> = (1..=n).map(|i| FE::from(i as u64 * 3 + 1)).collect();
        let dens: Vec<FE> = (1..=n).map(|i| FE::from(i as u64 * 7 + 5)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xBB]);
        let (proof, prover_point, prover_n, prover_d) =
            gkr_prove(&tree, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xBB]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "GKR verification should succeed for 1024 leaves (SVO path): {:?}",
            result.err()
        );
        let (verifier_point, verifier_n, verifier_d) = result.unwrap();

        assert_eq!(verifier_point, prover_point);
        assert_eq!(verifier_n, prover_n);
        assert_eq!(verifier_d, prover_d);

        let expected_n = evaluate_mle(&nums, &verifier_point);
        let expected_d = evaluate_mle(&dens, &verifier_point);
        assert_eq!(verifier_n, expected_n);
        assert_eq!(verifier_d, expected_d);
    }

    // ==================== Batch GKR tests ====================

    /// Helper: create a Layer::LogUpGeneric leaf from random-ish fractions.
    fn make_generic_leaf(n_vars: usize) -> Layer<GoldilocksField> {
        let size = 1 << n_vars;
        let denominators: Vec<FE> = (1..=size).map(|i| FE::from(i as u64 + 100)).collect();
        let numerators: Vec<FE> = (1..=size).map(|i| FE::from(i as u64)).collect();
        Layer::LogUpGeneric {
            numerators,
            denominators,
        }
    }

    /// Helper: create a Layer::LogUpSingles leaf.
    fn make_singles_leaf(n_vars: usize) -> Layer<GoldilocksField> {
        let size = 1 << n_vars;
        let denominators: Vec<FE> = (1..=size).map(|i| FE::from(i as u64 + 200)).collect();
        Layer::LogUpSingles { denominators }
    }

    #[test]
    fn test_batch_gkr_same_size_instances() {
        // 3 instances, all with 4 leaves (n_vars=2, 3 layers each)
        let instances: Vec<Vec<Layer<GoldilocksField>>> =
            (0..3).map(|_| gen_layers(make_generic_leaf(2))).collect();

        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, shared_point, final_claims) =
            gkr_prove_batch(instances, &mut prover_transcript);

        assert_eq!(proof.root_claims.len(), 3);
        assert_eq!(final_claims.len(), 3);

        let n_layers: Vec<usize> = proof
            .root_claims
            .iter()
            .map(|_| 2) // all same: 2 reduction steps
            .collect();

        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let result = gkr_verify_batch(&proof, &n_layers, &mut verifier_transcript);
        assert!(result.is_ok(), "batch verify failed: {:?}", result.err());

        let (v_point, v_claims) = result.unwrap();
        assert_eq!(v_point, shared_point);
        assert_eq!(v_claims, final_claims);
    }

    #[test]
    fn test_batch_gkr_mixed_size_instances() {
        // This is the key test: 3 instances with DIFFERENT sizes.
        // n_vars = 2 (4 leaves), 4 (16 leaves), 6 (64 leaves)
        // This exercises the instance_eval_point logic.
        let instances: Vec<Vec<Layer<GoldilocksField>>> = vec![
            gen_layers(make_generic_leaf(2)), // 3 layers, 2 reductions
            gen_layers(make_generic_leaf(4)), // 5 layers, 4 reductions
            gen_layers(make_generic_leaf(6)), // 7 layers, 6 reductions
        ];

        let n_layers: Vec<usize> = instances.iter().map(|l| l.len() - 1).collect();
        assert_eq!(n_layers, vec![2, 4, 6]);

        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, shared_point, final_claims) =
            gkr_prove_batch(instances, &mut prover_transcript);

        assert_eq!(proof.root_claims.len(), 3);

        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let result = gkr_verify_batch(&proof, &n_layers, &mut verifier_transcript);
        assert!(
            result.is_ok(),
            "mixed-size batch verify failed: {:?}",
            result.err()
        );

        let (v_point, v_claims) = result.unwrap();
        assert_eq!(v_point, shared_point);
        assert_eq!(v_claims, final_claims);
    }

    #[test]
    fn test_batch_gkr_mixed_size_with_singles() {
        // Mix of Generic and Singles leaves with different sizes
        let instances: Vec<Vec<Layer<GoldilocksField>>> = vec![
            gen_layers(make_singles_leaf(2)), // 3 layers
            gen_layers(make_generic_leaf(3)), // 4 layers
            gen_layers(make_singles_leaf(5)), // 6 layers
            gen_layers(make_generic_leaf(4)), // 5 layers
        ];

        let n_layers: Vec<usize> = instances.iter().map(|l| l.len() - 1).collect();

        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, shared_point, final_claims) =
            gkr_prove_batch(instances, &mut prover_transcript);

        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let result = gkr_verify_batch(&proof, &n_layers, &mut verifier_transcript);
        assert!(
            result.is_ok(),
            "mixed singles/generic batch verify failed: {:?}",
            result.err()
        );

        let (v_point, v_claims) = result.unwrap();
        assert_eq!(v_point, shared_point);
        assert_eq!(v_claims, final_claims);
    }

    #[test]
    fn test_batch_gkr_many_mixed_instances() {
        // Stress test: 20 instances with sizes varying from n_vars=1 to n_vars=6
        // Mimics the real VM scenario with many tables of different sizes
        let instances: Vec<Vec<Layer<GoldilocksField>>> = (0..20)
            .map(|i| {
                let n_vars = (i % 6) + 1; // 1, 2, 3, 4, 5, 6, 1, 2, ...
                gen_layers(make_generic_leaf(n_vars))
            })
            .collect();

        let n_layers: Vec<usize> = instances.iter().map(|l| l.len() - 1).collect();

        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, shared_point, final_claims) =
            gkr_prove_batch(instances, &mut prover_transcript);

        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let result = gkr_verify_batch(&proof, &n_layers, &mut verifier_transcript);
        assert!(
            result.is_ok(),
            "many mixed instances batch verify failed: {:?}",
            result.err()
        );

        let (v_point, v_claims) = result.unwrap();
        assert_eq!(v_point, shared_point);
        assert_eq!(v_claims, final_claims);
    }

    #[test]
    fn test_instance_eval_point() {
        // Verify the instance_eval_point helper
        let point: Vec<FE> = vec![
            FE::from(10u64), // eta
            FE::from(20u64), // c_0
            FE::from(30u64), // c_1
            FE::from(40u64), // c_2
        ];

        // n_vars = 0: empty
        assert!(instance_eval_point::<GoldilocksField>(&point, 0).is_empty());

        // n_vars = 4 (full): entire point
        assert_eq!(instance_eval_point(&point, 4), point);

        // n_vars = 1: [eta]
        assert_eq!(instance_eval_point(&point, 1), vec![FE::from(10u64)]);

        // n_vars = 2: [eta, c_2] (eta + last 1)
        assert_eq!(
            instance_eval_point(&point, 2),
            vec![FE::from(10u64), FE::from(40u64)]
        );

        // n_vars = 3: [eta, c_1, c_2] (eta + last 2)
        assert_eq!(
            instance_eval_point(&point, 3),
            vec![FE::from(10u64), FE::from(30u64), FE::from(40u64)]
        );
    }
}
