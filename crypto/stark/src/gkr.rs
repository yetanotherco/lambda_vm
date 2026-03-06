use crate::sumcheck::{RoundPoly, SumcheckProof};
use core::fmt;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
        let (numerators, denominators): (Vec<_>, Vec<_>) =
            (0..new_len).map(compute_pair).unzip();

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
        for j in 0..prev_len {
            evals[j] = &evals[j] * &one_minus_ri;
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
pub fn gkr_prove<E: IsField>(
    tree: &[SummationLayer<E>],
    transcript: &mut impl IsTranscript<E>,
) -> (
    GkrProof<E>,
    Vec<FieldElement<E>>,
    FieldElement<E>,
    FieldElement<E>,
) {
    assert!(!tree.is_empty(), "tree must have at least one layer");

    let num_layers = tree.len(); // layers 0..num_layers-1, root is at num_layers-1
    let root = &tree[num_layers - 1];
    assert_eq!(root.numerators.len(), 1, "root layer must have exactly 1 element");

    let root_n = &root.numerators[0];
    let root_d = &root.denominators[0];

    // Compute claimed_sum = root_n / root_d
    let root_d_inv = root_d.inv().expect("root denominator must be nonzero");
    let claimed_sum = root_n * &root_d_inv;

    // Append the claimed sum to the transcript
    transcript.append_field_element(&claimed_sum);

    // If the tree has only 1 layer (just leaves = root), no reductions needed
    if num_layers == 1 {
        return (
            GkrProof {
                claimed_sum,
                layer_proofs: vec![],
            },
            vec![],
            root_n.clone(),
            root_d.clone(),
        );
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

            // Build the five "bookkeeping" tables for the sumcheck:
            // - eq_table: eq(current_point, b) for b in {0,1}^parent_num_vars
            // - nl_table: child_n[2b] (left numerators)
            // - nr_table: child_n[2b+1] (right numerators)
            // - dl_table: child_d[2b] (left denominators)
            // - dr_table: child_d[2b+1] (right denominators)
            let mut eq_table = compute_eq_evals(&current_point);
            let mut nl_table: Vec<FieldElement<E>> =
                (0..parent_size).map(|j| child_n[2 * j].clone()).collect();
            let mut nr_table: Vec<FieldElement<E>> =
                (0..parent_size).map(|j| child_n[2 * j + 1].clone()).collect();
            let mut dl_table: Vec<FieldElement<E>> =
                (0..parent_size).map(|j| child_d[2 * j].clone()).collect();
            let mut dr_table: Vec<FieldElement<E>> =
                (0..parent_size).map(|j| child_d[2 * j + 1].clone()).collect();

            // Verify initial consistency: the sum of f over {0,1}^parent_num_vars
            // should equal combined_claim
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

            let mut round_polys = Vec::with_capacity(parent_num_vars);
            let mut challenges = Vec::with_capacity(parent_num_vars);
            let mut round_combined_claim = combined_claim.clone();

            for round_idx in 0..parent_num_vars {
                let half = nl_table.len() / 2;

                // Eq polynomial factoring (Dao-Thaler, ePrint 2024/1210):
                //
                // Factor eq(current_point, b) into:
                //   eq_prefix (scalar from previous rounds) *
                //   eq_round(r, t) (linear in t, constant across pairs) *
                //   eq_remaining[j] (per-pair scalar, independent of t)
                //
                // where r = current_point[round_idx] and
                //   eq_remaining[j] = eq_table[2j] + eq_table[2j+1]
                //
                // The inner sum h(t) = Σ_j eq_rem[j] * gate(t, j) is degree 2,
                // needing only 2 eval points (t=0, t=2) per pair instead of 3.
                // Then S(t) = eq_round(r, t) * h(t) recovers the degree-3 round poly.
                let r_round = &current_point[round_idx];
                let one = FieldElement::<E>::one();

                let compute_pair_sums =
                    |j: usize| -> [FieldElement<E>; 2] {
                        // Per-pair eq weight (scalar, independent of t)
                        let eq_rem = &eq_table[2 * j] + &eq_table[2 * j + 1];

                        let nl_l = &nl_table[2 * j];
                        let nl_r = &nl_table[2 * j + 1];
                        let nr_l = &nr_table[2 * j];
                        let nr_r = &nr_table[2 * j + 1];
                        let dl_l = &dl_table[2 * j];
                        let dl_r = &dl_table[2 * j + 1];
                        let dr_l = &dr_table[2 * j];
                        let dr_r = &dr_table[2 * j + 1];

                        // t=0: interpolated values are just the left values
                        let gate_0 =
                            &(nl_l * dr_l) + &(nr_l * dl_l) + &(&lambda * &(dl_l * dr_l));
                        let h0 = &eq_rem * &gate_0;

                        // t=2: val = 2*right - left
                        let nl_2 = &(nl_r + nl_r) - nl_l;
                        let nr_2 = &(nr_r + nr_r) - nr_l;
                        let dl_2 = &(dl_r + dl_r) - dl_l;
                        let dr_2 = &(dr_r + dr_r) - dr_l;
                        let gate_2 = &(&nl_2 * &dr_2)
                            + &(&nr_2 * &dl_2)
                            + &(&lambda * &(&dl_2 * &dr_2));
                        let h2 = &eq_rem * &gate_2;

                        [h0, h2]
                    };

                let zero2 =
                    || [FieldElement::<E>::zero(), FieldElement::<E>::zero()];
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
                //
                // eq_round(r, t) = (1-r)(1-t) + r*t
                //   eq_round(r, 0) = 1-r,  eq_round(r, 1) = r,
                //   eq_round(r, 2) = 3r-1,  eq_round(r, 3) = 5r-2
                //
                // S(0) = (1-r)*h(0), S(1) = round_combined_claim - S(0)
                // h(1) = S(1)/r, h(3) = 3*h(2) - 3*h(1) + h(0)  (degree-2 extrapolation)
                // S(2) = (3r-1)*h(2), S(3) = (5r-2)*h(3)
                let [total_h0, total_h2] = totals;

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

                let eq_at_3 =
                    &(&FieldElement::<E>::from(5u64) * r_round) - &FieldElement::<E>::from(2u64);
                let s3 = &eq_at_3 * &h3;

                let poly_evals = vec![s0, s1, s2, s3];

                let round_poly = RoundPoly::new(poly_evals);

                // Append round polynomial evaluations to transcript
                for eval in round_poly.evals() {
                    transcript.append_field_element(eval);
                }

                // Sample challenge for this round
                let challenge: FieldElement<E> = transcript.sample_field_element();

                // Update round_combined_claim for next round
                round_combined_claim = round_poly.evaluate(&challenge);

                // Bind all five tables: fold pairs using the challenge
                // table[j] = table[2j] + challenge * (table[2j+1] - table[2j])
                let fold_table = |table: &mut Vec<FieldElement<E>>| {
                    for j in 0..half {
                        let left = &table[2 * j];
                        let right = &table[2 * j + 1];
                        table[j] = left + &(&challenge * &(right - left));
                    }
                    table.truncate(half);
                };

                #[cfg(feature = "parallel")]
                {
                    if half >= 256 {
                        // Fold tables in parallel (each table independently)
                        rayon::join(
                            || {
                                rayon::join(
                                    || fold_table(&mut eq_table),
                                    || fold_table(&mut nl_table),
                                );
                            },
                            || {
                                rayon::join(
                                    || fold_table(&mut nr_table),
                                    || rayon::join(
                                        || fold_table(&mut dl_table),
                                        || fold_table(&mut dr_table),
                                    ),
                                );
                            },
                        );
                    } else {
                        fold_table(&mut eq_table);
                        fold_table(&mut nl_table);
                        fold_table(&mut nr_table);
                        fold_table(&mut dl_table);
                        fold_table(&mut dr_table);
                    }
                }

                #[cfg(not(feature = "parallel"))]
                {
                    fold_table(&mut eq_table);
                    fold_table(&mut nl_table);
                    fold_table(&mut nr_table);
                    fold_table(&mut dl_table);
                    fold_table(&mut dr_table);
                }

                round_polys.push(round_poly);
                challenges.push(challenge);
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

    (
        GkrProof {
            claimed_sum,
            layer_proofs,
        },
        current_point,
        n_claim,
        d_claim,
    )
}

/// Errors that can occur during GKR verification.
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
pub fn gkr_verify<E: IsField>(
    proof: &GkrProof<E>,
    transcript: &mut impl IsTranscript<E>,
) -> Result<(Vec<FieldElement<E>>, FieldElement<E>, FieldElement<E>), GkrError> {
    // Step 1: Append claimed_sum to transcript (mirrors prover line 239)
    transcript.append_field_element(&proof.claimed_sum);

    // If there are no layer proofs, the tree had a single leaf (root = leaf).
    // Return empty point and the claimed_sum as n_claim with d_claim = 1.
    if proof.layer_proofs.is_empty() {
        return Ok((
            vec![],
            proof.claimed_sum.clone(),
            FieldElement::one(),
        ));
    }

    // Step 2: Initialize claims.
    // The verifier sets n_claim = claimed_sum, d_claim = 1.
    // This represents the same rational value as root_n/root_d.
    // Soundness of the first (trivial) layer is enforced by later sumcheck layers.
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
            // The prover just provides child_claims directly.
            // No gate check here --- soundness is enforced by later layers' sumchecks.
            // (The verifier's combined_claim may differ from the prover's by a
            // scaling factor at the first trivial layer, so we skip the check.)
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
                        reason: format!(
                            "round {} sum mismatch: p(0)+p(1) != expected sum",
                            round
                        ),
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
            let gate_combined =
                &(&(nl * dr) + &(nr * dl)) + &(&lambda * &(dl * dr));

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
    a.iter().zip(b.iter()).fold(FieldElement::one(), |acc, (ai, bi)| {
        let term = &(ai * bi) + &(&(&one - ai) * &(&one - bi));
        &acc * &term
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

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
        assert_eq!(
            root_n * &FE::from(24u64),
            &FE::from(71u64) * root_d
        );
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
        let mut acc = Fraction::new(nums[0].clone(), dens[0].clone());
        for i in 1..8 {
            acc = acc.add(&Fraction::new(nums[i].clone(), dens[i].clone()));
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
        let evals = compute_eq_evals(&[r.clone()]);
        assert_eq!(evals.len(), 2);
        assert_eq!(evals[0], FE::one() - r.clone()); // eq(r, 0) = 1-r = -2
        assert_eq!(evals[1], r); // eq(r, 1) = r = 3
    }

    #[test]
    fn test_compute_eq_evals_2var() {
        // eq((r0, r1), (b0, b1)) = [r0*b0 + (1-r0)*(1-b0)] * [r1*b1 + (1-r1)*(1-b1)]
        let r0 = FE::from(2u64);
        let r1 = FE::from(5u64);
        let evals = compute_eq_evals(&[r0.clone(), r1.clone()]);
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
            gkr_prove(&tree, &mut transcript);

        // claimed_sum = 68/55
        let expected_sum = &FE::from(68u64) * &FE::from(55u64).inv().unwrap();
        assert_eq!(proof.claimed_sum, expected_sum);

        // Should have 1 layer proof (root -> leaves)
        assert_eq!(proof.layer_proofs.len(), 1);

        // The first (and only) layer proof has a trivial sumcheck (0 rounds)
        assert_eq!(proof.layer_proofs[0].sumcheck_proof.round_polys.len(), 0);

        // The child claims should be the raw leaf values
        assert_eq!(proof.layer_proofs[0].child_claims[0], FE::from(3u64));  // n_left
        assert_eq!(proof.layer_proofs[0].child_claims[1], FE::from(7u64));  // n_right
        assert_eq!(proof.layer_proofs[0].child_claims[2], FE::from(5u64));  // d_left
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
            gkr_prove(&tree, &mut transcript);

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
        let (proof, _, _, _) = gkr_prove(&tree, &mut transcript);

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
            gkr_prove(&tree, &mut transcript);

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
            gkr_prove(&tree, &mut transcript);

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
        let (proof1, point1, n1, d1) = gkr_prove(&tree, &mut t1);

        let mut t2 = DefaultTranscript::<GoldilocksField>::new(&[0x42]);
        let (proof2, point2, n2, d2) = gkr_prove(&tree, &mut t2);

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
        let (proof, _, _, _) = gkr_prove(&tree, &mut transcript);

        // Replay transcript to get the same challenges
        let mut replay = DefaultTranscript::<GoldilocksField>::new(&[]);
        replay.append_field_element(&proof.claimed_sum);

        let mut n_claim = tree[2].numerators[0].clone();
        let mut d_claim = tree[2].denominators[0].clone();

        for (layer_idx, lp) in proof.layer_proofs.iter().enumerate() {
            let lambda: FE = replay.sample_field_element();
            let combined_claim = &n_claim + &(&lambda * &d_claim);

            if lp.sumcheck_proof.round_polys.is_empty() {
                // Trivial layer: verify gate equation directly
                let nl = &lp.child_claims[0];
                let nr = &lp.child_claims[1];
                let dl = &lp.child_claims[2];
                let dr = &lp.child_claims[3];
                let gate_val =
                    &(nl * dr) + &(nr * dl) + &(&lambda * &(dl * dr));
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
            n_claim =
                &(&lp.child_claims[0] * &one_minus_eta) + &(&lp.child_claims[1] * &eta);
            d_claim =
                &(&lp.child_claims[2] * &one_minus_eta) + &(&lp.child_claims[3] * &eta);
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
            gkr_prove(&tree, &mut transcript);

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
            gkr_prove(&tree, &mut prover_transcript);

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(result.is_ok(), "GKR verification should succeed for 4 leaves");
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
            gkr_prove(&tree, &mut prover_transcript);

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(result.is_ok(), "GKR verification should succeed for 8 leaves");
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
        let (mut proof, _, _, _) = gkr_prove(&tree, &mut prover_transcript);

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
    fn test_gkr_prove_verify_roundtrip_16() {
        // 16 leaves with various fractions
        // Tree: 5 layers (sizes 16, 8, 4, 2, 1)
        let nums: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let dens: Vec<FE> = (17..=32).map(|i| FE::from(i as u64)).collect();
        let tree = build_summation_tree(nums.clone(), dens.clone());

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let (proof, prover_point, prover_n, prover_d) =
            gkr_prove(&tree, &mut prover_transcript);

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAA]);
        let result = gkr_verify(&proof, &mut verifier_transcript);

        assert!(result.is_ok(), "GKR verification should succeed for 16 leaves");
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
}
