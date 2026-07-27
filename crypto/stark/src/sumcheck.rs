use core::fmt;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

/// A degree-d univariate polynomial represented by its evaluations at the
/// integer nodes 0, 1, ..., d. This is the polynomial that the sumcheck prover
/// sends to the verifier in each round.
///
/// Storing evaluations (rather than coefficients) is natural for the sumcheck
/// protocol because:
/// - The prover constructs the polynomial by evaluating it at small integer points.
/// - The verifier only needs to check p(0) + p(1) and evaluate at a random challenge.
/// - Lagrange interpolation over integer nodes is cheap (small denominators).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct RoundPoly<E: IsField> {
    /// Evaluations at x = 0, 1, ..., d where d = evals.len() - 1.
    ///
    /// Deserialization bypasses [`RoundPoly::new`]'s non-empty assert, so
    /// consumers of untrusted proofs MUST length-check via
    /// [`RoundPoly::num_evals`] before calling [`RoundPoly::sum_at_binary`] or
    /// [`RoundPoly::evaluate`] (the batch GKR verifier rejects wrong shapes).
    evals: Vec<FieldElement<E>>,
}

impl<E: IsField> RoundPoly<E> {
    /// Create a new `RoundPoly` from evaluations at x = 0, 1, ..., d.
    ///
    /// `evals[i]` is the polynomial evaluated at x = i.
    /// The polynomial has degree `evals.len() - 1`.
    pub fn new(evals: Vec<FieldElement<E>>) -> Self {
        assert!(
            !evals.is_empty(),
            "RoundPoly must have at least one evaluation"
        );
        Self { evals }
    }

    /// Returns p(0) + p(1).
    ///
    /// In the sumcheck protocol, the verifier checks that this sum matches the
    /// claimed sum for the current round. This is the key consistency check.
    pub fn sum_at_binary(&self) -> FieldElement<E> {
        assert!(
            self.evals.len() >= 2,
            "sum_at_binary requires at least 2 evaluations (at 0 and 1)"
        );
        &self.evals[0] + &self.evals[1]
    }

    /// Evaluate the polynomial at an arbitrary field element using Lagrange
    /// interpolation over integer nodes 0, 1, ..., d.
    ///
    /// Given evaluations y_0, ..., y_d at nodes 0, 1, ..., d, the Lagrange
    /// interpolation formula is:
    ///
    ///   p(x) = sum_{i=0}^{d} y_i * prod_{j != i} (x - j) / (i - j)
    ///
    /// The denominators (i - j) for integer nodes are just products of small
    /// integers, so we precompute them as field elements.
    pub fn evaluate(&self, point: &FieldElement<E>) -> FieldElement<E> {
        let d = self.evals.len() - 1;

        // Special case: constant polynomial (degree 0)
        if d == 0 {
            return self.evals[0].clone();
        }

        // Precompute (point - j) for j = 0, ..., d
        let point_minus_j: Vec<FieldElement<E>> = (0..=d)
            .map(|j| point - &FieldElement::from(j as u64))
            .collect();

        // Check if point is one of the integer nodes (avoid division by zero)
        for (j, pm) in point_minus_j.iter().enumerate() {
            if *pm == FieldElement::zero() {
                return self.evals[j].clone();
            }
        }

        // Barycentric Lagrange interpolation with batch inversion.
        //
        // Barycentric weights: w_i = 1 / prod_{j≠i}(i - j) for integer nodes.
        // We batch-invert the weights together with point_minus_j to use only
        // 1 field inversion total (instead of 2(d+1) individual inversions).

        // Compute barycentric weight denominators: prod_{j≠i}(i - j)
        let mut to_invert: Vec<FieldElement<E>> = (0..=d)
            .map(|i| {
                let mut denom = FieldElement::<E>::one();
                for j in 0..=d {
                    if j != i {
                        denom = &denom * &FieldElement::from((i as i64) - (j as i64));
                    }
                }
                denom
            })
            .collect();

        // Append point_minus_j values; batch-invert everything in one call
        to_invert.extend(point_minus_j.iter().cloned());
        FieldElement::inplace_batch_inverse(&mut to_invert).expect("All values are nonzero");

        let (w_inv, pm_inv) = to_invert.split_at(d + 1);

        // master_product = prod_{j=0}^{d} (point - j)
        let master_product = point_minus_j
            .iter()
            .fold(FieldElement::one(), |acc, v| &acc * v);

        // result = master_product * Σ_i (evals[i] * w_inv[i] * pm_inv[i])
        let mut sum = FieldElement::zero();
        for i in 0..=d {
            sum = &sum + &(&self.evals[i] * &(&w_inv[i] * &pm_inv[i]));
        }

        &master_product * &sum
    }

    /// Returns the number of evaluations (degree + 1).
    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    /// Returns a reference to the evaluations.
    pub fn evals(&self) -> &[FieldElement<E>] {
        &self.evals
    }
}

/// Proof produced by the sumcheck prover: one round polynomial per variable.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct SumcheckProof<E: IsField> {
    pub round_polys: Vec<RoundPoly<E>>,
}

/// Run the sumcheck interactive proof (made non-interactive via Fiat-Shamir).
///
/// Proves that `claimed_sum == sum_{x in {0,1}^num_vars} f(x)` where `evals`
/// holds the evaluations of f over the Boolean hypercube in little-endian bit
/// order (index = b_0 + 2*b_1 + ...).
///
/// # Arguments
/// - `evals`: evaluations of f over {0,1}^num_vars, length must be 2^num_vars
/// - `claimed_sum`: the asserted sum
/// - `num_vars`: number of Boolean variables
/// - `max_degree`: maximum degree of round polynomials (need max_degree+1 evaluation points)
/// - `transcript`: Fiat-Shamir transcript for deterministic challenge sampling
///
/// # Returns
/// `(proof, challenges)` where `proof` contains one round polynomial per variable
/// and `challenges` is the vector of random points sampled during the protocol.
pub fn sumcheck_prove<E: IsField>(
    evals: &[FieldElement<E>],
    claimed_sum: &FieldElement<E>,
    num_vars: usize,
    max_degree: usize,
    transcript: &mut impl IsTranscript<E>,
) -> (SumcheckProof<E>, Vec<FieldElement<E>>) {
    assert_eq!(
        evals.len(),
        1 << num_vars,
        "evals length must be 2^num_vars"
    );
    assert!(
        max_degree >= 1,
        "max_degree must be at least 1 for a meaningful sumcheck"
    );

    let mut table = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut round_claimed_sum = claimed_sum.clone();

    for _round in 0..num_vars {
        let half = table.len() / 2;

        // p(0) = sum of left halves (zero multiplications)
        let mut p0 = FieldElement::<E>::zero();
        for j in 0..half {
            p0 = &p0 + &table[2 * j];
        }

        // p(1) = round_claimed_sum - p(0) (free, from sumcheck identity p(0)+p(1)=sum)
        let p1 = &round_claimed_sum - &p0;

        let mut poly_evals = Vec::with_capacity(max_degree + 1);
        poly_evals.push(p0);
        poly_evals.push(p1);

        // For t >= 2: use delta form.
        // Linear interpolation at t: table[2j] + t * (table[2j+1] - table[2j])
        // p(t) = Σ_j [table[2j] + t * delta_j] where delta_j = table[2j+1] - table[2j]
        for t in 2..=max_degree {
            let t_fe = FieldElement::<E>::from(t as u64);
            let mut sum = FieldElement::<E>::zero();
            for j in 0..half {
                let delta = &table[2 * j + 1] - &table[2 * j];
                sum = &sum + &(&table[2 * j] + &(&t_fe * &delta));
            }
            poly_evals.push(sum);
        }

        let round_poly = RoundPoly::new(poly_evals);

        // Append all evaluations of the round polynomial to the transcript
        for eval in round_poly.evals() {
            transcript.append_field_element(eval);
        }

        // Sample the challenge for this round
        let challenge: FieldElement<E> = transcript.sample_field_element();

        // Update round_claimed_sum for next round: p(challenge)
        round_claimed_sum = round_poly.evaluate(&challenge);

        // Bind the table: fold using the challenge
        // table[j] = table[2j] + challenge * (table[2j+1] - table[2j])
        for j in 0..half {
            let left = &table[2 * j];
            let right = &table[2 * j + 1];
            table[j] = left + &(&challenge * &(right - left));
        }
        table.truncate(half);

        round_polys.push(round_poly);
        challenges.push(challenge);
    }

    (SumcheckProof { round_polys }, challenges)
}

/// Errors that can occur during sumcheck verification.
#[derive(Debug, Clone)]
pub enum SumcheckError {
    /// The proof contains a different number of round polynomials than expected.
    WrongNumberOfRounds { expected: usize, got: usize },
    /// The round polynomial's p(0) + p(1) does not match the expected sum.
    RoundSumMismatch { round: usize },
}

impl fmt::Display for SumcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumcheckError::WrongNumberOfRounds { expected, got } => {
                write!(
                    f,
                    "wrong number of rounds: expected {}, got {}",
                    expected, got
                )
            }
            SumcheckError::RoundSumMismatch { round } => {
                write!(f, "round sum mismatch at round {}", round)
            }
        }
    }
}

/// Verify a sumcheck proof (non-interactive via Fiat-Shamir).
///
/// Checks that the prover's round polynomials are consistent with the claimed
/// sum. The transcript operations must exactly mirror those of `sumcheck_prove`
/// so that the same challenges are derived.
///
/// # Arguments
/// - `proof`: the sumcheck proof containing one round polynomial per variable
/// - `claimed_sum`: the asserted sum over the Boolean hypercube
/// - `num_vars`: number of Boolean variables
/// - `transcript`: Fiat-Shamir transcript (must have the same seed as the prover's)
///
/// # Returns
/// `Ok((challenges, final_eval))` where `challenges` is the random point and
/// `final_eval` is the evaluation claim at that point (i.e., the last round
/// polynomial evaluated at the last challenge).
pub fn sumcheck_verify<E: IsField>(
    proof: &SumcheckProof<E>,
    claimed_sum: &FieldElement<E>,
    num_vars: usize,
    transcript: &mut impl IsTranscript<E>,
) -> Result<(Vec<FieldElement<E>>, FieldElement<E>), SumcheckError> {
    if proof.round_polys.len() != num_vars {
        return Err(SumcheckError::WrongNumberOfRounds {
            expected: num_vars,
            got: proof.round_polys.len(),
        });
    }

    let mut current_sum = claimed_sum.clone();
    let mut challenges = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        // Check that p_r(0) + p_r(1) == current_sum
        if proof.round_polys[round].sum_at_binary() != current_sum {
            return Err(SumcheckError::RoundSumMismatch { round });
        }

        // Append all evaluations of the round polynomial to the transcript
        // (must match the prover exactly)
        for eval in proof.round_polys[round].evals() {
            transcript.append_field_element(eval);
        }

        // Sample the challenge for this round
        let challenge: FieldElement<E> = transcript.sample_field_element();

        // Update the running sum: next round's claimed sum is p_r(challenge)
        current_sum = proof.round_polys[round].evaluate(&challenge);

        challenges.push(challenge);
    }

    Ok((challenges, current_sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField;

    // Use the base Goldilocks field for tests. The cubic extension would work
    // identically since all operations are generic over IsField.
    type FE = FieldElement<GoldilocksField>;

    /// Helper: evaluate a polynomial given in coefficient form at a point.
    /// coeffs[i] is the coefficient of x^i.
    fn eval_poly_coeffs(coeffs: &[FE], x: &FE) -> FE {
        let mut result = FE::zero();
        let mut power = FE::one();
        for c in coeffs {
            result += c * power;
            power *= x;
        }
        result
    }

    #[test]
    fn test_sum_at_binary_linear() {
        // p(x) = 3 + 5x => p(0) = 3, p(1) = 8
        // sum_at_binary = 3 + 8 = 11
        let evals = vec![FE::from(3u64), FE::from(8u64)];
        let poly = RoundPoly::new(evals);
        assert_eq!(poly.sum_at_binary(), FE::from(11u64));
    }

    #[test]
    fn test_sum_at_binary_quadratic() {
        // p(x) = 2x^2 + 3x + 1 => p(0) = 1, p(1) = 6, p(2) = 15
        // sum_at_binary = p(0) + p(1) = 1 + 6 = 7
        let evals = vec![FE::from(1u64), FE::from(6u64), FE::from(15u64)];
        let poly = RoundPoly::new(evals);
        assert_eq!(poly.sum_at_binary(), FE::from(7u64));
    }

    #[test]
    fn test_evaluate_at_nodes_linear() {
        // p(x) = 3 + 5x => p(0) = 3, p(1) = 8
        // Evaluating at the nodes should return the stored evaluations.
        let evals = vec![FE::from(3u64), FE::from(8u64)];
        let poly = RoundPoly::new(evals);

        assert_eq!(poly.evaluate(&FE::from(0u64)), FE::from(3u64));
        assert_eq!(poly.evaluate(&FE::from(1u64)), FE::from(8u64));
    }

    #[test]
    fn test_evaluate_at_nodes_quadratic() {
        // p(x) = 2x^2 + 3x + 1 => p(0) = 1, p(1) = 6, p(2) = 15
        let evals = vec![FE::from(1u64), FE::from(6u64), FE::from(15u64)];
        let poly = RoundPoly::new(evals);

        assert_eq!(poly.evaluate(&FE::from(0u64)), FE::from(1u64));
        assert_eq!(poly.evaluate(&FE::from(1u64)), FE::from(6u64));
        assert_eq!(poly.evaluate(&FE::from(2u64)), FE::from(15u64));
    }

    #[test]
    fn test_evaluate_at_random_point_linear() {
        // p(x) = 3 + 5x
        // p(0) = 3, p(1) = 8
        // p(7) = 3 + 35 = 38
        let evals = vec![FE::from(3u64), FE::from(8u64)];
        let poly = RoundPoly::new(evals);

        let point = FE::from(7u64);
        let expected = FE::from(38u64);
        assert_eq!(poly.evaluate(&point), expected);
    }

    #[test]
    fn test_evaluate_at_random_point_quadratic() {
        // p(x) = 2x^2 + 3x + 1
        // p(0) = 1, p(1) = 6, p(2) = 15
        // p(5) = 2*25 + 3*5 + 1 = 50 + 15 + 1 = 66
        let coeffs = [FE::from(1u64), FE::from(3u64), FE::from(2u64)];
        let evals = vec![
            eval_poly_coeffs(&coeffs, &FE::from(0u64)),
            eval_poly_coeffs(&coeffs, &FE::from(1u64)),
            eval_poly_coeffs(&coeffs, &FE::from(2u64)),
        ];
        let poly = RoundPoly::new(evals);

        let point = FE::from(5u64);
        let expected = eval_poly_coeffs(&coeffs, &point);
        assert_eq!(poly.evaluate(&point), expected);
        assert_eq!(expected, FE::from(66u64));
    }

    #[test]
    fn test_evaluate_at_random_point_cubic() {
        // p(x) = x^3 + 2x^2 + 3x + 4
        // Need 4 evaluation points: 0, 1, 2, 3
        let coeffs = [
            FE::from(4u64),
            FE::from(3u64),
            FE::from(2u64),
            FE::from(1u64),
        ];
        let evals: Vec<FE> = (0..4)
            .map(|i| eval_poly_coeffs(&coeffs, &FE::from(i as u64)))
            .collect();
        let poly = RoundPoly::new(evals);

        // p(10) = 1000 + 200 + 30 + 4 = 1234
        let point = FE::from(10u64);
        let expected = eval_poly_coeffs(&coeffs, &point);
        assert_eq!(poly.evaluate(&point), expected);
        assert_eq!(expected, FE::from(1234u64));
    }

    #[test]
    fn test_evaluate_at_large_field_element() {
        // Test with a large field element as the evaluation point to ensure
        // field arithmetic works correctly (not just small integers).
        let coeffs = [FE::from(7u64), FE::from(11u64), FE::from(13u64)];
        let evals: Vec<FE> = (0..3)
            .map(|i| eval_poly_coeffs(&coeffs, &FE::from(i as u64)))
            .collect();
        let poly = RoundPoly::new(evals);

        // Use a large evaluation point
        let point = FE::from(1_000_000_007u64);
        let expected = eval_poly_coeffs(&coeffs, &point);
        assert_eq!(poly.evaluate(&point), expected);
    }

    #[test]
    fn test_constant_polynomial() {
        // p(x) = 42 (constant)
        let evals = vec![FE::from(42u64)];
        let poly = RoundPoly::new(evals);

        // A constant polynomial should evaluate to the same value everywhere
        assert_eq!(poly.evaluate(&FE::from(0u64)), FE::from(42u64));
        assert_eq!(poly.evaluate(&FE::from(100u64)), FE::from(42u64));
        assert_eq!(poly.evaluate(&FE::from(999u64)), FE::from(42u64));
    }

    #[test]
    fn test_num_evals() {
        let evals = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let poly = RoundPoly::new(evals);
        assert_eq!(poly.num_evals(), 3);
    }

    #[test]
    fn test_evals_accessor() {
        let evals = vec![FE::from(10u64), FE::from(20u64)];
        let poly = RoundPoly::new(evals);
        assert_eq!(poly.evals()[0], FE::from(10u64));
        assert_eq!(poly.evals()[1], FE::from(20u64));
    }

    #[test]
    #[should_panic(expected = "RoundPoly must have at least one evaluation")]
    fn test_empty_evals_panics() {
        let _poly = RoundPoly::<GoldilocksField>::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "sum_at_binary requires at least 2 evaluations")]
    fn test_sum_at_binary_single_eval_panics() {
        let poly = RoundPoly::new(vec![FE::from(1u64)]);
        let _ = poly.sum_at_binary();
    }

    #[test]
    fn test_evaluate_consistency_with_coefficients() {
        // Build a degree-4 polynomial from coefficients, generate evaluations
        // at 0..=4, construct RoundPoly, and verify evaluate() at many points.
        let coeffs = [
            FE::from(5u64),
            FE::from(3u64),
            FE::from(7u64),
            FE::from(2u64),
            FE::from(1u64),
        ];
        let evals: Vec<FE> = (0..5)
            .map(|i| eval_poly_coeffs(&coeffs, &FE::from(i as u64)))
            .collect();
        let poly = RoundPoly::new(evals);

        // Check at several points including nodes and non-nodes
        for x in [0u64, 1, 2, 3, 4, 5, 10, 100, 12345] {
            let point = FE::from(x);
            let expected = eval_poly_coeffs(&coeffs, &point);
            assert_eq!(poly.evaluate(&point), expected, "Mismatch at x = {}", x);
        }
    }

    // ==================== Sumcheck prover tests ====================

    #[test]
    fn test_sumcheck_prove_2var_linear() {
        // f(x1, x2) = 3*x1 + 5*x2 + 7
        // Hypercube evaluations in little-endian bit order:
        //   index 0 = (x1=0, x2=0): f = 7
        //   index 1 = (x1=1, x2=0): f = 10
        //   index 2 = (x1=0, x2=1): f = 12
        //   index 3 = (x1=1, x2=1): f = 15
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64); // 7 + 10 + 12 + 15

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, challenges) = sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut transcript);

        // Should have 2 round polynomials (one per variable)
        assert_eq!(proof.round_polys.len(), 2);
        assert_eq!(challenges.len(), 2);

        // First round poly: sum over x2 of f(x1, x2)
        //   p1(0) = f(0,0) + f(0,1) = 7 + 12 = 19
        //   p1(1) = f(1,0) + f(1,1) = 10 + 15 = 25
        //   p1(0) + p1(1) = 19 + 25 = 44 = claimed_sum
        assert_eq!(proof.round_polys[0].sum_at_binary(), claimed_sum);
        assert_eq!(proof.round_polys[0].evals()[0], FE::from(19u64));
        assert_eq!(proof.round_polys[0].evals()[1], FE::from(25u64));

        // Each round poly should have max_degree+1 = 2 evaluations
        assert_eq!(proof.round_polys[0].num_evals(), 2);
        assert_eq!(proof.round_polys[1].num_evals(), 2);

        // Second round: the claimed sum for round 2 is p1(r1) where r1 = challenges[0].
        // Verify that p2(0) + p2(1) = p1(r1).
        let r1_eval = proof.round_polys[0].evaluate(&challenges[0]);
        assert_eq!(proof.round_polys[1].sum_at_binary(), r1_eval);
    }

    #[test]
    fn test_sumcheck_prove_1var() {
        // f(x) = 2x + 3 => f(0) = 3, f(1) = 5
        // Sum = 3 + 5 = 8
        let evals = vec![FE::from(3u64), FE::from(5u64)];
        let claimed_sum = FE::from(8u64);

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, challenges) = sumcheck_prove(&evals, &claimed_sum, 1, 1, &mut transcript);

        assert_eq!(proof.round_polys.len(), 1);
        assert_eq!(challenges.len(), 1);
        assert_eq!(proof.round_polys[0].sum_at_binary(), claimed_sum);
        assert_eq!(proof.round_polys[0].evals()[0], FE::from(3u64));
        assert_eq!(proof.round_polys[0].evals()[1], FE::from(5u64));
    }

    #[test]
    fn test_sumcheck_prove_3var_constant() {
        // f(x1, x2, x3) = 1 for all inputs
        // Sum = 8 (2^3 points, each contributing 1)
        let evals = vec![FE::one(); 8];
        let claimed_sum = FE::from(8u64);

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, challenges) = sumcheck_prove(&evals, &claimed_sum, 3, 1, &mut transcript);

        assert_eq!(proof.round_polys.len(), 3);

        // First round: p(0) = 4, p(1) = 4, sum = 8
        assert_eq!(proof.round_polys[0].sum_at_binary(), claimed_sum);
        assert_eq!(proof.round_polys[0].evals()[0], FE::from(4u64));
        assert_eq!(proof.round_polys[0].evals()[1], FE::from(4u64));

        // Each subsequent round's sum should match the previous round's evaluation at challenge
        for r in 1..3 {
            let prev_eval = proof.round_polys[r - 1].evaluate(&challenges[r - 1]);
            assert_eq!(proof.round_polys[r].sum_at_binary(), prev_eval);
        }
    }

    #[test]
    fn test_sumcheck_prove_higher_degree() {
        // Test with max_degree = 3 (4 evaluation points per round poly).
        // For a multilinear f, higher-degree evaluations are determined by
        // the degree-1 interpolation, so they are just linear extrapolation.
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64);

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, challenges) = sumcheck_prove(&evals, &claimed_sum, 2, 3, &mut transcript);

        assert_eq!(proof.round_polys.len(), 2);

        // Each round poly should have max_degree+1 = 4 evaluations
        assert_eq!(proof.round_polys[0].num_evals(), 4);
        assert_eq!(proof.round_polys[1].num_evals(), 4);

        // p(0) + p(1) must still equal claimed sum
        assert_eq!(proof.round_polys[0].sum_at_binary(), claimed_sum);

        // Since the underlying function is multilinear, the round poly is degree 1.
        // Verify that evaluations at t=2 and t=3 are consistent with linear interpolation.
        let p0 = &proof.round_polys[0].evals()[0];
        let p1 = &proof.round_polys[0].evals()[1];
        // p(t) = p0*(1-t) + p1*t = p0 + (p1 - p0)*t
        let slope = p1 - p0;
        let p2_expected = p0 + (slope * FE::from(2u64));
        let p3_expected = p0 + (slope * FE::from(3u64));
        assert_eq!(proof.round_polys[0].evals()[2], p2_expected);
        assert_eq!(proof.round_polys[0].evals()[3], p3_expected);

        // Consistency between rounds
        for r in 1..2 {
            let prev_eval = proof.round_polys[r - 1].evaluate(&challenges[r - 1]);
            assert_eq!(proof.round_polys[r].sum_at_binary(), prev_eval);
        }
    }

    #[test]
    fn test_sumcheck_prove_deterministic() {
        // Same inputs and transcript seed should produce identical proofs
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64);

        let mut t1 = DefaultTranscript::<GoldilocksField>::new(&[0x42]);
        let (proof1, challenges1) = sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut t1);

        let mut t2 = DefaultTranscript::<GoldilocksField>::new(&[0x42]);
        let (proof2, challenges2) = sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut t2);

        assert_eq!(challenges1, challenges2);
        for (rp1, rp2) in proof1.round_polys.iter().zip(proof2.round_polys.iter()) {
            assert_eq!(rp1.evals(), rp2.evals());
        }
    }

    #[test]
    fn test_sumcheck_prove_final_value() {
        // After all rounds, the table should contain a single element which is
        // f evaluated at the random challenge point. We verify by checking that
        // the last round poly evaluated at the last challenge gives this value.
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64);

        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, challenges) = sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut transcript);

        // Manually compute f(r1, r2) by multilinear evaluation
        let r1 = &challenges[0];
        let r2 = &challenges[1];
        // f(r1, r2) = f(0,0)*(1-r1)*(1-r2) + f(1,0)*r1*(1-r2) + f(0,1)*(1-r1)*r2 + f(1,1)*r1*r2
        let one = FE::one();
        let one_minus_r1 = one - r1;
        let one_minus_r2 = one - r2;
        let f_at_r = ((evals[0] * one_minus_r1) * one_minus_r2)
            + ((evals[1] * r1) * one_minus_r2)
            + ((evals[2] * one_minus_r1) * r2)
            + ((evals[3] * r1) * r2);

        // The last round poly evaluated at the last challenge should give f(r1, r2)
        let last_round_eval = proof.round_polys[1].evaluate(&challenges[1]);
        assert_eq!(last_round_eval, f_at_r);
    }

    #[test]
    #[should_panic(expected = "evals length must be 2^num_vars")]
    fn test_sumcheck_prove_wrong_length_panics() {
        let evals = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)]; // length 3, not a power of 2
        let mut transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let _ = sumcheck_prove(&evals, &FE::from(6u64), 2, 1, &mut transcript);
    }

    // ==================== Sumcheck verifier tests ====================

    #[test]
    fn test_sumcheck_verify() {
        // f(x1, x2) = 3*x1 + 5*x2 + 7
        // Hypercube evaluations (little-endian bit order):
        //   (0,0)=7, (1,0)=10, (0,1)=12, (1,1)=15
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64); // 7 + 10 + 12 + 15

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, prover_challenges) =
            sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut prover_transcript);

        // Verify with a fresh transcript (same seed)
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let result = sumcheck_verify(&proof, &claimed_sum, 2, &mut verifier_transcript);

        assert!(result.is_ok(), "Verification should succeed");
        let (challenges, _final_eval) = result.unwrap();

        // Verifier must derive the same challenges as the prover
        assert_eq!(challenges, prover_challenges);
    }

    #[test]
    fn test_sumcheck_verify_wrong_sum() {
        // f(x1, x2) = 3*x1 + 5*x2 + 7
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64);

        // Prove with the correct sum
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, _) = sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut prover_transcript);

        // Verify with a WRONG claimed sum
        let wrong_sum = FE::from(99u64);
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let result = sumcheck_verify(&proof, &wrong_sum, 2, &mut verifier_transcript);

        assert!(result.is_err(), "Verification should fail with wrong sum");
        match result.unwrap_err() {
            SumcheckError::RoundSumMismatch { round } => {
                assert_eq!(round, 0, "Mismatch should be detected at round 0");
            }
            other => panic!("Expected RoundSumMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_sumcheck_verify_roundtrip_3var() {
        // f(x1, x2, x3) = x1 * x2 + x3 + 2
        // Hypercube evaluations in little-endian bit order:
        //   index = b0 + 2*b1 + 4*b2, where b0=x1, b1=x2, b2=x3
        //   (0,0,0)=2, (1,0,0)=2, (0,1,0)=2, (1,1,0)=3, (0,0,1)=3, (1,0,1)=3, (0,1,1)=3, (1,1,1)=4
        let evals = vec![
            FE::from(2u64), // f(0,0,0) = 0*0 + 0 + 2 = 2
            FE::from(2u64), // f(1,0,0) = 1*0 + 0 + 2 = 2
            FE::from(2u64), // f(0,1,0) = 0*1 + 0 + 2 = 2
            FE::from(3u64), // f(1,1,0) = 1*1 + 0 + 2 = 3
            FE::from(3u64), // f(0,0,1) = 0*0 + 1 + 2 = 3
            FE::from(3u64), // f(1,0,1) = 1*0 + 1 + 2 = 3
            FE::from(3u64), // f(0,1,1) = 0*1 + 1 + 2 = 3
            FE::from(4u64), // f(1,1,1) = 1*1 + 1 + 2 = 4
        ];
        let claimed_sum: FE = evals.iter().fold(FE::zero(), |acc, v| acc + v); // = 22

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAB]);
        let (proof, prover_challenges) =
            sumcheck_prove(&evals, &claimed_sum, 3, 1, &mut prover_transcript);

        // Verify
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[0xAB]);
        let result = sumcheck_verify(&proof, &claimed_sum, 3, &mut verifier_transcript);

        assert!(result.is_ok(), "3-var verification should succeed");
        let (challenges, _final_eval) = result.unwrap();
        assert_eq!(challenges.len(), 3);
        assert_eq!(challenges, prover_challenges);
    }

    #[test]
    fn test_sumcheck_verify_final_eval() {
        // Verify that final_eval matches the MLE evaluated at the challenge point.
        // f(x1, x2) = 3*x1 + 5*x2 + 7
        let evals = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(12u64),
            FE::from(15u64),
        ];
        let claimed_sum = FE::from(44u64);

        // Prove
        let mut prover_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (proof, _) = sumcheck_prove(&evals, &claimed_sum, 2, 1, &mut prover_transcript);

        // Verify
        let mut verifier_transcript = DefaultTranscript::<GoldilocksField>::new(&[]);
        let (challenges, final_eval) =
            sumcheck_verify(&proof, &claimed_sum, 2, &mut verifier_transcript).unwrap();

        // Compute MLE at the challenge point manually:
        // f(r1, r2) = f(0,0)*(1-r1)*(1-r2) + f(1,0)*r1*(1-r2)
        //           + f(0,1)*(1-r1)*r2     + f(1,1)*r1*r2
        let r1 = &challenges[0];
        let r2 = &challenges[1];
        let one = FE::one();
        let one_minus_r1 = one - r1;
        let one_minus_r2 = one - r2;
        let mle_at_r = ((evals[0] * one_minus_r1) * one_minus_r2)
            + ((evals[1] * r1) * one_minus_r2)
            + ((evals[2] * one_minus_r1) * r2)
            + ((evals[3] * r1) * r2);

        assert_eq!(
            final_eval, mle_at_r,
            "final_eval from verifier must match MLE evaluated at challenge point"
        );
    }
}
