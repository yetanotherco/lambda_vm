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
pub struct RoundPoly<E: IsField> {
    /// Evaluations at x = 0, 1, ..., d where d = evals.len() - 1.
    evals: Vec<FieldElement<E>>,
}

impl<E: IsField> RoundPoly<E> {
    /// Create a new `RoundPoly` from evaluations at x = 0, 1, ..., d.
    ///
    /// `evals[i]` is the polynomial evaluated at x = i.
    /// The polynomial has degree `evals.len() - 1`.
    pub fn new(evals: Vec<FieldElement<E>>) -> Self {
        assert!(!evals.is_empty(), "RoundPoly must have at least one evaluation");
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

        // Compute the "master" numerator product: N(x) = prod_{j=0}^{d} (x - j)
        let master_product = point_minus_j
            .iter()
            .fold(FieldElement::one(), |acc, v| &acc * v);

        // For each node i, the Lagrange basis polynomial value is:
        //   L_i(x) = N(x) / (x - i) / prod_{j != i}(i - j)
        //
        // The denominator prod_{j != i}(i - j) for integer nodes is:
        //   prod_{j=0, j!=i}^{d} (i - j) = i! * (-1)^(d-i) * (d-i)!
        //   which simplifies to: (-1)^(d-i) * i! * (d-i)!
        //
        // We compute this directly as a field element.
        let mut result = FieldElement::zero();

        for i in 0..=d {
            // Compute the barycentric weight: 1 / prod_{j != i}(i - j)
            let mut denom = FieldElement::one();
            for j in 0..=d {
                if j != i {
                    // (i - j) as a signed integer, converted to field element
                    let diff = (i as i64) - (j as i64);
                    denom = &denom * &FieldElement::from(diff);
                }
            }
            let denom_inv = denom.inv().expect("Lagrange denominator must be nonzero");

            // L_i(point) = master_product / (point - i) * (1 / denom)
            // We already have (point - i) in point_minus_j[i]
            let point_minus_i_inv = point_minus_j[i]
                .inv()
                .expect("Already checked point != node");

            let basis_value = &(&master_product * &point_minus_i_inv) * &denom_inv;
            result = &result + &(&self.evals[i] * &basis_value);
        }

        result
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

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

    // Use the base Goldilocks field for tests. The cubic extension would work
    // identically since all operations are generic over IsField.
    type FE = FieldElement<GoldilocksField>;

    /// Helper: evaluate a polynomial given in coefficient form at a point.
    /// coeffs[i] is the coefficient of x^i.
    fn eval_poly_coeffs(coeffs: &[FE], x: &FE) -> FE {
        let mut result = FE::zero();
        let mut power = FE::one();
        for c in coeffs {
            result = &result + &(c * &power);
            power = &power * x;
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
        let coeffs = [FE::from(4u64), FE::from(3u64), FE::from(2u64), FE::from(1u64)];
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
            assert_eq!(
                poly.evaluate(&point),
                expected,
                "Mismatch at x = {}",
                x
            );
        }
    }
}
