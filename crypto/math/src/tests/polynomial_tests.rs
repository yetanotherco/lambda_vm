#[cfg(test)]
mod tests {
    use crate::field::element::FieldElement;
    use crate::field::goldilocks::GoldilocksField;
    use crate::field::traits::{IsField, IsPrimeField};
    use crate::polynomial::{Polynomial, pad_with_zero_coefficients};
    use alloc::string::{String, ToString};
    use alloc::{format, vec, vec::Vec};

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    // ==================== Test helper functions (moved from polynomial/mod.rs) ====================

    /// Evaluates a polynomial at a slice of points
    fn evaluate_slice<F: IsField>(
        poly: &Polynomial<FieldElement<F>>,
        input: &[FieldElement<F>],
    ) -> Vec<FieldElement<F>> {
        input.iter().map(|x| poly.evaluate(x)).collect()
    }

    /// Returns the derivative of the polynomial with respect to x
    fn differentiate<F: IsField>(
        poly: &Polynomial<FieldElement<F>>,
    ) -> Polynomial<FieldElement<F>> {
        let degree = poly.degree();
        if degree == 0 {
            return Polynomial::zero();
        }
        let mut derivative = Vec::with_capacity(degree);
        for (i, coeff) in poly.coefficients().iter().enumerate().skip(1) {
            derivative.push(FieldElement::<F>::from(i as u64) * coeff);
        }
        Polynomial::new(&derivative)
    }

    /// Print the polynomial as a string ready to be used in SageMath
    fn print_as_sage_poly<F: IsPrimeField>(
        poly: &Polynomial<FieldElement<F>>,
        var_name: Option<char>,
    ) -> String {
        let var_name = var_name.unwrap_or('x');
        if poly.coefficients().is_empty()
            || poly.coefficients().len() == 1 && poly.coefficients()[0] == FieldElement::zero()
        {
            return String::new();
        }

        let mut string = String::new();
        let zero = FieldElement::<F>::zero();

        for (i, coeff) in poly.coefficients().iter().rev().enumerate() {
            if *coeff == zero {
                continue;
            }

            let coeff_str = coeff.canonical().to_string();

            if i == poly.coefficients().len() - 1 {
                string.push_str(&coeff_str);
            } else if i == poly.coefficients().len() - 2 {
                string.push_str(&format!("{coeff_str}*{var_name} + "));
            } else {
                string.push_str(&format!(
                    "{}*{}^{} + ",
                    coeff_str,
                    var_name,
                    poly.coefficients().len() - 1 - i
                ));
            }
        }

        string
    }

    /// Computes the composition of polynomials P1(t) and P2(t), that is P1(P2(t))
    fn compose<F: IsField>(
        poly_1: &Polynomial<FieldElement<F>>,
        poly_2: &Polynomial<FieldElement<F>>,
    ) -> Polynomial<FieldElement<F>> {
        let max_degree: u64 = (poly_1.degree() * poly_2.degree()) as u64;

        let mut interpolation_points = vec![];
        for i in 0_u64..max_degree + 1 {
            interpolation_points.push(FieldElement::<F>::from(i));
        }

        let values: Vec<_> = interpolation_points
            .iter()
            .map(|value| {
                let intermediate_value = poly_2.evaluate(value);
                poly_1.evaluate(&intermediate_value)
            })
            .collect();

        Polynomial::interpolate(interpolation_points.as_slice(), values.as_slice())
            .expect("xs and ys have equal length and xs are unique")
    }

    // ==================== End of test helper functions ====================

    fn polynomial_a() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)])
    }

    fn polynomial_minus_a() -> Polynomial<FE> {
        Polynomial::new(&[-FE::new(1), -FE::new(2), -FE::new(3)])
    }

    fn polynomial_b() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5)])
    }

    fn polynomial_a_plus_b() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(4), FE::new(6), FE::new(8)])
    }

    fn polynomial_b_minus_a() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(2), FE::new(2), FE::new(2)])
    }

    #[test]
    fn adding_a_and_b_equals_a_plus_b() {
        assert_eq!(polynomial_a() + polynomial_b(), polynomial_a_plus_b());
    }

    #[test]
    fn adding_a_and_a_plus_b_does_not_equal_b() {
        assert_ne!(polynomial_a() + polynomial_a_plus_b(), polynomial_b());
    }

    #[test]
    fn add_5_to_0_is_5() {
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p1 + p2, Polynomial::new(&[FE::new(5)]));
    }

    #[test]
    fn add_0_to_5_is_5() {
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p2 + p1, Polynomial::new(&[FE::new(5)]));
    }

    #[test]
    fn negating_0_returns_0() {
        let p1 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(-p1, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn negating_a_is_equal_to_minus_a() {
        assert_eq!(-polynomial_a(), polynomial_minus_a());
    }

    #[test]
    fn negating_a_is_not_equal_to_a() {
        assert_ne!(-polynomial_a(), polynomial_a());
    }

    #[test]
    fn substracting_5_5_gives_0() {
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(5)]);
        let p3 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p1 - p2, p3);
    }

    #[test]
    fn substracting_b_and_a_equals_b_minus_a() {
        assert_eq!(polynomial_b() - polynomial_a(), polynomial_b_minus_a());
    }

    #[test]
    fn constructor_removes_zeros_at_the_end_of_polynomial() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(0)]);
        assert_eq!(p1.coefficients, &[FE::new(3), FE::new(4)]);
    }

    #[test]
    fn pad_with_zero_coefficients_returns_polynomials_with_zeros_until_matching_size() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(4)]);
        let p2 = Polynomial::new(&[FE::new(3)]);

        assert_eq!(p2.coefficients, &[FE::new(3)]);
        let (pp1, pp2) = pad_with_zero_coefficients(&p1, &p2);
        assert_eq!(pp1, p1);
        assert_eq!(pp2.coefficients, &[FE::new(3), FE::new(0)]);
    }

    #[test]
    fn multiply_5_and_0_is_0() {
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p1 * p2, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn multiply_0_and_x_is_0() {
        let p1 = Polynomial::new(&[FE::new(0)]);
        let p2 = Polynomial::new(&[FE::new(0), FE::new(1)]);
        assert_eq!(p1 * p2, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn multiply_2_by_3_is_6() {
        let p1 = Polynomial::new(&[FE::new(2)]);
        let p2 = Polynomial::new(&[FE::new(3)]);
        assert_eq!(p1 * p2, Polynomial::new(&[FE::new(6)]));
    }

    #[test]
    fn multiply_2xx_3x_3_times_x_4() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = Polynomial::new(&[FE::new(4), FE::new(1)]);
        assert_eq!(
            p1 * p2,
            Polynomial::new(&[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn multiply_x_4_times_2xx_3x_3() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = Polynomial::new(&[FE::new(4), FE::new(1)]);
        assert_eq!(
            p2 * p1,
            Polynomial::new(&[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn evaluate_constant_polynomial_returns_constant() {
        let three = FE::new(3);
        let p = Polynomial::new(&[three]);
        assert_eq!(p.evaluate(&FE::new(10)), three);
    }

    #[test]
    fn test_evaluate_slice() {
        let three = FE::new(3);
        let p = Polynomial::new(&[three]);
        let ret = evaluate_slice(&p, &[FE::new(10), FE::new(15)]);
        assert_eq!(ret, [three, three]);
    }

    #[test]
    fn create_degree_0_new_monomial() {
        assert_eq!(
            Polynomial::new_monomial(FE::new(3), 0),
            Polynomial::new(&[FE::new(3)])
        );
    }

    #[test]
    fn zero_poly_evals_0_in_3() {
        assert_eq!(
            Polynomial::new_monomial(FE::new(0), 0).evaluate(&FE::new(3)),
            FE::new(0)
        );
    }

    #[test]
    fn evaluate_degree_1_new_monomial() {
        let two = FE::new(2);
        let four = FE::new(4);
        let p = Polynomial::new_monomial(two, 1);
        assert_eq!(p.evaluate(&two), four);
    }

    #[test]
    fn evaluate_degree_2_monomyal() {
        let two = FE::new(2);
        let eight = FE::new(8);
        let p = Polynomial::new_monomial(two, 2);
        assert_eq!(p.evaluate(&two), eight);
    }

    #[test]
    fn evaluate_3_term_polynomial() {
        let p = Polynomial::new(&[FE::new(3), -FE::new(2), FE::new(4)]);
        assert_eq!(p.evaluate(&FE::new(2)), FE::new(15));
    }

    #[test]
    fn simple_interpolating_polynomial_by_hand_works() {
        let denominator = Polynomial::new(&[FE::new(1) * (FE::new(2) - FE::new(4)).inv().unwrap()]);
        let numerator = Polynomial::new(&[-FE::new(4), FE::new(1)]);
        let interpolating = numerator * denominator;
        assert_eq!(
            (FE::new(2) - FE::new(4)) * (FE::new(1) * (FE::new(2) - FE::new(4)).inv().unwrap()),
            FE::new(1)
        );
        assert_eq!(interpolating.evaluate(&FE::new(2)), FE::new(1));
        assert_eq!(interpolating.evaluate(&FE::new(4)), FE::new(0));
    }

    #[test]
    fn interpolate_x_2_y_3() {
        let p = Polynomial::interpolate(&[FE::new(2)], &[FE::new(3)]).unwrap();
        assert_eq!(FE::new(3), p.evaluate(&FE::new(2)));
    }

    #[test]
    fn interpolate_x_0_2_y_3_4() {
        let p =
            Polynomial::interpolate(&[FE::new(0), FE::new(2)], &[FE::new(3), FE::new(4)]).unwrap();
        assert_eq!(FE::new(3), p.evaluate(&FE::new(0)));
        assert_eq!(FE::new(4), p.evaluate(&FE::new(2)));
    }

    #[test]
    fn interpolate_x_2_5_7_y_10_19_43() {
        let p = Polynomial::interpolate(
            &[FE::new(2), FE::new(5), FE::new(7)],
            &[FE::new(10), FE::new(19), FE::new(43)],
        )
        .unwrap();

        assert_eq!(FE::new(10), p.evaluate(&FE::new(2)));
        assert_eq!(FE::new(19), p.evaluate(&FE::new(5)));
        assert_eq!(FE::new(43), p.evaluate(&FE::new(7)));
    }

    #[test]
    fn interpolate_x_0_0_y_1_1() {
        let p =
            Polynomial::interpolate(&[FE::new(0), FE::new(1)], &[FE::new(0), FE::new(1)]).unwrap();

        assert_eq!(FE::new(0), p.evaluate(&FE::new(0)));
        assert_eq!(FE::new(1), p.evaluate(&FE::new(1)));
    }

    #[test]
    fn interpolate_x_0_y_0() {
        let p = Polynomial::interpolate(&[FE::new(0)], &[FE::new(0)]).unwrap();
        assert_eq!(FE::new(0), p.evaluate(&FE::new(0)));
    }

    #[test]
    fn composition_works() {
        let p = Polynomial::new(&[FE::new(0), FE::new(2)]);
        let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(1)]);
        assert_eq!(
            compose(&p, &q),
            Polynomial::new(&[FE::new(0), FE::new(0), FE::new(2)])
        );
    }

    #[test]
    fn break_in_parts() {
        // p = 3 X^3 + X^2 + 2X + 1
        let p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(1), FE::new(3)]);
        let p0_expected = Polynomial::new(&[FE::new(1), FE::new(1)]);
        let p1_expected = Polynomial::new(&[FE::new(2), FE::new(3)]);
        let parts = p.break_in_parts(2);
        assert_eq!(parts.len(), 2);
        let p0 = &parts[0];
        let p1 = &parts[1];
        assert_eq!(p0, &p0_expected);
        assert_eq!(p1, &p1_expected);
    }

    #[test]
    fn test_differentiate() {
        // 3x^2 + 2x + 42
        let px = Polynomial::new(&[FE::new(42), FE::new(2), FE::new(3)]);
        // 6x + 2
        let dpdx = differentiate(&px);
        assert_eq!(dpdx, Polynomial::new(&[FE::new(2), FE::new(6)]));

        // 128
        let px = Polynomial::new(&[FE::new(128)]);
        // 0
        let dpdx = differentiate(&px);
        assert_eq!(dpdx, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn test_print_as_sage_poly() {
        let p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        assert_eq!(print_as_sage_poly(&p, None), "3*x^2 + 2*x + 1");
    }
}
