#[cfg(test)]
mod tests {
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::field::traits::{IsField, IsPrimeField, IsSubFieldOf};
    use crate::polynomial::{Polynomial, pad_with_zero_coefficients};
    use alloc::string::{String, ToString};
    use alloc::{format, vec, vec::Vec};

    // Some of these tests work when the finite field has order greater than 2.
    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
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

    /// Computes the quotient of the division of P(x) with x - b using Ruffini's rule
    fn ruffini_division<F, L>(
        poly: &Polynomial<FieldElement<F>>,
        b: &FieldElement<L>,
    ) -> Polynomial<FieldElement<L>>
    where
        L: IsField,
        F: IsSubFieldOf<L>,
    {
        if let Some(c) = poly.coefficients().last() {
            let mut c = c.clone().to_extension();
            let mut coefficients = Vec::with_capacity(poly.degree());
            for coeff in poly.coefficients().iter().rev().skip(1) {
                coefficients.push(c.clone());
                c = coeff + c * b;
            }
            coefficients = coefficients.into_iter().rev().collect();
            Polynomial::new(&coefficients)
        } else {
            Polynomial::zero()
        }
    }

    /// Computes quotient only (discards remainder)
    fn div_with_ref<F: IsField>(
        poly: Polynomial<FieldElement<F>>,
        dividend: &Polynomial<FieldElement<F>>,
    ) -> Polynomial<FieldElement<F>> {
        let (quotient, _remainder) = poly.long_division_with_remainder(dividend);
        quotient
    }

    /// Extended Euclidean Algorithm for polynomials
    fn xgcd<F: IsField>(
        poly: &Polynomial<FieldElement<F>>,
        y: &Polynomial<FieldElement<F>>,
    ) -> (
        Polynomial<FieldElement<F>>,
        Polynomial<FieldElement<F>>,
        Polynomial<FieldElement<F>>,
    ) {
        let one = Polynomial::new(&[FieldElement::one()]);
        let zero = Polynomial::zero();
        let (mut old_r, mut r) = (poly.clone(), y.clone());
        let (mut old_s, mut s) = (one.clone(), zero.clone());
        let (mut old_t, mut t) = (zero.clone(), one.clone());

        while r != Polynomial::zero() {
            let quotient = div_with_ref(old_r.clone(), &r);
            old_r = old_r - &quotient * &r;
            core::mem::swap(&mut old_r, &mut r);
            old_s = old_s - &quotient * &s;
            core::mem::swap(&mut old_s, &mut s);
            old_t = old_t - &quotient * &t;
            core::mem::swap(&mut old_t, &mut t);
        }

        let lcinv = old_r.leading_coefficient().inv().unwrap();
        (
            old_s.scale_coeffs(&lcinv),
            old_t.scale_coeffs(&lcinv),
            old_r.scale_coeffs(&lcinv),
        )
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

            let coeff_str = coeff.representative().to_string();

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
        Polynomial::new(&[FE::new(ORDER - 1), FE::new(ORDER - 2), FE::new(ORDER - 3)])
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
    fn division_works() {
        let p1 = Polynomial::new(&[FE::new(1), FE::new(3)]);
        let p2 = Polynomial::new(&[FE::new(1), FE::new(3)]);
        let p3 = p1.mul_with_ref(&p2);
        assert_eq!(div_with_ref(p3, &p2), p1);
    }

    #[test]
    fn division_by_zero_degree_polynomial_works() {
        let four = FE::new(4);
        let two = FE::new(2);
        let p1 = Polynomial::new(&[four, four]);
        let p2 = Polynomial::new(&[two]);
        assert_eq!(Polynomial::new(&[two, two]), div_with_ref(p1, &p2));
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

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn ruffini_inplace_equals_division(p in any::<Vec<u64>>(), b in any::<u64>()) {
            let p: Vec<_> = p.into_iter().map(FE::from).collect();
            let mut p = Polynomial::new(&p);
            let b = FE::from(b);

            let p_ref = p.clone();
            let m = Polynomial::new_monomial(FE::one(), 1) - b;

            p.ruffini_division_inplace(&b);
            prop_assert_eq!(p, div_with_ref(p_ref, &m));
        }
    }

    proptest! {
        #[test]
        fn ruffini_inplace_equals_ruffini(p in any::<Vec<u64>>(), b in any::<u64>()) {
            let p: Vec<_> = p.into_iter().map(FE::from).collect();
            let mut p = Polynomial::new(&p);
            let b = FE::from(b);
            let q = ruffini_division(&p, &b);
            p.ruffini_division_inplace(&b);
            prop_assert_eq!(q, p);
        }
    }
    #[test]
    fn test_xgcd() {
        // Case 1: Simple polynomials
        let p1 = Polynomial::new(&[FE::new(1), FE::new(0), FE::new(1)]); // x^2 + 1
        let p2 = Polynomial::new(&[FE::new(1), FE::new(1)]); // x + 1
        let (a, b, g) = xgcd(&p1, &p2);
        // Check that a * p1 + b * p2 = g
        let lhs = a.mul_with_ref(&p1) + b.mul_with_ref(&p2);
        assert_eq!(a, Polynomial::new(&[FE::new(12)]));
        assert_eq!(b, Polynomial::new(&[FE::new(12), FE::new(11)]));
        assert_eq!(lhs, g);
        assert_eq!(g, Polynomial::new(&[FE::new(1)]));

        // x^2-1 :
        let p3 = Polynomial::new(&[FE::new(ORDER - 1), FE::new(0), FE::new(1)]);
        // x^3-x = x(x^2-1)
        let p4 = Polynomial::new(&[FE::new(0), FE::new(ORDER - 1), FE::new(0), FE::new(1)]);
        let (a, b, g) = xgcd(&p3, &p4);

        let lhs = a.mul_with_ref(&p3) + b.mul_with_ref(&p4);
        assert_eq!(a, Polynomial::new(&[FE::new(1)]));
        assert_eq!(b, Polynomial::zero());
        assert_eq!(lhs, g);
        assert_eq!(g, p3);
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
    fn test_reverse() {
        let p = Polynomial::new(&[FE::new(3), FE::new(2), FE::new(1)]);
        assert_eq!(
            p.reverse(3),
            Polynomial::new(&[FE::new(0), FE::new(1), FE::new(2), FE::new(3)])
        );
    }

    #[test]
    fn test_truncate() {
        let p = Polynomial::new(&[FE::new(3), FE::new(2), FE::new(1)]);
        assert_eq!(p.truncate(2), Polynomial::new(&[FE::new(3), FE::new(2)]));
    }

    #[test]
    fn test_print_as_sage_poly() {
        let p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        assert_eq!(print_as_sage_poly(&p, None), "3*x^2 + 2*x + 1");
    }
}

#[cfg(test)]
mod dense_multilinear_poly_tests {
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    pub fn evals(r: Vec<FE>) -> Vec<FE> {
        let mut evals: Vec<FE> = vec![FE::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[i / 2];
                evals[i] = scalar * j;
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    pub fn compute_factored_evals(r: Vec<FE>) -> (Vec<FE>, Vec<FE>) {
        let size = r.len();
        let (left_num_vars, _right_num_vars) = (size / 2, size - size / 2);
        let l = evals(r[..left_num_vars].to_vec());
        let r = evals(r[left_num_vars..size].to_vec());
        (l, r)
    }

    fn evaluate_with_lr(z: &[FE], r: &[FE]) -> FE {
        let (l, r) = compute_factored_evals(r.to_vec());
        let size = r.len();
        // Ensure size is even.
        assert!(size % 2 == 0);
        // n = 2^size
        let n = (2usize).pow(size as u32);
        // Compute m = sqrt(n) = 2^(l/2)
        let m = (n as f64).sqrt() as usize;
        // Compute vector-matrix product between L and Z (viewed as a matrix)
        let lz = (0..m)
            .map(|i| {
                (0..m).fold(FE::zero(), |mut acc, j| {
                    acc += l[j] * z[j * m + i];
                    acc
                })
            })
            .collect::<Vec<FE>>();
        // Compute dot product between LZ and R
        (0..lz.len()).map(|i| lz[i] * r[i]).sum()
    }

    #[test]
    fn evaluation() {
        // Example: Z = [1, 2, 1, 4]
        let z = vec![FE::one(), FE::from(2u64), FE::one(), FE::from(4u64)];
        // r = [4, 3]
        let r = vec![FE::from(4u64), FE::from(3u64)];
        let eval_with_lr = evaluate_with_lr(&z, &r);
        let poly = DenseMultilinearPolynomial::new(z);
        let eval = poly.evaluate(r).unwrap();
        assert_eq!(eval, FE::from(28u64));
        assert_eq!(eval_with_lr, eval);
    }

    #[test]
    fn evaluate_with() {
        let two = FE::from(2);
        let z = vec![
            FE::zero(),
            FE::zero(),
            FE::zero(),
            FE::one(),
            FE::one(),
            FE::one(),
            FE::zero(),
            two,
        ];
        let x = vec![FE::one(), FE::one(), FE::one()];
        let y = DenseMultilinearPolynomial::<F>::evaluate_with(z.as_slice(), x.as_slice()).unwrap();
        assert_eq!(y, two);
    }

    #[test]
    fn add() {
        use core::ops::Add;
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(7); 4]);
        let c = a.add(b).unwrap();
        assert_eq!(*c.evals(), vec![FE::from(10); 4]);
    }

    #[test]
    fn mul() {
        use core::ops::Mul;
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = a.mul(&FE::from(2));
        assert_eq!(*b.evals(), vec![FE::from(6); 4]);
    }

    // Take a multilinear polynomial of length 2^2 and merge with a polynomial of 2^1.
    // The resulting polynomial should be padded to length 2^3 = 8 and the last two evaluations should be FE::zero().
    #[test]
    fn merge() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 2]);
        let c = DenseMultilinearPolynomial::merge(&[a, b]);
        assert_eq!(c.len(), 8);
        assert_eq!(c[c.len() - 1], FE::zero());
        assert_eq!(c[c.len() - 2], FE::zero());
    }

    #[test]
    fn extend() {
        let mut a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        a.extend(&b);
        assert_eq!(a.len(), 8);
        assert_eq!(a.num_vars(), 3);
    }

    #[test]
    #[should_panic]
    fn extend_unequal() {
        let mut a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 2]);
        a.extend(&b);
    }
}

#[cfg(test)]
mod sparse_multilinear_poly_tests {
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::polynomial::sparse_multilinear_poly::SparseMultilinearPolynomial;
    use alloc::vec;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    #[test]
    fn evaluate() {
        // Let the polynomial have 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
        // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

        let two = FE::from(2);
        let z = vec![(3, FE::one()), (5, FE::one()), (7, two)];
        let m_poly = SparseMultilinearPolynomial::<F>::new(3, z.clone());

        let x = vec![FE::one(), FE::one(), FE::one()];
        assert_eq!(m_poly.evaluate(x.as_slice()).unwrap(), two);
        assert_eq!(
            SparseMultilinearPolynomial::evaluate_with(3, &z, x.as_slice()).unwrap(),
            two
        );

        let x = vec![FE::one(), FE::zero(), FE::one()];
        assert_eq!(m_poly.evaluate(x.as_slice()).unwrap(), FE::one());
        assert_eq!(
            SparseMultilinearPolynomial::evaluate_with(3, &z, x.as_slice()).unwrap(),
            FE::one()
        );
    }
}
