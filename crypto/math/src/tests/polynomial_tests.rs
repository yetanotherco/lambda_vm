#[cfg(test)]
mod tests {
    use crate::field::element::FieldElement;
    use crate::field::goldilocks::GoldilocksField;
    use crate::field::traits::{IsField, IsPrimeField};
    use crate::polynomial::{Polynomial, pad_with_zero_coefficients};
    use alloc::string::{String, ToString};
    use alloc::{format, vec::Vec};

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

    // ==================== End of test helper functions ====================

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
        let interpolating = numerator.mul_with_ref(&denominator);
        assert_eq!(
            (FE::new(2) - FE::new(4)) * (FE::new(1) * (FE::new(2) - FE::new(4)).inv().unwrap()),
            FE::new(1)
        );
        assert_eq!(interpolating.evaluate(&FE::new(2)), FE::new(1));
        assert_eq!(interpolating.evaluate(&FE::new(4)), FE::new(0));
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

#[cfg(test)]
mod row_major_lde_tests {
    use crate::fft::bowers_fft::LayerTwiddles;
    use crate::fft::two_half_fft::TwoHalfTwiddles;
    use crate::field::element::FieldElement;
    use crate::field::goldilocks::GoldilocksField;
    use crate::polynomial::Polynomial;
    use alloc::vec::Vec;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    /// Differential test: `coset_lde_full_expand_row_major` on a row-major
    /// buffer holding M columns must produce the same per-cell output as
    /// running `coset_lde_full_expand` on each of those M columns
    /// independently, then transposing the M LDE columns back into row order.
    /// Covers a range of (log_n, M, blowup) to catch off-by-one bugs in the
    /// M-block bit-reverse and in the row scaling step.
    #[test]
    fn coset_lde_full_expand_row_major_matches_single_column_per_column() {
        for log_n in 2..=8 {
            let n = 1usize << log_n;
            for &blowup_factor in &[2usize, 4] {
                let lde_size = n * blowup_factor;
                let inv_tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
                let fwd_tw = LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64).unwrap();
                let two_inv = TwoHalfTwiddles::<F>::new(log_n, true).unwrap();
                let two_fwd =
                    TwoHalfTwiddles::<F>::new(lde_size.trailing_zeros() as usize, false).unwrap();

                let offset = FE::from(3u64);
                let n_inv = FE::from(n as u64).inv().unwrap();
                let mut weights = Vec::with_capacity(n);
                let mut offset_power = n_inv;
                for _ in 0..n {
                    weights.push(offset_power);
                    offset_power = &offset_power * &offset;
                }

                for &m in &[1usize, 2, 3, 5, 8] {
                    let cols: Vec<Vec<FE>> = (0..m)
                        .map(|c| {
                            (0..n)
                                .map(|i| {
                                    FE::from((c as u64).wrapping_mul(1_000_003) + i as u64 + 17)
                                })
                                .collect()
                        })
                        .collect();

                    // Reference: single-column coset_lde_full_expand on each column.
                    let expected_cols: Vec<Vec<FE>> = cols
                        .iter()
                        .map(|c| {
                            let mut buf = c.clone();
                            Polynomial::<FE>::coset_lde_full_expand::<F>(
                                &mut buf,
                                blowup_factor,
                                &weights,
                                &inv_tw,
                                &fwd_tw,
                            )
                            .unwrap();
                            buf
                        })
                        .collect();

                    // Subject under test: row-major batched pipeline.
                    let mut row_major: Vec<FE> = Vec::with_capacity(n * m);
                    #[allow(clippy::needless_range_loop)]
                    for r in 0..n {
                        for c in 0..m {
                            row_major.push(cols[c][r]);
                        }
                    }
                    Polynomial::<FE>::coset_lde_full_expand_row_major::<F>(
                        &mut row_major,
                        m,
                        blowup_factor,
                        &weights,
                        &two_inv,
                        &two_fwd,
                    )
                    .unwrap();
                    assert_eq!(row_major.len(), lde_size * m);

                    for r in 0..lde_size {
                        for c in 0..m {
                            assert_eq!(
                                row_major[r * m + c],
                                expected_cols[c][r],
                                "log_n={log_n} blowup={blowup_factor} m={m} r={r} c={c}",
                            );
                        }
                    }
                }
            }
        }
    }
}
