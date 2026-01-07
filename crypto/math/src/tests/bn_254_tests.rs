#[cfg(test)]
mod field_extension_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::{
        BN254_PRIME_FIELD_ORDER, BN254PrimeField, Degree2ExtensionField, Degree4ExtensionField,
        Degree6ExtensionField, Degree12ExtensionField,
    };
    use crate::field::element::FieldElement;

    type FpE = FieldElement<BN254PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;
    type Fp4E = FieldElement<Degree4ExtensionField>;
    type Fp6E = FieldElement<Degree6ExtensionField>;
    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn embed_base_field_with_degree_2_extension() {
        let a = FpE::from(3);
        let a_extension = Fp2E::from(3);
        assert_eq!(a.to_extension::<Degree2ExtensionField>(), a_extension);
    }

    #[test]
    fn add_base_field_with_degree_2_extension() {
        let a = FpE::from(3);
        let a_extension = Fp2E::from(3);
        let b = Fp2E::from(2);
        assert_eq!(a + &b, a_extension + b);
    }

    #[test]
    fn mul_degree_2_with_degree_6_extension() {
        let a = Fp2E::new([FpE::from(3), FpE::from(4)]);
        let a_extension = a.clone().to_extension::<Degree2ExtensionField>();
        let b = Fp6E::from(2);
        assert_eq!(a * &b, a_extension * b);
    }

    #[test]
    fn mul_degree_2_with_degree_4_extension() {
        let a = Fp2E::new([FpE::from(3), FpE::from(4)]);
        let a_extension = a.clone().to_extension::<Degree4ExtensionField>();
        let b = Fp4E::from(2);
        assert_eq!(a * &b, a_extension * b);
    }

    #[test]
    fn div_degree_6_degree_12_extension() {
        let a = Fp6E::from(3);
        let a_extension = Fp12E::from(3);
        let b = Fp12E::from(2);
        assert_eq!((a / &b).unwrap(), (a_extension / b).unwrap());
    }

    #[test]
    fn double_equals_sum_two_times() {
        let a = FpE::from(3);
        assert_eq!(a.double(), a.clone() + a);
    }

    #[test]
    fn base_field_sum_is_asociative() {
        let a = FpE::from(3);
        let b = FpE::from(2);
        let c = &a + &b;
        assert_eq!(a.double() + b, a + c);
    }

    #[test]
    fn degree_2_extension_mul_is_conmutative() {
        let a = Fp2E::from(3);
        let b = Fp2E::new([FpE::from(2), FpE::from(4)]);
        assert_eq!(&a * &b, b * a);
    }

    #[test]
    fn base_field_pow_p_is_identity() {
        let a = FpE::from(3);
        assert_eq!(a.pow(BN254_PRIME_FIELD_ORDER), a);
    }
}

#[cfg(test)]
mod curve_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::Degree2ExtensionField;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::pairing::X;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    #[allow(clippy::upper_case_acronyms)]
    type FpE = FieldElement<BN254PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;

    fn point() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FpE::from_hex_unchecked(
            "27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb",
        );
        let y = FpE::from_hex_unchecked(
            "2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d",
        );
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_times_5() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FpE::from_hex_unchecked(
            "16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3",
        );
        let y = FpE::from_hex_unchecked(
            "2512347279ba1049ef97d4ec348d838f939d2b7623e88f4826643cf3889599b2",
        );
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_works() {
        let point = point();
        let point_times_5 = point_times_5();
        assert_eq!(point.operate_with_self(5_u16), point_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point();
        assert_eq!(
            *p.x(),
            FpE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FpE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );
        assert_eq!(*p.z(), FpE::one());
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let p = point();
        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();
        assert_eq!(p.operate_with(&neutral_element), p);
    }

    #[test]
    fn neutral_element_plus_neutral_element_is_neutral_element() {
        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();
        assert_eq!(
            neutral_element.operate_with(&neutral_element),
            neutral_element
        );
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BN254Curve::create_point_from_affine(FpE::from(0), FpE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = BN254Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = BN254Curve::generator();
        let g2 = g.operate_with_self(2_u64);

        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        let three = FpE::from(3);
        let y_sq_0 = x.pow(3_u16) + three;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BN254Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn operate_with_self_works_2() {
        let g = BN254TwistCurve::generator();
        assert_eq!(
            (g.operate_with_self(X)).double(),
            (g.operate_with_self(2 * X))
        )
    }

    #[test]
    fn operate_with_self_works_3() {
        let g = BN254TwistCurve::generator();
        assert_eq!(
            (g.operate_with_self(X)).operate_with(&g),
            (g.operate_with_self(X + 1))
        )
    }

    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BN254TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn other_g2_point_is_in_subgroup() {
        let g = BN254TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn invalid_g2_is_not_in_subgroup() {
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ])
        .unwrap();
        assert!(!q.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_two_times_is_identity() {
        let a = Fp2E::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();
        assert_eq!(a, expected);
    }
}

#[cfg(test)]
mod compression_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::{
        BN254PrimeField, Degree2ExtensionField,
    };
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::elliptic_curve::short_weierstrass::traits::Compress;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use crate::field::element::FieldElement;

    type FpE = FieldElement<BN254PrimeField>;
    #[allow(dead_code)]
    type Fp2E = FieldElement<Degree2ExtensionField>;
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;

    #[cfg(feature = "alloc")]
    use crate::{
        cyclic_group::IsGroup, traits::ByteConversion, unsigned_integer::element::UnsignedInteger,
    };

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_generator() {
        let g = BN254Curve::generator();
        let mut compressed_g = BN254Curve::compress_g1_point(&g);
        let first_byte = compressed_g.first().unwrap();

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        compressed_g[0] = first_byte_without_control_bits;

        let compressed_g_x = FpE::from_bytes_be(&compressed_g).unwrap();
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_point_at_inf() {
        let inf = G1Point::neutral_element();
        let compressed_inf = BN254Curve::compress_g1_point(&inf);
        let first_byte = compressed_inf.first().unwrap();

        assert_eq!(*first_byte >> 6, 1_u8);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g1_compress_decompress_is_identity() {
        let g = BN254Curve::generator();
        let mut compressed_g_slice: [u8; 32] = BN254Curve::compress_g1_point(&g);
        let decompressed_g = BN254Curve::decompress_g1_point(&mut compressed_g_slice).unwrap();
        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g1_compress_decompress_is_identity_2() {
        let g = BN254Curve::generator().operate_with_self(UnsignedInteger::<4>::from("2"));
        let mut compressed_g_slice: [u8; 32] = BN254Curve::compress_g1_point(&g);
        let decompressed_g = BN254Curve::decompress_g1_point(&mut compressed_g_slice).unwrap();
        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g2_compress_generator() {
        let g = BN254TwistCurve::generator();
        let mut compressed_g = BN254Curve::compress_g2_point(&g);
        let first_byte = compressed_g.first().unwrap();

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        compressed_g[0] = first_byte_without_control_bits;

        let [x1, x0] = FieldElement::<Degree2ExtensionField>::from_bytes_be(&compressed_g)
            .unwrap()
            .value()
            .clone();
        let compressed_g_x = FieldElement::<Degree2ExtensionField>::new([x0, x1]);
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g2_compress_point_at_inf() {
        let inf = G2Point::neutral_element();
        let compressed_inf = BN254Curve::compress_g2_point(&inf);
        let first_byte = compressed_inf.first().unwrap();

        assert_eq!(*first_byte >> 6, 1_u8);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g2_compress_decompress_is_identity() {
        let g = BN254TwistCurve::generator();
        let mut compressed_g_slice: [u8; 64] = BN254Curve::compress_g2_point(&g);
        let decompressed_g = BN254Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();
        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g2_compress_decompress_is_identity_2() {
        let g = BN254TwistCurve::generator().operate_with_self(UnsignedInteger::<4>::from("2"));
        let mut compressed_g_slice: [u8; 64] = BN254Curve::compress_g2_point(&g);
        let decompressed_g = BN254Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();
        assert_eq!(g, decompressed_g);
    }

    #[test]
    fn g1_decompress_wrong_bytes_length() {
        let mut input_bytes: [u8; 31] = [0; 31];
        let result = BN254Curve::decompress_g1_point(&mut input_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn g2_decompress_wrong_bytes_length() {
        let mut input_bytes: [u8; 65] = [0; 65];
        let result = BN254Curve::decompress_g2_point(&mut input_bytes);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod pairing_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::{
        Degree2ExtensionField, Degree12ExtensionField,
    };
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::pairing::{
        BN254AtePairing, TWO_INV, X, cyclotomic_pow_x, cyclotomic_square, frobenius,
        frobenius_cube, frobenius_square, miller_optimized,
    };
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::elliptic_curve::traits::{FromAffine, IsEllipticCurve, IsPairing};
    use crate::field::element::FieldElement;
    use crate::unsigned_integer::element::U256;
    use crate::{cyclic_group::IsGroup, unsigned_integer::element::U384};

    type FpE = FieldElement<BN254PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;
    type Fp12E = FieldElement<Degree12ExtensionField>;
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;

    #[test]
    fn batch_ate_pairing_bilinearity() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();

        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result_1 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.operate_with_self(a * b), &q.neg()),
        ])
        .unwrap();
        assert_eq!(result_1, Fp12E::one());
    }

    #[test]
    fn ate_pairing_returns_one_when_one_element_is_the_neutral_element() {
        let p1 = BN254Curve::generator();
        let q1 = G2Point::neutral_element();
        let result_1 = BN254AtePairing::compute_batch(&[(&p1, &q1)]).unwrap();
        assert_eq!(result_1, Fp12E::one());

        let p2 = G1Point::neutral_element();
        let q2 = BN254TwistCurve::generator();
        let result_2 = BN254AtePairing::compute_batch(&[(&p2, &q2)]).unwrap();
        assert_eq!(result_2, Fp12E::one());
    }

    #[test]
    fn ate_pairing_errors_when_g2_element_is_not_in_subgroup() {
        let p = BN254Curve::generator();
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ])
        .unwrap();
        let result = BN254AtePairing::compute_batch(&[(&p, &q)]);
        assert!(result.is_err())
    }

    #[test]
    fn apply_12_times_frobenius_is_identity() {
        let f = Fp12E::from_coefficients(&[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ]);
        let mut result = frobenius(&f);
        for _ in 1..12 {
            result = frobenius(&result);
        }
        assert_eq!(f, result)
    }

    #[test]
    fn two_pairs_of_points_match_1() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2)]).unwrap();
        assert_eq!(result, Fp12E::one());
    }

    const R: U256 = U256::from_hex_unchecked(
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
    );

    #[test]
    fn pairing_result_pow_r_is_one() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let pairing_result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
        assert_eq!(pairing_result.pow(R), Fp12E::one());
    }

    #[test]
    fn pairing_is_non_degenerate() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let pairing_result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
        assert_ne!(pairing_result, Fp12E::one());
    }

    #[test]
    fn cyclotomic_square_equals_square() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller_optimized(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius(&frobenius(&f_easy_aux)) * f_easy_aux;
        assert_eq!(cyclotomic_square(&f_easy), f_easy.square());
    }

    #[test]
    fn cyclotomic_pow_x_equals_pow() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller_optimized(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius(&frobenius(&f_easy_aux)) * f_easy_aux;
        assert_eq!(cyclotomic_pow_x(&f_easy), f_easy.pow(X));
    }

    #[test]
    fn apply_6_times_frobenius_square_is_identity() {
        let f = Fp12E::from_coefficients(&[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ]);
        let mut result = frobenius_square(&f);
        for _ in 1..6 {
            result = frobenius_square(&result);
        }
        assert_eq!(f, result)
    }

    #[test]
    fn apply_4_times_frobenius_cube_is_identity() {
        let f = Fp12E::from_coefficients(&[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ]);
        let mut result = frobenius_cube(&f);
        for _ in 1..4 {
            result = frobenius_cube(&result);
        }
        assert_eq!(f, result)
    }

    #[test]
    fn two_pairs_of_points_match_2() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "105456a333e6d636854f987ea7bb713dfd0ae8371a72aea313ae0c32c0bf1016",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0cf031d41b41557f3e7e3ba0c51bebe5da8e6ecd855ec50fc87efcdeac168bcc",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "290158a80cd3d66530f74dc94c94adb88f5cdb481acca997b6e60071f08a115f",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "1a2c3013d2ea92e13c800cde68ef56a294b883f6ac35d25f587c09b1b3c635f7",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "29d1691530ca701b4a106054688728c9972c8512e9789e9567aae23e302ccd75",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "2f997f3dbd66a7afe07fe7862ce239edba9e05c5afff7f8a1259c9733b2dfbb9",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2)]).unwrap();
        assert_eq!(result, Fp12E::one());
    }

    #[test]
    fn two_pairs_of_points_fail() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2)]).unwrap();
        assert!(result != Fp12E::one());
    }

    #[test]
    fn three_pairs_of_points_fail() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "105456a333e6d636854f987ea7bb713dfd0ae8371a72aea313ae0c32c0bf1016",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0cf031d41b41557f3e7e3ba0c51bebe5da8e6ecd855ec50fc87efcdeac168bcc",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "290158a80cd3d66530f74dc94c94adb88f5cdb481acca997b6e60071f08a115f",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "1a2c3013d2ea92e13c800cde68ef56a294b883f6ac35d25f587c09b1b3c635f7",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "0692e55db067300e6e3fe56218fa2f940054e57e7ef92bf7d475a9d8a8502fd2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "00cacf3523caf879d7d05e30549f1e6fdce364cbb8724b0329c6c2a39d4f018e",
                )),
            ]),
        )
        .unwrap();

        let p3 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q3 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2), (&p3, &q3)]).unwrap();
        assert!(result != Fp12E::one());
    }

    #[test]
    fn constant_two_inv_is_two_inverse() {
        assert_eq!(TWO_INV, FpE::from(2).inv().unwrap());
        assert_eq!(TWO_INV * FpE::from(2), FpE::one());
    }
}

#[cfg(test)]
mod twist_tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bn_254::field_extension::{BN254PrimeField, Degree2ExtensionField},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::U256,
    };

    use crate::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
    type Level0FE = FieldElement<BN254PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;

    #[cfg(feature = "alloc")]
    use crate::elliptic_curve::short_weierstrass::point::{
        Endianness, PointFormat, ShortWeierstrassProjectivePoint,
    };

    #[test]
    fn create_generator() {
        let g = BN254TwistCurve::generator();
        let [x, y, _] = g.coordinates();
        assert_eq!(BN254TwistCurve::defining_equation(x, y), Level1FE::zero());
    }

    #[cfg(feature = "std")]
    #[test]
    fn serialize_deserialize_generator() {
        let g = BN254TwistCurve::generator();
        let bytes = g.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let deserialized = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::deserialize(
            &bytes,
            PointFormat::Projective,
            Endianness::LittleEndian,
        )
        .unwrap();

        assert_eq!(deserialized, g);
    }

    #[test]
    fn add_points() {
        let px = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "8ae7459fe0d23419ec54b150574b77b1d0aa0785ce98d43365898a1d9168a2a",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "235ec75b0bbcca3f1bab9f3aa4c65d52ccb479cb398b54bd4b0f7e3a24454b44",
            )),
        ]);
        let py = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "214ebea9c718706be05072da305b74c1585f9e75dbf99c7859bf2292e07c1691",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "2efdafd49b6e2d718b4e3b3d78939a6463f5f84f4343ec6b1161971dd38af12f",
            )),
        ]);
        let qx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "2b87b85159311f97b20b4c0e27eed978cf94984b06203f853c43192de1579324",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "27b8bc83db0109e29df764a1379a0b9ba3dab41db33aa9be4aef88d4dd9cb275",
            )),
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "18e358db9be18771bb6d8ba89a7be0d521782f10af8398e981b1dd252d114bed",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "8aa3f8a241032d3832b0f52403eb4ea852e23ec4c1e6d39b08de5ac36a2d43b",
            )),
        ]);
        let expectedx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "27e1bb6cb3f893ef4af84ff82bd36b0c0832e3c5d4649da024b41bfecdc74233",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "b04a4feada4eba73191184c5f39f98e7319dc888a2b258697511a2035723656",
            )),
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "a5490e3b00bc8e434f9a1ba734b05c27c525889bf117bb4d293f5aa54b238c5",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "136ce0ba382e5d37c3e05eff8365e0e6857eefa150096af33bdbdf327649c0eb",
            )),
        ]);
        let p = BN254TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BN254TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BN254TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }
}

#[cfg(test)]
mod sqrt_tests {
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254FieldElement;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::Degree2ExtensionField;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::sqrt::sqrt_qfe;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
    use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use crate::field::element::FieldElement;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

    #[test]
    /// We took the q1 point of the test two_pairs_of_points_match_1 from pairing.rs
    /// to get the values of x and y.
    fn test_sqrt_qfe() {
        // Coordinate x of q.
        let x = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
            ),
            BN254FieldElement::from_hex_unchecked(
                "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
            ),
        ]);

        let qfe_b = BN254TwistCurve::b();
        // The equation of the twisted curve is y^2 = x^3 + 3 /(9+u)
        let y_square = x.square() * &x + qfe_b;
        let y = sqrt_qfe(&y_square, 0).unwrap();

        // Coordinate y of q.
        let y_expected = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
            ),
            BN254FieldElement::from_hex_unchecked(
                "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
            ),
        ]);

        let value_y = y.value();
        let value_y_expected = y_expected.value();

        assert_eq!(value_y[0].clone(), value_y_expected[0].clone());
        assert_eq!(value_y[1].clone(), value_y_expected[1].clone());
    }

    #[test]
    /// We took the q1 point of the test two_pairs_of_points_match_2 from pairing.rs
    fn test_sqrt_qfe_2() {
        let x = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
            ),
            BN254FieldElement::from_hex_unchecked(
                "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
            ),
        ]);

        let qfe_b = BN254TwistCurve::b();

        let y_square = x.pow(3_u64) + qfe_b;
        let y = sqrt_qfe(&y_square, 0).unwrap();

        let y_expected = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
            ),
            BN254FieldElement::from_hex_unchecked(
                "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
            ),
        ]);

        let value_y = y.value();
        let value_y_expected = y_expected.value();

        assert_eq!(value_y[0].clone(), value_y_expected[0].clone());
        assert_eq!(value_y[1].clone(), value_y_expected[1].clone());
    }

    #[test]
    fn test_sqrt_qfe_3() {
        let g = BN254TwistCurve::generator().to_affine();
        let y = &g.coordinates()[1];
        let y_square = &y.square();
        let y_result = sqrt_qfe(y_square, 0).unwrap();

        assert_eq!(y_result, y.clone());
    }

    #[test]
    fn test_sqrt_qfe_4() {
        let g = BN254TwistCurve::generator()
            .operate_with_self(2_u16)
            .to_affine();
        let y = &g.coordinates()[1];
        let y_square = &y.square();
        let y_result = sqrt_qfe(y_square, 0).unwrap();

        assert_eq!(y_result, y.clone());
    }

    #[test]
    fn test_sqrt_qfe_5() {
        let a = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from(3),
            BN254FieldElement::from(4),
        ]);
        let a_square = a.square();
        let a_result = sqrt_qfe(&a_square, 0).unwrap();

        assert_eq!(a_result, a);
    }

    #[test]
    fn test_sqrt_qfe_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let a_val: u64 = rng.r#gen();
        let b_val: u64 = rng.r#gen();
        let a = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from(a_val),
            BN254FieldElement::from(b_val),
        ]);
        let a_square = a.square();
        let a_result = sqrt_qfe(&a_square, 0).unwrap();

        assert_eq!(a_result, a);
    }
}
