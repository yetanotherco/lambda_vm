#[cfg(test)]
mod field_extension_tests {
    use crate::elliptic_curve::{
        short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve, traits::IsEllipticCurve,
    };

    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::{
        BLS12381PrimeField, Degree2ExtensionField, Degree12ExtensionField,
    };
    use crate::field::element::FieldElement;
    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn element_squared_1() {
        // base = 1 + u + (1 + u)v + (1 + u)v^2 + ((1+u) + (1 + u)v + (1+ u)v^2)w
        let element_ones =
            Fp12E::from_coefficients(&["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]);
        let element_ones_squared = Fp12E::from_coefficients(&[
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa1",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa5",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa9",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa3",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa7",
            "c",
            "0",
            "c",
        ]);
        assert_eq!(element_ones.pow(2_u16), element_ones_squared);
        assert_eq!(element_ones.square(), element_ones_squared);
    }

    #[test]
    fn element_squared_2() {
        let element_sequence =
            Fp12E::from_coefficients(&["1", "2", "5", "6", "9", "a", "3", "4", "7", "8", "b", "c"]);

        let element_sequence_squared = Fp12E::from_coefficients(&[
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa87d",
            "199",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa851",
            "20b",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa955",
            "1cd",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa845",
            "1e8",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8a9",
            "202",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa5d",
            "16c",
        ]);

        assert_eq!(element_sequence.pow(2_u16), element_sequence_squared);
        assert_eq!(element_sequence.square(), element_sequence_squared);
    }

    #[test]
    fn to_fp12_unnormalized_computes_correctly() {
        let g = BLS12381TwistCurve::generator();
        let expectedx = Fp12E::from_coefficients(&[
            "0",
            "0",
            "24aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
            "13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
        ]);
        let expectedy = Fp12E::from_coefficients(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801",
            "606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be",
            "0",
            "0",
        ]);
        let [g_to_fp12_x, g_to_fp12_y] = g.to_fp12_unnormalized();
        assert_eq!(g_to_fp12_x, expectedx);
        assert_eq!(g_to_fp12_y, expectedy);
    }

    #[test]
    fn add_base_field_with_degree_2_extension() {
        let a = FieldElement::<BLS12381PrimeField>::from(3);
        let a_extension = FieldElement::<Degree2ExtensionField>::from(3);
        let b = FieldElement::<Degree2ExtensionField>::from(2);
        assert_eq!(a + &b, a_extension + b);
    }

    #[test]
    fn double_base_field_with_degree_2_extension() {
        let a = FieldElement::<BLS12381PrimeField>::from(3);
        let b = FieldElement::<Degree2ExtensionField>::from(2);
        assert_eq!(a.double(), a.clone() + a);
        assert_eq!(b.double(), b.clone() + b);
    }

    #[test]
    fn mul_base_field_with_degree_2_extension() {
        let a = FieldElement::<BLS12381PrimeField>::from(3);
        let a_extension = FieldElement::<Degree2ExtensionField>::from(3);
        let b = FieldElement::<Degree2ExtensionField>::from(2);
        assert_eq!(a * &b, a_extension * b);
    }

    #[test]
    fn sub_base_field_with_degree_2_extension() {
        let a = FieldElement::<BLS12381PrimeField>::from(3);
        let a_extension = FieldElement::<Degree2ExtensionField>::from(3);
        let b = FieldElement::<Degree2ExtensionField>::from(2);
        assert_eq!(a - &b, a_extension - b);
    }

    #[test]
    fn div_base_field_with_degree_2_extension() {
        let a = FieldElement::<BLS12381PrimeField>::from(3);
        let a_extension = FieldElement::<Degree2ExtensionField>::from(3);
        let b = FieldElement::<Degree2ExtensionField>::from(2);
        assert_eq!((a / &b).unwrap(), (a_extension / b).unwrap());
    }

    #[test]
    fn embed_base_field_with_degree_2_extension() {
        let a = FieldElement::<BLS12381PrimeField>::from(3);
        let a_extension = FieldElement::<Degree2ExtensionField>::from(3);
        assert_eq!(a.to_extension::<Degree2ExtensionField>(), a_extension);
    }
}

#[cfg(test)]
mod pairing_tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::traits::{IsEllipticCurve, IsPairing},
        unsigned_integer::element::U384,
    };

    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::Degree12ExtensionField;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::{
        BLS12381AtePairing, X, add_accumulate_line, cyclotomic_pow_x, cyclotomic_square,
        double_accumulate_line, frobenius, frobenius_square, miller,
    };
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::field::element::FieldElement;

    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn batch_ate_pairing_bilinearity() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result = BLS12381AtePairing::compute_batch(&[
            (
                &p.operate_with_self(a).to_affine(),
                &q.operate_with_self(b).to_affine(),
            ),
            (
                &p.operate_with_self(a * b).to_affine(),
                &q.neg().to_affine(),
            ),
        ])
        .unwrap();
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_returns_one_when_one_element_is_the_neutral_element() {
        let p = BLS12381Curve::generator().to_affine();
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BLS12381AtePairing::compute_batch(&[(&p.to_affine(), &q)]).unwrap();
        assert_eq!(result, FieldElement::one());

        let p = ShortWeierstrassProjectivePoint::neutral_element();
        let q = BLS12381TwistCurve::generator();
        let result = BLS12381AtePairing::compute_batch(&[(&p, &q.to_affine())]).unwrap();
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_errors_when_one_element_is_not_in_subgroup() {
        // p = (0, 2, 1) is in the curve but not in the subgroup.
        // Recall that the BLS 12-381 curve equation is y^2 = x^3 + 4.
        let p = ShortWeierstrassProjectivePoint::new([
            FieldElement::zero(),
            FieldElement::from(2),
            FieldElement::one(),
        ])
        .unwrap();
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BLS12381AtePairing::compute_batch(&[(&p.to_affine(), &q)]);
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
    fn cyclotomic_square_equals_square() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius(&frobenius(&f_easy_aux)) * f_easy_aux;
        assert_eq!(cyclotomic_square(&f_easy), f_easy.square());
    }

    #[test]
    fn cyclotomic_pow_x_equals_pow() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius(&frobenius(&f_easy_aux)) * f_easy_aux;
        assert_eq!(cyclotomic_pow_x(&f_easy), f_easy.pow(X));
    }

    #[test]
    fn test_double_accumulate_line_doubles_point_correctly() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let mut r = g2.clone();
        let mut f = FieldElement::one();
        double_accumulate_line(&mut r, &g1, &mut f);
        assert_eq!(r, g2.operate_with(&g2));
    }

    #[test]
    fn test_double_accumulate_line_doubles_point_correctly_2() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let mut r = g2.clone();
        let mut f = FieldElement::one();
        double_accumulate_line(&mut r, &g1, &mut f);
        let expected_r = g2.operate_with(&g2);
        assert_eq!(r.to_affine(), expected_r.to_affine());
    }

    #[test]
    fn test_add_accumulate_line_adds_points_correctly() {
        let g1 = BLS12381Curve::generator();
        let g = BLS12381TwistCurve::generator();
        let a: u64 = 12;
        let b: u64 = 23;
        let g2 = g.operate_with_self(a).to_affine();
        let g3 = g.operate_with_self(b).to_affine();
        let expected = g.operate_with_self(a + b);
        let mut r = g2;
        let mut f = FieldElement::one();
        add_accumulate_line(&mut r, &g3, &g1, &mut f);
        assert_eq!(r, expected);
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
}

#[cfg(test)]
mod curve_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::{
        BLS12381PrimeField, Degree2ExtensionField,
    };
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U384,
    };

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;
    #[allow(clippy::upper_case_acronyms)]
    type FTE = FieldElement<Degree2ExtensionField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base(
            "32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c",
        );
        let y = FEE::new_base(
            "11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88",
        );
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(
            *p.x(),
            FEE::new_base(
                "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5"
            )
        );
        assert_eq!(
            *p.y(),
            FEE::new_base(
                "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0"
            )
        );
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BLS12381Curve::create_point_from_affine(FEE::from(0), FEE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = BLS12381Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = BLS12381Curve::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of BLS12-381 equation
        let four = FieldElement::from(4);
        let y_sq_0 = x.pow(3_u16) + four;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BLS12381Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_g1_is_in_subgroup() {
        let g = BLS12381Curve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g1_point_is_in_subgroup() {
        let g = BLS12381Curve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g1_point_not_in_subgroup() {
        let x = FEE::new_base(
            "178212cbe4a3026c051d4f867364b3ea84af623f93233b347ffcd3d6b16f16e0a7aedbe1c78d33c6beca76b2b75c8486",
        );
        let y = FEE::new_base(
            "13a8b1347e5b43bc4051754b2a29928b5df78cf03ca3b1f73d0424b09fccdef116c9f0ecbec7420a99b2dd785209e9d",
        );
        let p = BLS12381Curve::create_point_from_affine(x, y).unwrap();
        assert!(!p.is_in_subgroup())
    }

    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BLS12381TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g2_point_is_in_subgroup() {
        let g = BLS12381TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g2_point_not_in_subgroup() {
        let x = FTE::new([
            FEE::new(U384::from_hex_unchecked(
                "97798b4a61ac301bbee71e36b5174e2f4adfe3e1729bdae1fcc9965ae84181be373aa80414823eed694f1270014012d",
            )),
            FEE::new(U384::from_hex_unchecked(
                "c9852cc6e61868966249aec153b50b29b3c22409f4c7880fd13121981c103c8ef84d9ea29b552431360e82cf69219fa",
            )),
        ]);
        let y = FTE::new([
            FEE::new(U384::from_hex_unchecked(
                "16cb3a60f3fa52c8273aceeb94c4c7303e8074aa9eedec7355bbb1e8cceedd4ec1497f573f62822140377b8e339619ed",
            )),
            FEE::new(U384::from_hex_unchecked(
                "1cd919b08afe06bebe9adf6223a55868a6fd8b77efc5c67b60fff39be36e9b44b7f10db16827c83b43ad2dad1947778",
            )),
        ]);

        let p = BLS12381TwistCurve::create_point_from_affine(x, y).unwrap();
        assert!(!p.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_works() {
        let a = FTE::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }

    #[test]
    fn untwist_morphism_has_minimal_poly() {
        use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381TwistCurveFieldElement;
        use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381_PRIME_FIELD_ORDER;
        use crate::unsigned_integer::element::U256;

        // -15132376222941642751 = MILLER_LOOP_CONSTANT + 1 = -d20100000000ffff
        // we want the positive of this coordinate based on x^2 - tx + q
        const TRACE_OF_FROBENIUS: U256 = U256::from_u64(15132376222941642751);

        const ENDO_U_2: BLS12381TwistCurveFieldElement =
            BLS12381TwistCurveFieldElement::const_from_raw([
                FieldElement::from_hex_unchecked(
                    "1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac",
                ),
                FieldElement::from_hex_unchecked("0"),
            ]);

        const ENDO_V_2: BLS12381TwistCurveFieldElement =
            BLS12381TwistCurveFieldElement::const_from_raw([
                FieldElement::from_hex_unchecked(
                    "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa",
                ),
                FieldElement::from_hex_unchecked("0"),
            ]);

        fn psi_square(
            p: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
        ) -> ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
            let [x, y, z] = p.coordinates();
            ShortWeierstrassProjectivePoint::new([x * ENDO_U_2, y * ENDO_V_2, z.clone()]).unwrap()
        }

        // generator
        let p = BLS12381TwistCurve::generator();
        let psi_sq = psi_square(&p);
        let tx = p.psi().operate_with_self(TRACE_OF_FROBENIUS).neg();
        let q = p.operate_with_self(BLS12381_PRIME_FIELD_ORDER);
        // Minimal Polynomial of Untwist Frobenius Endomorphism: X^2 + tX + q, where X = psi(P) -> psi(p)^2 - t * psi(p) + q * p = 0
        let min_poly = psi_sq.operate_with(&tx.neg()).operate_with(&q);
        assert!(min_poly.is_neutral_element())
    }
}

#[cfg(test)]
mod compression_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::{
        BLS12381Curve, BLS12381FieldElement,
    };
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::elliptic_curve::short_weierstrass::traits::Compress;
    use crate::elliptic_curve::traits::{FromAffine, IsEllipticCurve};

    type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;

    #[cfg(feature = "alloc")]
    use crate::{
        cyclic_group::IsGroup, traits::ByteConversion, unsigned_integer::element::UnsignedInteger,
    };

    #[test]
    fn test_zero_point() {
        let g1 = BLS12381Curve::generator();

        assert!(g1.is_in_subgroup());
        let new_x = BLS12381FieldElement::zero();
        let new_y = BLS12381FieldElement::one() + BLS12381FieldElement::one();

        let false_point2 = G1Point::from_affine(new_x, new_y).unwrap();

        assert!(!false_point2.is_in_subgroup());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_generator() {
        let g = BLS12381Curve::generator();
        let mut compressed_g = BLS12381Curve::compress_g1_point(&g);
        let first_byte = compressed_g.first().unwrap();

        let first_byte_without_control_bits = (first_byte << 3) >> 3;
        compressed_g[0] = first_byte_without_control_bits;

        let compressed_g_x = BLS12381FieldElement::from_bytes_be(&compressed_g).unwrap();
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_point_at_inf() {
        let inf = G1Point::neutral_element();
        let compressed_inf = BLS12381Curve::compress_g1_point(&inf);
        let first_byte = compressed_inf.first().unwrap();

        assert_eq!(*first_byte >> 6, 3_u8);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_generator() {
        let g = BLS12381Curve::generator();
        let mut compressed_g_slice = BLS12381Curve::compress_g1_point(&g);

        let decompressed_g = BLS12381Curve::decompress_g1_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_2g() {
        let g = BLS12381Curve::generator();
        let g_2 = g.operate_with_self(UnsignedInteger::<4>::from("2"));

        let mut compressed_g2_slice: [u8; 48] = BLS12381Curve::compress_g1_point(&g_2);

        let decompressed_g2 = BLS12381Curve::decompress_g1_point(&mut compressed_g2_slice).unwrap();

        assert_eq!(g_2, decompressed_g2);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_generator_g2() {
        use crate::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;

        let g = BLS12381TwistCurve::generator();
        let mut compressed_g_slice = BLS12381Curve::compress_g2_point(&g);

        let decompressed_g = BLS12381Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_generator_g2_neg() {
        use crate::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;

        let g = BLS12381TwistCurve::generator();
        let g_neg = g.neg();

        let mut compressed_g_neg_slice = BLS12381Curve::compress_g2_point(&g_neg);

        let decompressed_g_neg =
            BLS12381Curve::decompress_g2_point(&mut compressed_g_neg_slice).unwrap();

        assert_eq!(g_neg, decompressed_g_neg);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_g2() {
        use crate::{
            elliptic_curve::short_weierstrass::{
                curves::bls12_381::{
                    field_extension::Degree2ExtensionField, twist::BLS12381TwistCurve,
                },
                traits::Compress,
            },
            field::element::FieldElement,
        };

        // Valid G2 point coordinates:
        let x_0 = BLS12381FieldElement::from_hex_unchecked("02");
        let x_1 = BLS12381FieldElement::from_hex_unchecked("0");
        let y_0 = BLS12381FieldElement::from_hex_unchecked(
            "013a59858b6809fca4d9a3b6539246a70051a3c88899964a42bc9a69cf9acdd9dd387cfa9086b894185b9a46a402be73",
        );
        let y_1 = BLS12381FieldElement::from_hex_unchecked(
            "02d27e0ec3356299a346a09ad7dc4ef68a483c3aed53f9139d2f929a3eecebf72082e5e58c6da24ee32e03040c406d4f",
        );

        let x: FieldElement<Degree2ExtensionField> = FieldElement::new([x_0, x_1]);
        let y: FieldElement<Degree2ExtensionField> = FieldElement::new([y_0, y_1]);

        let point = BLS12381TwistCurve::create_point_from_affine(x, y).unwrap();

        let compress_point = BLS12381Curve::compress_g2_point(&point);

        let mut valid_compressed_point = [0_u8; 96];
        valid_compressed_point[0] |= 1 << 7;
        valid_compressed_point[95] |= 1 << 1;

        assert_eq!(compress_point, valid_compressed_point);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_decompress_g2() {
        use crate::{
            elliptic_curve::short_weierstrass::curves::bls12_381::{
                field_extension::Degree2ExtensionField, twist::BLS12381TwistCurve,
            },
            field::element::FieldElement,
        };

        let mut compressed_point = [0_u8; 96];
        compressed_point[0] |= 1 << 7;
        compressed_point[95] |= 1 << 1;

        // Valid G2 point coordinates:
        let x_0 = BLS12381FieldElement::from_hex_unchecked("02");
        let x_1 = BLS12381FieldElement::from_hex_unchecked("0");
        let y_0 = BLS12381FieldElement::from_hex_unchecked(
            "013a59858b6809fca4d9a3b6539246a70051a3c88899964a42bc9a69cf9acdd9dd387cfa9086b894185b9a46a402be73",
        );
        let y_1 = BLS12381FieldElement::from_hex_unchecked(
            "02d27e0ec3356299a346a09ad7dc4ef68a483c3aed53f9139d2f929a3eecebf72082e5e58c6da24ee32e03040c406d4f",
        );

        let x: FieldElement<Degree2ExtensionField> = FieldElement::new([x_0, x_1]);
        let y: FieldElement<Degree2ExtensionField> = FieldElement::new([y_0, y_1]);

        let valid_g2_point = BLS12381TwistCurve::create_point_from_affine(x, y).unwrap();

        let decompressed_point = BLS12381Curve::decompress_g2_point(&mut compressed_point).unwrap();

        assert_eq!(valid_g2_point, decompressed_point);
    }
}

#[cfg(test)]
mod twist_tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::field_extension::{BLS12381PrimeField, Degree2ExtensionField},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::U384,
    };

    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
    type Level0FE = FieldElement<BLS12381PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;

    #[cfg(feature = "alloc")]
    use crate::elliptic_curve::short_weierstrass::point::{
        Endianness, PointFormat, ShortWeierstrassProjectivePoint,
    };

    #[test]
    fn create_generator() {
        let g = BLS12381TwistCurve::generator();
        let [x, y, _] = g.coordinates();
        assert_eq!(
            BLS12381TwistCurve::defining_equation(x, y),
            Level1FE::zero()
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn serialize_deserialize_generator() {
        let g = BLS12381TwistCurve::generator();
        let bytes = g.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let deserialized = ShortWeierstrassProjectivePoint::<BLS12381TwistCurve>::deserialize(
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
            Level0FE::new(U384::from_hex_unchecked(
                "97798b4a61ac301bbee71e36b5174e2f4adfe3e1729bdae1fcc9965ae84181be373aa80414823eed694f1270014012d",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "c9852cc6e61868966249aec153b50b29b3c22409f4c7880fd13121981c103c8ef84d9ea29b552431360e82cf69219fa",
            )),
        ]);
        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "16cb3a60f3fa52c8273aceeb94c4c7303e8074aa9eedec7355bbb1e8cceedd4ec1497f573f62822140377b8e339619ed",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "1cd919b08afe06bebe9adf6223a55868a6fd8b77efc5c67b60fff39be36e9b44b7f10db16827c83b43ad2dad1947778",
            )),
        ]);
        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "b6bce994c23f6505131a5f6d4ce4ba30f5dab726780bef00517585cab02e17f4d015b26eeaff376dc236af26c0210f1",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "163163e71fdd96a84b6a24d3e7cd9d7c0f06961e6fe8b7ec9b27bef1dbef5cbaf557563f725229fc79814a294c0b8511",
            )),
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "1c6afffac96cd457b4ac797e5cef6951c83bb328737f57df44ba0cc513d499f736816877a6cf87f1359e79d10151e14",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "79e40e569c20182726c148ca72a6e862d03317a2cf75cd19c2be36e29e03da70acbefbfa7a4c4e1c088bf94ae6ba6ce",
            )),
        ]);
        let expectedx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "63f209cd306e632cc91089bd6b3bb02a6679fd02931a6e2292976589426dfdff9366829d5f45d982413e8b9514e8965",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "11aae43845fcb3e633217c2851889cddb939a3d2ddf00a64e4e0a723c362dff2caabc640a1095ac5be4075d4f7edf17f",
            )),
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "83e21ca01826bca9221373faf03132b80128c24760c639b44bd7e0b6c11537ef239c01d31a25a58c7f67fb16df0234b",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "f45243fd699bba6c6ca644ad8070f7812e4987fb2c91f64139a293958ed373814ef7317c11c3496cd93b88871f5d2c7",
            )),
        ]);
        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BLS12381TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }

    #[test]
    fn add_points2() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0x1414a51107b5ca989957046a1126425d371f5124215e294770f67fbf14dd92bbf1c9c2dbf35441769fa88427c17f0bb5",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x6224c8c8d6ecb882197551c68a25340be33975948d7da7568f6e00131307dc3688d320ad3c3c7cb95625082a47908f2",
            )),
        ]);
        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0xa69bb992a48dabcc49ab3fa1508bbc1acae14a9af09db39290b303de518806cec0067486adb6044f936d4bd2e5a151",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x98d34282ed5a2e265e455af63c66f7b5dd1557296f775463bcea891a14a801baa172e923055c4bb0fd5343e86294f41",
            )),
        ]);

        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0xae980b8c7483736e1904a1643e7a46f9980e52a5e65f0c5f7d195b30efd7173adea992c49a9073572d64ba67470e406",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x57d195c5f11d93558b52a74be27ae07f82f908ce35fabe58ce212c6d0bcef4a9f25e31fe92b2a49ea3fbc5d6c8cde99",
            )),
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0x160c8799bfe8b80732e9ccb33ad35d1d23b6d01d8170ad088118a3cfe2a97dba2cf06ac3ce202fe039b105082ea48c22",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0xc1e4cf5b2ca3deb2ec4f95b0ca0dbe79b0fc119f16f525e1d00054f009bcf2e2bde26f820b163e488bdc248adee1bfe",
            )),
        ]);

        let expectedx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0xc9ef648884ce132f76f51a3fc536cc33e93c3a687c518abe1a0ddd4389df68c28214505a4b4a11a62d5b251badc7f9",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x6a43f39c279791e3ae0d36c3c24bee770593b80f44f931d70bbdcda17f9ed53b682dba192aa3c92b18d25e5d49b7d04",
            )),
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0x128764b35cb5dbfe604968c5e3734ca80e16ee940f966c260096c29dd15aa6c6de3b1fc68a085403b8bdc012bf3b5b30",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x13471bd588bda43ce76f52dba32298bb46cfcf97a5f4484486e4394f6e38f7bd807ba62216d57ed8fd9df5f608c55ef1",
            )),
        ]);

        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BLS12381TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }

    #[test]
    // Numbers checked in SAGE
    fn operate_with_self_test() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0x1414a51107b5ca989957046a1126425d371f5124215e294770f67fbf14dd92bbf1c9c2dbf35441769fa88427c17f0bb5",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x6224c8c8d6ecb882197551c68a25340be33975948d7da7568f6e00131307dc3688d320ad3c3c7cb95625082a47908f2",
            )),
        ]);

        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0xa69bb992a48dabcc49ab3fa1508bbc1acae14a9af09db39290b303de518806cec0067486adb6044f936d4bd2e5a151",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x98d34282ed5a2e265e455af63c66f7b5dd1557296f775463bcea891a14a801baa172e923055c4bb0fd5343e86294f41",
            )),
        ]);

        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0x16ba99ac9a28190dd74b8988e7a833f60e398472c363c2254c7db3138aff3a0858fb23e6cd2ca814a021b6b3b983f14a",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0xe1356660c4a00b7ba5021f81949bd96680df9fa464a70d257c7b1bcae0e28ec15d84ddcef2ca2e4e8531f50177685dd",
            )),
        ]);

        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked(
                "0x9883c1c7d10c32d584f1cf5f0a7c0742f9b283144290afd6871abcb585e434516cefd2b159d99d75771f5658f0af628",
            )),
            Level0FE::new(U384::from_hex_unchecked(
                "0x13e5df65d734c9decf24356dacfcf9c4a317e5d21a7d1ada728f59e46ddfb137214bab47e8629a8016b6e508cafe141a",
            )),
        ]);

        let scalar = U384::from_hex_unchecked(
            "0x1752428b56412bc55b5c6aca6e1811d1b5d810afd55169d8cffeae326bc8d6ea",
        );

        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();

        assert_eq!(p.operate_with_self(scalar), q);
    }
}

#[cfg(test)]
mod sqrt_tests {
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::Degree2ExtensionField;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::sqrt::sqrt_qfe;
    use crate::field::element::FieldElement;

    type BLS12381TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

    #[test]
    fn test_sqrt_qfe() {
        let c1 = BLS12381FieldElement::from_hex(
            "0x13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
        ).unwrap();
        let c0 = BLS12381FieldElement::from_hex(
        "0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"
        ).unwrap();
        let qfe = BLS12381TwistCurveFieldElement::new([c0, c1]);

        let b1 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let b0 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let qfe_b = BLS12381TwistCurveFieldElement::new([b0, b1]);

        let cubic_value = qfe.pow(3_u64) + qfe_b;
        let root = sqrt_qfe(&cubic_value, 0).unwrap();

        let c0_expected = BLS12381FieldElement::from_hex("0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801").unwrap();
        let c1_expected = BLS12381FieldElement::from_hex("0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be").unwrap();
        let qfe_expected = BLS12381TwistCurveFieldElement::new([c0_expected, c1_expected]);

        let value_root = root.value();
        let value_qfe_expected = qfe_expected.value();

        assert_eq!(value_root[0].clone(), value_qfe_expected[0].clone());
        assert_eq!(value_root[1].clone(), value_qfe_expected[1].clone());
    }

    #[test]
    fn test_sqrt_qfe_2() {
        let c0 = BLS12381FieldElement::from_hex("0x02").unwrap();
        let c1 = BLS12381FieldElement::from_hex("0x00").unwrap();
        let qfe = BLS12381TwistCurveFieldElement::new([c0, c1]);

        let c0_expected = BLS12381FieldElement::from_hex("0x013a59858b6809fca4d9a3b6539246a70051a3c88899964a42bc9a69cf9acdd9dd387cfa9086b894185b9a46a402be73").unwrap();
        let c1_expected = BLS12381FieldElement::from_hex("0x02d27e0ec3356299a346a09ad7dc4ef68a483c3aed53f9139d2f929a3eecebf72082e5e58c6da24ee32e03040c406d4f").unwrap();
        let qfe_expected = BLS12381TwistCurveFieldElement::new([c0_expected, c1_expected]);

        let b1 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let b0 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let qfe_b = BLS12381TwistCurveFieldElement::new([b0, b1]);

        let root = sqrt_qfe(&(qfe.pow(3_u64) + qfe_b), 0).unwrap();

        let value_root = root.value();
        let value_qfe_expected = qfe_expected.value();

        assert_eq!(value_root[0].clone(), value_qfe_expected[0].clone());
        assert_eq!(value_root[1].clone(), value_qfe_expected[1].clone());
    }
}
