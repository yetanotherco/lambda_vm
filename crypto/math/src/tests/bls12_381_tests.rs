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
        BLS12381AtePairing, X, cyclotomic_pow_x, cyclotomic_square, frobenius, miller,
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
}
