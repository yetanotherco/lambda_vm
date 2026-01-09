#[cfg(test)]
mod tests {
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::{
        CURVE_COFACTOR, SUBGROUP_ORDER,
    };
    use crate::elliptic_curve::short_weierstrass::point::{
        Endianness, PointFormat, ShortWeierstrassJacobianPoint, ShortWeierstrassProjectivePoint,
    };
    use crate::elliptic_curve::traits::{FromAffine, IsEllipticCurve};
    use crate::errors::DeserializationError;

    #[cfg(feature = "alloc")]
    use crate::{
        elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
        field::element::FieldElement,
    };

    #[cfg(feature = "alloc")]
    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;

    #[cfg(feature = "alloc")]
    fn point() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn operate_with_works_jacobian() {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );
        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        assert_eq!(p.operate_with(&p), p.double());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn operate_with_self_works_jacobian() {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );
        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        assert_eq!(
            p.operate_with_self(5_u16),
            p.double().double().operate_with(&p)
        );
    }
    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_projective() {
        let expected_point = point();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_uncompressed() {
        let expected_point = point();
        let bytes_be = expected_point.serialize(PointFormat::Uncompressed, Endianness::BigEndian);
        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_le_projective() {
        let expected_point = point();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_le_uncompressed() {
        let expected_point = point();
        let bytes_be =
            expected_point.serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_projective() {
        let bytes = point().serialize(PointFormat::Projective, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Projective,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_uncompressed() {
        let bytes = point().serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work_projective() {
        let bytes = point().serialize(PointFormat::Projective, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work_uncompressed() {
        let bytes = point().serialize(PointFormat::Uncompressed, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_le_projective() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_le_uncompressed() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_be_projective() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Projective,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_be_uncompressed() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn test_jacobian_vs_projective_operation() {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );

        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x.clone(), y.clone())
            .unwrap();
        let q = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        let sum_jacobian = p.operate_with_self(7_u16);
        let sum_projective = q.operate_with_self(7_u16);

        // Convert the result to affine coordinates
        let sum_jacobian_affine = sum_jacobian.to_affine();
        let [x_j, y_j, _] = sum_jacobian_affine.coordinates();

        // Convert the result to affine coordinates
        let binding = sum_projective.to_affine();
        let [x_p, y_p, _] = binding.coordinates();

        assert_eq!(x_j, x_p, "x coordintates do not match");
        assert_eq!(y_j, y_p, "y coordinates do not match");
    }

    #[test]
    fn test_multiplication_by_order_projective() {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );

        let p = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(x.clone(), y.clone())
            .unwrap();

        let g = p
            .operate_with_self(SUBGROUP_ORDER)
            .operate_with_self(CURVE_COFACTOR);

        assert!(
            g.is_neutral_element(),
            "Multiplication by order should result in the neutral element"
        );
    }

    #[test]
    fn test_multiplication_by_order_jacobian() {
        let x = FEE::new_base(
            "36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5",
        );
        let y = FEE::new_base(
            "7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0",
        );

        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x.clone(), y.clone())
            .unwrap();
        let g = p
            .operate_with_self(SUBGROUP_ORDER)
            .operate_with_self(CURVE_COFACTOR);

        assert!(
            g.is_neutral_element(),
            "Multiplication by order should result in the neutral element"
        );
    }
}
