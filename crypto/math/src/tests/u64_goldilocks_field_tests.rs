#[cfg(test)]
mod tests {
    use crate::field::errors::FieldError;
    use crate::field::fields::goldilocks::{reduce_128, Goldilocks64Field};
    use crate::field::traits::{IsField, IsPrimeField};
    type F = Goldilocks64Field;

    // Over the Goldilocks field, the following set of equations hold
    // p               = 0
    // 2^64 - 2^32 + 1 = 0
    // 2^64            = 2^32 - 1
    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(F::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn from_hex_for_0x1_a_is_26() {
        assert_eq!(F::from_hex("0x1a").unwrap(), 26);
    }

    #[test]
    fn bit_size_of_field_is_64() {
        assert_eq!(
            <F as crate::field::traits::IsPrimeField>::field_bit_size(),
            64
        );
    }

    #[test]
    fn one_plus_one_is_two() {
        let a = F::one();
        let b = F::one();
        let c = F::add(&a, &b);
        assert_eq!(c, 2u64);
    }

    #[test]
    fn neg_one_plus_one_is_zero() {
        let a = F::neg(&F::one());
        let b = F::one();
        let c = F::add(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn neg_one_plus_two_is_one() {
        let a = F::neg(&F::one());
        let b = F::from_base_type(2u64);
        let c = F::add(&a, &b);
        assert_eq!(c, F::one());
    }

    #[test]
    fn max_order_plus_one_is_zero() {
        let a = F::from_base_type(F::ORDER - 1);
        let b = F::one();
        let c = F::add(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn comparing_13_and_13_are_equal() {
        let a = F::from_base_type(13);
        let b = F::from_base_type(13);
        assert_eq!(a, b);
    }

    #[test]
    fn comparing_13_and_8_they_are_not_equal() {
        let a = F::from_base_type(13);
        let b = F::from_base_type(8);
        assert_ne!(a, b);
    }

    #[test]
    fn one_sub_one_is_zero() {
        let a = F::one();
        let b = F::one();
        let c = F::sub(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn zero_sub_one_is_order_minus_1() {
        let a = F::zero();
        let b = F::one();
        let c = F::sub(&a, &b);
        assert_eq!(c, F::ORDER - 1);
    }

    #[test]
    fn neg_one_sub_neg_one_is_zero() {
        let a = F::neg(&F::one());
        let b = F::neg(&F::one());
        let c = F::sub(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn neg_one_sub_one_is_neg_one() {
        let a = F::neg(&F::one());
        let b = F::zero();
        let c = F::sub(&a, &b);
        assert_eq!(c, F::neg(&F::one()));
    }

    #[test]
    fn mul_neutral_element() {
        let a = F::from_base_type(1);
        let b = F::from_base_type(2);
        let c = F::mul(&a, &b);
        assert_eq!(c, F::from_base_type(2));
    }

    #[test]
    fn mul_two_three_is_six() {
        let a = F::from_base_type(2);
        let b = F::from_base_type(3);
        assert_eq!(a * b, F::from_base_type(6));
    }

    #[test]
    fn mul_order_neg_one() {
        let a = F::from_base_type(F::ORDER - 1);
        let b = F::from_base_type(F::ORDER - 1);
        let c = F::mul(&a, &b);
        assert_eq!(c, F::from_base_type(1));
    }

    #[test]
    fn pow_p_neg_one() {
        assert_eq!(F::pow(&F::from_base_type(2), F::ORDER - 1), F::one())
    }

    #[test]
    fn inv_zero_error() {
        let result = F::inv(&F::zero());
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_two() {
        let result = F::inv(&F::from_base_type(2u64)).unwrap();
        // sage: 1 / F(2) = 9223372034707292161
        assert_eq!(result, 9223372034707292161);
    }

    #[test]
    fn pow_two_three() {
        assert_eq!(F::pow(&F::from_base_type(2), 3_u64), 8)
    }

    #[test]
    fn div_one() {
        assert_eq!(
            F::div(&F::from_base_type(2), &F::from_base_type(1)).unwrap(),
            2
        )
    }

    #[test]
    fn div_4_2() {
        assert_eq!(
            F::div(&F::from_base_type(4), &F::from_base_type(2)).unwrap(),
            2
        )
    }

    // 1431655766
    #[test]
    fn div_4_3() {
        // sage: F(4) / F(3) = 12297829379609722882
        assert_eq!(
            F::div(&F::from_base_type(4), &F::from_base_type(3)).unwrap(),
            12297829379609722882
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = F::from_base_type(2);

        assert_eq!(F::add(&two, &F::neg(&two)), F::zero())
    }

    #[test]
    fn from_u64_test() {
        let num = F::from_u64(1u64);
        assert_eq!(num, F::one());
    }

    #[test]
    fn from_u64_zero_test() {
        let num = F::from_u64(0);
        assert_eq!(num, F::zero());
    }

    #[test]
    fn from_u64_max_test() {
        let num = F::from_u64(u64::MAX);
        assert_eq!(num, u32::MAX as u64 - 1);
    }

    #[test]
    fn from_u64_order_test() {
        let num = F::from_u64(F::ORDER);
        assert_eq!(num, F::zero());
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_1() {
        let change = 1;
        let f1 = F::from_base_type(F::ORDER + change);
        let f2 = F::from_base_type(F::representative(&f1));
        assert_eq!(f1, f2);
    }

    #[test]
    fn reduct_128() {
        let x = u128::MAX;
        let y = reduce_128(x);
        // The following equalitiy sequence holds, modulo p = 2^64 - 2^32 + 1
        // 2^128 - 1 = (2^64 - 1) * (2^64 + 1)
        //           = (2^32 - 1 - 1) * (2^32 - 1 + 1)
        //           = (2^32 - 2) * (2^32)
        //           = 2^64 - 2 * 2^32
        //           = 2^64 - 2^33
        //           = 2^32 - 1 - 2^33
        //           = - 2^32 - 1
        let expected_result = F::neg(&F::add(&F::from_base_type(2_u64.pow(32)), &F::one()));
        assert_eq!(y, expected_result);
    }

    #[test]
    fn u64_max_as_representative_less_than_u32_max_sub_1() {
        let f = F::from_base_type(u64::MAX);
        assert_eq!(F::representative(&f), u32::MAX as u64 - 1)
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_2() {
        let change = 8;
        let f1 = F::from_base_type(F::ORDER + change);
        let f2 = F::from_base_type(F::representative(&f1));
        assert_eq!(f1, f2);
    }

    #[test]
    fn from_base_type_test() {
        let b = F::from_base_type(1u64);
        assert_eq!(b, F::one());
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let num = F::from_hex("B").unwrap();
        assert_eq!(F::to_hex(&num), "B");
    }
}
