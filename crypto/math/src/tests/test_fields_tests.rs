#[cfg(test)]
mod tests_u64_test_field {
    use crate::field::{
        element::FieldElement,
        test_fields::u64_test_field::{U64TestField, U64TestFieldExtension},
        traits::IsPrimeField,
    };

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(U64TestField::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn bit_size_of_test_field_is_64() {
        assert_eq!(
            <U64TestField as crate::field::traits::IsPrimeField>::field_bit_size(),
            64
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_to_subfield_vec() {
        let a = FieldElement::<U64TestFieldExtension>::from(&[
            FieldElement::from(1),
            FieldElement::from(3),
        ]);
        let b = a.to_subfield_vec::<U64TestField>();
        assert_eq!(b, alloc::vec![FieldElement::from(1), FieldElement::from(3)]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let num = U64TestField::from_hex("B").unwrap();
        assert_eq!(U64TestField::to_hex(&num), "B");
    }
}

#[cfg(test)]
mod tests_u32_test_field {
    use crate::field::{test_fields::u32_test_field::U32TestField, traits::IsPrimeField};

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(U32TestField::from_hex("B").unwrap(), 11);
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let num = U32TestField::from_hex("B").unwrap();
        assert_eq!(U32TestField::to_hex(&num), "B");
    }

    #[test]
    fn bit_size_of_test_field_is_31() {
        assert_eq!(
            <U32TestField as crate::field::traits::IsPrimeField>::field_bit_size(),
            31
        );
    }
}
