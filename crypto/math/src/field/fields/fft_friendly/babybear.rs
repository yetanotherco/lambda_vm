use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsFFTField,
    },
    unsigned_integer::element::{U64, UnsignedInteger},
};

pub type U64MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 1>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigBabybear31PrimeField;
impl IsModulus<U64> for MontgomeryConfigBabybear31PrimeField {
    //Babybear Prime p = 2^31 - 2^27 + 1 = 0x78000001
    const MODULUS: U64 = U64::from_u64(2013265921);
}

pub type Babybear31PrimeField =
    U64MontgomeryBackendPrimeField<MontgomeryConfigBabybear31PrimeField>;

//a two-adic primitive root of unity is 21^(2^24)
// 21^(2^24)=1 mod 2013265921
// 2^27(2^4-1)+1 where n=27 (two-adicity) and k=2^4+1

impl IsFFTField for Babybear31PrimeField {
    const TWO_ADICITY: u64 = 24;

    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = UnsignedInteger { limbs: [21] };

    fn field_name() -> &'static str {
        "babybear31"
    }
}

impl FieldElement<Babybear31PrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_le_bytes()
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_be_bytes()
    }
}
