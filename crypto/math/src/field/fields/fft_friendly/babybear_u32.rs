use crate::field::{
    fields::u32_montgomery_backend_prime_field::U32MontgomeryBackendPrimeField, traits::IsFFTField,
};

// Babybear Prime p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
pub type Babybear31PrimeField = U32MontgomeryBackendPrimeField<2013265921>;

// p = 2^31 - 2^27 + 1 = 2^27 * (2^4-1) + 1, then
// there is a gruop in the field of order 2^27.
// Since we want to have margin to be able to define a bigger group (blow-up group),
// we define TWO_ADICITY as 24 (so the blow-up factor can be 2^3 = 8).
// A two-adic primitive root of unity is 21^(2^24) because
// 21^(2^24)=1 mod 2013265921.
// In the future we should allow this with cuda feature, and just dispatch it to the CPU until the implementation is done
#[cfg(not(feature = "cuda"))]
impl IsFFTField for Babybear31PrimeField {
    const TWO_ADICITY: u64 = 24;

    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = 21;

    fn field_name() -> &'static str {
        "babybear31"
    }
}
