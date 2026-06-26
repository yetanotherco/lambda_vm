//! Arithmetic in the secp256k1 base field `F_p` with `p = 2^256 - 2^32 - 977`.
//!
//! Elements are stored as `U256` always reduced into `[0, p)`. This is test-only
//! reference arithmetic for cross-checking the k256-backed witness generator.

use crypto_bigint::modular::ConstMontyForm;
use crypto_bigint::{NonZero, U256};

use crate::p;

crypto_bigint::const_monty_params!(
    Secp256k1Field,
    U256,
    "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
);

type FpMonty = ConstMontyForm<Secp256k1Field, 4>;

/// An element of the secp256k1 base field, kept reduced into `[0, p)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Fp(pub(crate) U256);

impl Fp {
    /// Reduces an arbitrary value into the field.
    pub(crate) fn new(v: U256) -> Self {
        let nz = NonZero::new(p()).expect("p != 0");
        let (_, r) = v.div_rem(&nz);
        Fp(r)
    }

    pub(crate) fn from_u64(v: u64) -> Self {
        Fp::new(U256::from(v))
    }

    pub(crate) fn add(&self, other: &Fp) -> Fp {
        Fp((FpMonty::new(&self.0) + FpMonty::new(&other.0)).retrieve())
    }

    pub(crate) fn sub(&self, other: &Fp) -> Fp {
        Fp((FpMonty::new(&self.0) - FpMonty::new(&other.0)).retrieve())
    }

    pub(crate) fn mul(&self, other: &Fp) -> Fp {
        Fp((FpMonty::new(&self.0) * FpMonty::new(&other.0)).retrieve())
    }

    /// Multiplicative inverse via Fermat's little theorem (`p` is prime): `self^(p-2)`.
    /// Returns zero for a zero input (which never occurs for valid curve arithmetic).
    pub(crate) fn inv(&self) -> Fp {
        let exp = p().wrapping_sub(&U256::from(2u32));
        Fp(FpMonty::new(&self.0).pow(&exp).retrieve())
    }
}
