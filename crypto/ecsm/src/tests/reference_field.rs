//! Arithmetic in the secp256k1 base field `F_p` with `p = 2^256 - 2^32 - 977`.
//!
//! Elements are stored as `BigUint` always reduced into `[0, p)`. This is test-only
//! reference arithmetic for cross-checking the k256-backed witness generator.

use num_bigint::BigUint;

use crate::p;

/// An element of the secp256k1 base field, kept reduced into `[0, p)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Fp(pub(crate) BigUint);

impl Fp {
    /// Reduces an arbitrary value into the field.
    pub(crate) fn new(v: BigUint) -> Self {
        Fp(v % p())
    }

    pub(crate) fn from_u64(v: u64) -> Self {
        Fp(BigUint::from(v) % p())
    }

    /// `self + other mod p`. Both operands must already be reduced.
    pub(crate) fn add(&self, other: &Fp) -> Fp {
        Fp((&self.0 + &other.0) % p())
    }

    /// `self - other mod p`. Both operands must already be reduced.
    pub(crate) fn sub(&self, other: &Fp) -> Fp {
        let t = &self.0 + p(); // in [p, 2p)
        Fp((t - &other.0) % p())
    }

    /// `self * other mod p`. Both operands must already be reduced.
    pub(crate) fn mul(&self, other: &Fp) -> Fp {
        Fp((&self.0 * &other.0) % p())
    }

    /// Multiplicative inverse via Fermat's little theorem (`p` is prime): `self^(p-2)`.
    /// Returns zero for a zero input (which never occurs for valid curve arithmetic).
    pub(crate) fn inv(&self) -> Fp {
        Fp(self.0.modpow(&(p() - BigUint::from(2u32)), &p()))
    }
}
