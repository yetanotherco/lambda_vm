//! Arithmetic in the secp256k1 base field `F_p` with `p = 2^256 - 2^32 - 977`.
//!
//! Elements are stored as `BigUint` always reduced into `[0, p)`. This is reference
//! arithmetic used to derive accelerator witnesses — it runs once per `ECALL`, never
//! in a hot loop, so clarity is preferred over speed.

use num_bigint::BigUint;

use crate::p;

/// An element of the secp256k1 base field, kept reduced into `[0, p)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp(pub BigUint);

impl Fp {
    /// Reduces an arbitrary value into the field.
    pub fn new(v: BigUint) -> Self {
        Fp(v % p())
    }

    pub fn from_u64(v: u64) -> Self {
        Fp(BigUint::from(v) % p())
    }

    pub fn zero() -> Self {
        Fp(BigUint::from(0u8))
    }

    pub fn is_zero(&self) -> bool {
        self.0 == BigUint::from(0u8)
    }

    /// `self + other mod p`. Both operands must already be reduced.
    pub fn add(&self, other: &Fp) -> Fp {
        Fp((&self.0 + &other.0) % p())
    }

    /// `self - other mod p`. Both operands must already be reduced.
    pub fn sub(&self, other: &Fp) -> Fp {
        let t = &self.0 + p(); // in [p, 2p)
        Fp((t - &other.0) % p())
    }

    /// `self * other mod p`. Both operands must already be reduced.
    pub fn mul(&self, other: &Fp) -> Fp {
        Fp((&self.0 * &other.0) % p())
    }

    /// Multiplicative inverse via Fermat's little theorem (`p` is prime): `self^(p-2)`.
    /// Returns zero for a zero input (which never occurs for valid curve arithmetic).
    pub fn inv(&self) -> Fp {
        Fp(self.0.modpow(&(p() - BigUint::from(2u32)), &p()))
    }

    /// Square root via `a^((p+1)/4)` (valid because `p ≡ 3 mod 4`).
    /// Returns `None` if `self` is not a quadratic residue.
    pub fn sqrt(&self) -> Option<Fp> {
        let exp = (p() + BigUint::from(1u32)) >> 2u32;
        let r = Fp(self.0.modpow(&exp, &p()));
        if &r.mul(&r) == self { Some(r) } else { None }
    }
}
