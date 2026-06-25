//! Arithmetic in the secp256k1 base field `F_p` with `p = 2^256 - 2^32 - 977`.
//!
//! Elements are stored as `U256` always reduced into `[0, p)`. This is test-only
//! reference arithmetic for cross-checking the k256-backed witness generator.

use crypto_bigint::modular::runtime_mod::{DynResidue, DynResidueParams};
use crypto_bigint::{NonZero, U256};

use crate::p;

fn params() -> DynResidueParams<4> {
    DynResidueParams::new(&p())
}

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
        let pm = params();
        let a = DynResidue::new(&self.0, pm);
        let b = DynResidue::new(&other.0, pm);
        Fp((a + b).retrieve())
    }

    pub(crate) fn sub(&self, other: &Fp) -> Fp {
        let pm = params();
        let a = DynResidue::new(&self.0, pm);
        let b = DynResidue::new(&other.0, pm);
        Fp((a - b).retrieve())
    }

    pub(crate) fn mul(&self, other: &Fp) -> Fp {
        let pm = params();
        let a = DynResidue::new(&self.0, pm);
        let b = DynResidue::new(&other.0, pm);
        Fp((a * b).retrieve())
    }

    /// Multiplicative inverse via Fermat's little theorem (`p` is prime): `self^(p-2)`.
    /// Returns zero for a zero input (which never occurs for valid curve arithmetic).
    pub(crate) fn inv(&self) -> Fp {
        let pm = params();
        let a = DynResidue::new(&self.0, pm);
        // exponent = p - 2
        let exp = p().wrapping_sub(&U256::from(2u32));
        Fp(a.pow(&exp).retrieve())
    }
}
