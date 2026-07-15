//! Field-extension multiply-add backend: `a*b + c`.
//!
//! Software by default (host, and every non-accelerated field). On the riscv64
//! guest the native degree-3 Goldilocks extension routes through the FEXT
//! precompile, so the in-VM STARK verifier's Fp3 multiplies run on the
//! accelerator instead of the software schoolbook. Mirrors the host/guest split
//! of [`crate::hash::platform_keccak`].
//!
//! The trait carries software defaults, so the generic verifier can call
//! `FieldExtension::fma(..)` / `ext_mul(..)` unconditionally: on host (and any
//! non-Fp3 field) it is the plain arithmetic; on the guest the Fp3 impl
//! overrides it. The two impls live under mutually exclusive `cfg`s, so there is
//! never an overlap (no `specialization` needed — the host builds on stable).

use math::field::element::FieldElement;
use math::field::traits::IsField;

/// `a*b + c` over a field extension. Default is software; accelerated fields
/// override on targets where an accelerator exists.
pub trait Fp3Fma: IsField + Sized {
    /// Fused multiply-add: `a * b + c`.
    #[inline(always)]
    fn fma(
        a: &FieldElement<Self>,
        b: &FieldElement<Self>,
        c: &FieldElement<Self>,
    ) -> FieldElement<Self> {
        a * b + c
    }

    /// Extension multiply: `a * b`.
    #[inline(always)]
    fn ext_mul(a: &FieldElement<Self>, b: &FieldElement<Self>) -> FieldElement<Self> {
        a * b
    }
}

#[cfg(not(target_arch = "riscv64"))]
mod imp {
    use super::Fp3Fma;
    use math::field::traits::IsField;

    impl<E: IsField> Fp3Fma for E {}
}

#[cfg(target_arch = "riscv64")]
mod imp {
    use super::Fp3Fma;
    use lambda_vm_syscalls::syscalls::{fext_fma, fext_load, fext_store};
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksElement;

    // Verifier-scratch field-storage handles, in a reserved high range no guest
    // picks for its own field-storage. FMA requires `out/a/b/c` pairwise
    // distinct; `H_ZERO` is never written, so it reads as the zero element.
    const BASE: u64 = 0xFFFF_0000_0000_0000;
    const H_A: u64 = BASE;
    const H_B: u64 = BASE + 1;
    const H_C: u64 = BASE + 2;
    const H_OUT: u64 = BASE + 3;
    const H_ZERO: u64 = BASE + 4;

    type Fp3 = Degree3GoldilocksExtensionField;

    #[inline]
    fn coeffs(x: &FieldElement<Fp3>) -> [u64; 3] {
        let v = x.value();
        [
            v[0].canonical_u64(),
            v[1].canonical_u64(),
            v[2].canonical_u64(),
        ]
    }

    #[inline]
    fn from_coeffs(c: [u64; 3]) -> FieldElement<Fp3> {
        FieldElement::from_raw([
            GoldilocksElement::from_raw(c[0]),
            GoldilocksElement::from_raw(c[1]),
            GoldilocksElement::from_raw(c[2]),
        ])
    }

    impl Fp3Fma for Fp3 {
        fn fma(
            a: &FieldElement<Fp3>,
            b: &FieldElement<Fp3>,
            c: &FieldElement<Fp3>,
        ) -> FieldElement<Fp3> {
            fext_load(H_A, &coeffs(a));
            fext_load(H_B, &coeffs(b));
            fext_load(H_C, &coeffs(c));
            fext_fma(H_A, H_B, H_C, H_OUT);
            from_coeffs(fext_store(H_OUT))
        }

        fn ext_mul(a: &FieldElement<Fp3>, b: &FieldElement<Fp3>) -> FieldElement<Fp3> {
            fext_load(H_A, &coeffs(a));
            fext_load(H_B, &coeffs(b));
            fext_fma(H_A, H_B, H_ZERO, H_OUT);
            from_coeffs(fext_store(H_OUT))
        }
    }
}
