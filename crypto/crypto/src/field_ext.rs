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

    /// A resident accumulator for `acc += a*b*c` chains. On the accelerated
    /// guest it lives in field-storage across the chain (operands loaded, the
    /// accumulator never stored/reloaded mid-chain), removing the per-op
    /// LOAD/STORE roundtrip of [`Fp3Fma::fma`]. On host it is a plain field
    /// element.
    type ProdAcc;

    /// A fresh zero accumulator.
    fn prod_acc_new() -> Self::ProdAcc;

    /// `acc += a * b * c`.
    fn prod_acc_add(
        acc: &mut Self::ProdAcc,
        a: &FieldElement<Self>,
        b: &FieldElement<Self>,
        c: &FieldElement<Self>,
    );

    /// Materialize the accumulator as a field element.
    fn prod_acc_finish(acc: Self::ProdAcc) -> FieldElement<Self>;

    /// A fresh zero two-way resident accumulator for `acc += a*b` folds. `slot`
    /// picks one of a fixed set of independent accumulators (the guest keeps each
    /// in its own field-storage handle pair) so several folds can run at once —
    /// e.g. the regular and symmetric row sums of a DEEP query. Callers must keep
    /// distinct live accumulators on distinct slots and not exceed the guest's
    /// slot count; host ignores `slot`. Materialize with `prod_acc_finish`.
    fn mul_acc_new(slot: u8) -> Self::ProdAcc;

    /// `acc += a * b`. Like `prod_acc_add` but without the third factor: on the
    /// guest it is a single FMA into the resident accumulator (no `a*b` temp),
    /// matching the `fma(a, b, acc)` folds the verifier runs over trace columns.
    fn mul_acc_add(acc: &mut Self::ProdAcc, a: &FieldElement<Self>, b: &FieldElement<Self>);
}

#[cfg(not(target_arch = "riscv64"))]
mod imp {
    use super::Fp3Fma;
    use math::field::element::FieldElement;
    use math::field::traits::IsField;

    impl<E: IsField> Fp3Fma for E {
        type ProdAcc = FieldElement<E>;

        fn prod_acc_new() -> FieldElement<E> {
            FieldElement::zero()
        }

        fn prod_acc_add(
            acc: &mut FieldElement<E>,
            a: &FieldElement<E>,
            b: &FieldElement<E>,
            c: &FieldElement<E>,
        ) {
            *acc = &*acc + &(a * b * c);
        }

        fn prod_acc_finish(acc: FieldElement<E>) -> FieldElement<E> {
            acc
        }

        fn mul_acc_new(_slot: u8) -> FieldElement<E> {
            FieldElement::zero()
        }

        fn mul_acc_add(acc: &mut FieldElement<E>, a: &FieldElement<E>, b: &FieldElement<E>) {
            *acc = &*acc + &(a * b);
        }
    }
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
    // ProdAcc scratch: `H_T` holds `a*b`; the accumulator ping-pongs between
    // `H_ACC0`/`H_ACC1` so every emitted FMA has `out != c` (the executor's
    // pairwise-distinct guard forbids in-place accumulation).
    const H_T: u64 = BASE + 5;
    // Resident-accumulator handle pairs, one per slot: slot `s` owns
    // `(ACC_BASE + 2s, ACC_BASE + 2s + 1)` and ping-pongs between them so every
    // emitted FMA has `out != c`. `MAX_SLOTS` independent accumulators can be
    // live at once (e.g. the regular and symmetric DEEP row sums).
    const ACC_BASE: u64 = BASE + 6;
    const MAX_SLOTS: u8 = 4;

    #[inline]
    fn acc_pair(slot: u8) -> (u64, u64) {
        debug_assert!(slot < MAX_SLOTS);
        let lo = ACC_BASE + 2 * slot as u64;
        (lo, lo + 1)
    }

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

        type ProdAcc = super::GuestAcc;

        fn prod_acc_new() -> super::GuestAcc {
            // Zero the starting handle: it may hold a stale value from a
            // previous chain (uninitialized-reads-as-zero doesn't apply here).
            let (lo, _) = acc_pair(0);
            fext_load(lo, &[0, 0, 0]);
            super::GuestAcc { buf: 0, slot: 0 }
        }

        fn prod_acc_add(
            acc: &mut super::GuestAcc,
            a: &FieldElement<Fp3>,
            b: &FieldElement<Fp3>,
            c: &FieldElement<Fp3>,
        ) {
            fext_load(H_A, &coeffs(a));
            fext_load(H_B, &coeffs(b));
            fext_load(H_C, &coeffs(c));
            // tmp = a * b
            fext_fma(H_A, H_B, H_ZERO, H_T);
            let (lo, hi) = acc_pair(acc.slot);
            let (cur, alt) = if acc.buf == 0 { (lo, hi) } else { (hi, lo) };
            // alt = tmp * c + cur   (out=alt != c=cur, satisfies the guard)
            fext_fma(H_T, H_C, cur, alt);
            acc.buf ^= 1;
        }

        fn prod_acc_finish(acc: super::GuestAcc) -> FieldElement<Fp3> {
            let (lo, hi) = acc_pair(acc.slot);
            let cur = if acc.buf == 0 { lo } else { hi };
            from_coeffs(fext_store(cur))
        }

        fn mul_acc_new(slot: u8) -> super::GuestAcc {
            let (lo, _) = acc_pair(slot);
            fext_load(lo, &[0, 0, 0]);
            super::GuestAcc { buf: 0, slot }
        }

        fn mul_acc_add(
            acc: &mut super::GuestAcc,
            a: &FieldElement<Fp3>,
            b: &FieldElement<Fp3>,
        ) {
            fext_load(H_A, &coeffs(a));
            fext_load(H_B, &coeffs(b));
            let (lo, hi) = acc_pair(acc.slot);
            let (cur, alt) = if acc.buf == 0 { (lo, hi) } else { (hi, lo) };
            // alt = a * b + cur   (out=alt != c=cur, satisfies the guard)
            fext_fma(H_A, H_B, cur, alt);
            acc.buf ^= 1;
        }
    }
}

/// Guest resident-accumulator state: `slot` selects the handle pair, `buf`
/// selects which of that pair currently holds the running value. (Both stay
/// private; only this module's guest impl constructs and reads them.)
#[cfg(target_arch = "riscv64")]
pub struct GuestAcc {
    buf: u8,
    slot: u8,
}

#[cfg(test)]
mod tests {
    use super::Fp3Fma;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Fp3F;
    use math::field::goldilocks::GoldilocksElement;

    type Fp3 = FieldElement<Fp3F>;

    fn e(x: [u64; 3]) -> Fp3 {
        Fp3::from_raw([
            GoldilocksElement::from(x[0]),
            GoldilocksElement::from(x[1]),
            GoldilocksElement::from(x[2]),
        ])
    }

    /// `fma`/`ext_mul` must equal the plain field arithmetic they replace. On
    /// host this exercises the software default (which also runs for every
    /// non-Fp3 field); the guest FEXT impl is covered by the executor's
    /// `fext_fma` tests and the recursion prove/verify E2E.
    #[test]
    fn fma_and_ext_mul_match_field_arithmetic() {
        let cases = [
            ([1u64, 2, 3], [4u64, 5, 6], [7u64, 8, 9]),
            ([0, 0, 0], [9, 9, 9], [1, 2, 3]),
            ([u64::MAX - 1, 0, 5], [2, 3, 4], [0, 0, 0]),
            ([10, 20, 30], [10, 20, 30], [5, 5, 5]),
            ([123456789, 987654321, 555], [1, 0, 0], [0, 1, 0]),
        ];
        for (a, b, c) in cases {
            let (a, b, c) = (e(a), e(b), e(c));
            assert_eq!(Fp3F::fma(&a, &b, &c), &a * &b + &c);
            assert_eq!(Fp3F::ext_mul(&a, &b), &a * &b);
        }
    }

    /// The resident accumulators must fold to the same value as the plain sums
    /// they replace: `prod_acc` to `Σ aᵢ·bᵢ·cᵢ`, `mul_acc` to `Σ aᵢ·bᵢ`. Two
    /// `mul_acc` chains on distinct slots run interleaved (as the verifier folds
    /// the regular and symmetric row sums at once) and must stay independent.
    #[test]
    fn resident_accumulators_match_plain_sums() {
        let terms = [
            ([1u64, 2, 3], [4u64, 5, 6], [7u64, 8, 9]),
            ([10, 0, 5], [2, 3, 4], [1, 1, 1]),
            ([123, 456, 789], [9, 8, 7], [2, 0, 4]),
        ];

        let mut prod = Fp3F::prod_acc_new();
        let mut expected_prod = Fp3::from_raw([GoldilocksElement::from(0); 3]);
        for (a, b, c) in terms {
            let (a, b, c) = (e(a), e(b), e(c));
            Fp3F::prod_acc_add(&mut prod, &a, &b, &c);
            expected_prod = &expected_prod + &(&a * &b * &c);
        }
        assert_eq!(Fp3F::prod_acc_finish(prod), expected_prod);

        let mut acc0 = Fp3F::mul_acc_new(0);
        let mut acc1 = Fp3F::mul_acc_new(1);
        let mut expected0 = Fp3::from_raw([GoldilocksElement::from(0); 3]);
        let mut expected1 = Fp3::from_raw([GoldilocksElement::from(0); 3]);
        for (a, b, c) in terms {
            let (a, b, c) = (e(a), e(b), e(c));
            Fp3F::mul_acc_add(&mut acc0, &a, &b);
            Fp3F::mul_acc_add(&mut acc1, &a, &c);
            expected0 = &expected0 + &(&a * &b);
            expected1 = &expected1 + &(&a * &c);
        }
        assert_eq!(Fp3F::prod_acc_finish(acc0), expected0);
        assert_eq!(Fp3F::prod_acc_finish(acc1), expected1);
    }
}
