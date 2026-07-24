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
use math::field::traits::{IsField, IsSubFieldOf};

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

    /// Goldilocks×Fp3 asymmetric product `base · ext`, where `base` is a single
    /// element of a subfield `F` (one base coefficient, not a full extension
    /// element). Default is the cheap software subfield product `F * E -> E`
    /// (three base multiplies for Fp3); the accelerated guest routes it through
    /// the FEXT_BASE_MUL chip. `F` must be a proper subfield — a single-
    /// coefficient embedding — which is exactly how the verifier's base columns,
    /// FRI-butterfly `𝜐⁻¹·ζ`, and OOD `g·z` walk use it.
    #[inline(always)]
    fn base_mul<F: IsSubFieldOf<Self>>(
        base: &FieldElement<F>,
        ext: &FieldElement<Self>,
    ) -> FieldElement<Self> {
        // The `*` operator's `F: IsSubFieldOf<Self>` impl — the exact asymmetric
        // subfield product `base · ext` the verifier used before, byte-identical.
        base * ext
    }

    /// A resident accumulator for `acc += a*b(*c)` chains. On the accelerated
    /// guest it lives in field-storage across the chain (operands loaded, the
    /// accumulator never stored/reloaded mid-chain), removing the per-op
    /// LOAD/STORE roundtrip of [`Fp3Fma::fma`]. On host it is a plain field
    /// element.
    ///
    /// Accumulators must be created and finished in LIFO order on the guest (the
    /// backend hands each one a distinct ping-pong region off a stack); the
    /// verifier's chains satisfy this by construction.
    type ProdAcc;

    /// A fresh zero accumulator.
    fn prod_acc_new() -> Self::ProdAcc;

    /// `acc += a * b * c`. Three-operand form (two ext muls); on the guest this
    /// still costs the same ecalls as [`Fp3Fma::fma`] because the product needs
    /// a temporary — prefer [`Fp3Fma::prod_acc_add2`] where the chain is `a*b`.
    fn prod_acc_add(
        acc: &mut Self::ProdAcc,
        a: &FieldElement<Self>,
        b: &FieldElement<Self>,
        c: &FieldElement<Self>,
    );

    /// `acc += a * b`. Two-operand fused accumulate: on the accelerated guest
    /// this is LOAD a, LOAD b, FMA(a, b, acc -> acc) — three ecalls, versus the
    /// five of a stateless `fma` (which reloads and restores the accumulator).
    fn prod_acc_add2(acc: &mut Self::ProdAcc, a: &FieldElement<Self>, b: &FieldElement<Self>);

    /// Materialize the accumulator as a field element.
    fn prod_acc_finish(acc: Self::ProdAcc) -> FieldElement<Self>;

    /// The geometric sequence `[1, base, base², …, base^(count-1)]`.
    ///
    /// Default is the naive running product `cur = &cur * base` (one ext mul per
    /// element). On the accelerated guest the Fp3 impl overrides it to keep both
    /// `base` and the running power resident in field-storage across the whole
    /// sequence — each element costs one FMA + one STORE (2 ecalls) instead of
    /// the two LOADs + FMA + STORE (4 ecalls, plus recanonicalizing `base` every
    /// step) that `&cur * base` emits through the `*` operator. Byte-identical to
    /// the default: the same power values in the same order.
    fn geometric_powers(
        base: &FieldElement<Self>,
        count: usize,
    ) -> alloc::vec::Vec<FieldElement<Self>> {
        let mut out = alloc::vec::Vec::with_capacity(count);
        let mut cur = FieldElement::<Self>::one();
        for _ in 0..count {
            out.push(cur.clone());
            cur = &cur * base;
        }
        out
    }
}

// The `fext-accel` feature routes the Fp3 `fma`/`ext_mul`/`prod_acc` chain
// through the real FEXT accelerator on the riscv64 guest (PR #818/#831).
// Production ships #831 unconditional; here it is a measurement toggle so a
// guest build with the feature ON (accelerator) can be compared cycle-for-cycle
// against an otherwise byte-identical build with it OFF (the software Fp3
// arithmetic below runs on the same target). The verifier's `FieldExtension::fma`
// call sites stay unconditional — only the backend selected here changes.
#[cfg(not(all(target_arch = "riscv64", feature = "fext-accel")))]
mod imp {
    use super::Fp3Fma;
    use math::field::element::FieldElement;
    use math::field::traits::IsField;

    impl<E: IsField> Fp3Fma for E {
        type ProdAcc = FieldElement<E>;

        // `#[inline(always)]` (matching the `fma`/`ext_mul` trait defaults) is
        // load-bearing on the software path: without it the `&mut acc` escapes
        // to a non-inlined call, forcing the 24-byte Fp3 accumulator to stack
        // memory every iteration instead of staying in registers across the
        // chain. Inlined, this compiles to the same register-resident SSA as the
        // stateless `fma` it replaces.
        #[inline(always)]
        fn prod_acc_new() -> FieldElement<E> {
            FieldElement::zero()
        }

        #[inline(always)]
        fn prod_acc_add(
            acc: &mut FieldElement<E>,
            a: &FieldElement<E>,
            b: &FieldElement<E>,
            c: &FieldElement<E>,
        ) {
            *acc = &*acc + &(a * b * c);
        }

        #[inline(always)]
        fn prod_acc_add2(acc: &mut FieldElement<E>, a: &FieldElement<E>, b: &FieldElement<E>) {
            *acc = &*acc + &(a * b);
        }

        #[inline(always)]
        fn prod_acc_finish(acc: FieldElement<E>) -> FieldElement<E> {
            acc
        }
    }
}

#[cfg(all(target_arch = "riscv64", feature = "fext-accel"))]
mod imp {
    use super::Fp3Fma;
    #[cfg(feature = "fext-base-mul")]
    use lambda_vm_syscalls::syscalls::fext_base_mul;
    use lambda_vm_syscalls::syscalls::{fext_fma, fext_load, fext_store};
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksElement;
    use math::field::traits::IsSubFieldOf;

    // Verifier-scratch field-storage handles, in a reserved high range no guest
    // picks for its own field-storage. FMA requires `out/a/b/c` pairwise
    // distinct; `H_ZERO` is never written, so it reads as the zero element.
    const BASE: u64 = 0xFFFF_0000_0000_0000;
    const H_A: u64 = BASE;
    const H_B: u64 = BASE + 1;
    const H_C: u64 = BASE + 2;
    const H_OUT: u64 = BASE + 3;
    const H_ZERO: u64 = BASE + 4;
    // ProdAcc scratch: `H_T` holds `a*b` for the three-operand `prod_acc_add`.
    const H_T: u64 = BASE + 5;

    // Resident-accumulator regions. Each live accumulator owns a ping-pong pair
    // `(H_ACC_BASE + 2*region, +1)` so every emitted FMA has `out != c` (the
    // executor's pairwise-distinct guard forbids in-place accumulation). Regions
    // are handed out off a stack (LIFO); the verifier's chains nest at most
    // `MAX_ACC` deep (2 for the trace-term pair, +2 for the base-row-sum pair
    // that runs inside the full-software column loop).
    const H_ACC_BASE: u64 = BASE + 8;
    const MAX_ACC: u32 = 8;

    // `geometric_powers` scratch, well clear of the H_ACC ping-pong range
    // (`BASE+8..=BASE+23`). `H_GEO_MUL` holds the resident multiplier; `H_GEO_A`
    // / `H_GEO_B` ping-pong the running power. The sequence runs to completion
    // in one call, never nested inside a live `prod_acc` chain, so these three
    // are free for its exclusive use.
    const H_GEO_MUL: u64 = BASE + 0x40;
    const H_GEO_A: u64 = BASE + 0x41;
    const H_GEO_B: u64 = BASE + 0x42;

    // Single-threaded guest: a plain stack depth counter. `prod_acc_new` pushes,
    // `prod_acc_finish` pops. Regions above `MAX_ACC` would alias, so the pool
    // is sized past the deepest nesting the verifier reaches.
    static ACC_DEPTH: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

    #[inline]
    fn acc_handle(region: u32, buf: u8) -> u64 {
        H_ACC_BASE + 2 * region as u64 + buf as u64
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

        // `#[inline(always)]` (matching the trait default and the `*` operator it
        // replaces) is load-bearing on the software branch: without it the
        // generic call doesn't inline into the verifier's butterfly/`g·z` loops,
        // adding ~2M cycles of call overhead across the 55k base×ext sites and
        // making the fext-base-mul-off baseline diverge from the reference.
        #[inline(always)]
        fn base_mul<F: IsSubFieldOf<Fp3>>(
            base: &FieldElement<F>,
            ext: &FieldElement<Fp3>,
        ) -> FieldElement<Fp3> {
            // `F` is the base field: `embed(base) = [base, 0, 0]`, so coeff 0 is
            // the Goldilocks scalar. The chip does the three base multiplies
            // `out[d] = base · ext[d]`; the base rides register x10 by value, so
            // only `ext` needs a LOAD (base_mul is LOAD/BASE_MUL/STORE = 3 ecalls).
            // `fext-base-mul` is a measurement sub-toggle: with it off, base×ext
            // stays the software subfield product (byte-identical to the trait
            // default) while `fma`/`ext_mul` still ride the accelerator, so
            // FEXT_BASE_MUL's delta can be isolated from the rest of the chip.
            #[cfg(feature = "fext-base-mul")]
            {
                let base_u64 = F::embed(base.value().clone())[0].canonical_u64();
                fext_load(H_B, &coeffs(ext));
                fext_base_mul(base_u64, H_B, H_OUT);
                from_coeffs(fext_store(H_OUT))
            }
            #[cfg(not(feature = "fext-base-mul"))]
            {
                // Byte-identical to the `*` operator the verifier used before.
                base * ext
            }
        }

        type ProdAcc = super::GuestAcc;

        #[inline(always)]
        fn prod_acc_new() -> super::GuestAcc {
            use core::sync::atomic::Ordering::Relaxed;
            let region = ACC_DEPTH.fetch_add(1, Relaxed);
            debug_assert!(region < MAX_ACC, "resident-accumulator pool exhausted");
            // Zero the starting handle: field-storage reads unwritten cells as
            // zero, but a reused region may hold a stale value from a prior
            // chain, so clear it explicitly.
            fext_load(acc_handle(region, 0), &[0, 0, 0]);
            super::GuestAcc { region, buf: 0 }
        }

        #[inline(always)]
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
            let cur = acc_handle(acc.region, acc.buf);
            let alt = acc_handle(acc.region, acc.buf ^ 1);
            // alt = tmp * c + cur   (out=alt != c=cur, satisfies the guard)
            fext_fma(H_T, H_C, cur, alt);
            acc.buf ^= 1;
        }

        #[inline(always)]
        fn prod_acc_add2(acc: &mut super::GuestAcc, a: &FieldElement<Fp3>, b: &FieldElement<Fp3>) {
            fext_load(H_A, &coeffs(a));
            fext_load(H_B, &coeffs(b));
            let cur = acc_handle(acc.region, acc.buf);
            let alt = acc_handle(acc.region, acc.buf ^ 1);
            // alt = a * b + cur   (out=alt != c=cur, satisfies the guard). The
            // accumulator stays resident: no LOAD/STORE of `cur` per step.
            fext_fma(H_A, H_B, cur, alt);
            acc.buf ^= 1;
        }

        #[inline(always)]
        fn prod_acc_finish(acc: super::GuestAcc) -> FieldElement<Fp3> {
            use core::sync::atomic::Ordering::Relaxed;
            let cur = acc_handle(acc.region, acc.buf);
            let out = from_coeffs(fext_store(cur));
            ACC_DEPTH.fetch_sub(1, Relaxed);
            out
        }

        fn geometric_powers(
            base: &FieldElement<Fp3>,
            count: usize,
        ) -> alloc::vec::Vec<FieldElement<Fp3>> {
            let mut out = alloc::vec::Vec::with_capacity(count);
            if count == 0 {
                return out;
            }
            // `base` stays resident for the whole sequence (loaded + canonicalized
            // once, not once per element as the `*` operator would). The running
            // power ping-pongs between H_GEO_A/H_GEO_B, so each step is a single
            // FMA (out=alt != a=cur, != b=mul, != c=H_ZERO — pairwise distinct)
            // plus the STORE that materializes the emitted power. No per-element
            // LOAD of either operand and no re-canonicalization of `base`.
            fext_load(H_GEO_MUL, &coeffs(base));
            fext_load(H_GEO_A, &[1, 0, 0]); // base^0 = one
            let mut cur = H_GEO_A;
            let mut alt = H_GEO_B;
            for i in 0..count {
                out.push(from_coeffs(fext_store(cur)));
                if i + 1 < count {
                    fext_fma(cur, H_GEO_MUL, H_ZERO, alt);
                    core::mem::swap(&mut cur, &mut alt);
                }
            }
            out
        }
    }
}

/// Guest resident-accumulator state: the ping-pong `region` this accumulator
/// owns (allocated off the `ACC_DEPTH` stack) and which of its two double-buffer
/// handles currently holds the running value. (Fields stay private; only this
/// module's guest impl constructs and reads them.)
#[cfg(all(target_arch = "riscv64", feature = "fext-accel"))]
pub struct GuestAcc {
    region: u32,
    buf: u8,
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

    /// `base_mul(base, ext)` must equal the asymmetric subfield product
    /// `base * ext` the verifier used before. Exercises the software default
    /// (also the reference the guest FEXT_BASE_MUL chip is validated against).
    #[test]
    fn base_mul_matches_subfield_product() {
        use math::field::goldilocks::GoldilocksField;
        type Gl = FieldElement<GoldilocksField>;
        for (base, ext) in [
            (3u64, [4u64, 5, 6]),
            (0, [7, 8, 9]),
            (1, [11, 22, 33]),
            (123456789, [987654321, 5, 0]),
        ] {
            let base = Gl::from(base);
            let ext = e(ext);
            assert_eq!(Fp3F::base_mul(&base, &ext), &base * &ext);
        }
    }

    /// `geometric_powers(base, n)` must equal `[1, base, base², …, base^(n-1)]`,
    /// the running-product sequence it replaces. Exercises the software default
    /// (the guest resident-field-storage impl is covered by the recursion
    /// prove/verify E2E); also checks the `n == 0` / `n == 1` edges.
    #[test]
    fn geometric_powers_match_running_product() {
        for base in [e([1, 0, 0]), e([2, 3, 4]), e([0, 0, 0]), e([7, 0, 11])] {
            for n in [0usize, 1, 2, 5, 33] {
                let got = Fp3F::geometric_powers(&base, n);
                assert_eq!(got.len(), n);
                let mut expected = alloc::vec::Vec::with_capacity(n);
                let mut cur = Fp3::one();
                for _ in 0..n {
                    expected.push(cur);
                    cur = &cur * &base;
                }
                assert_eq!(got, expected);
            }
        }
    }
}
