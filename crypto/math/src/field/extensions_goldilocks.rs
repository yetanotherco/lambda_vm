//! Quadratic and cubic extensions of the Goldilocks field.
//!
//! These extensions use the optimized Goldilocks implementation (no Montgomery form).
//!

use crate::field::{
    element::FieldElement,
    errors::FieldError,
    goldilocks::{GOLDILOCKS_PRIME, GoldilocksField, dot_product_2, dot_product_3, mul_by_7_raw},
    traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
};
use crate::traits::{AsBytes, ByteConversion};

impl ByteConversion for [FpE; 2] {
    const BYTE_LEN: usize = 16;

    type FixedBytes = [u8; 16];

    fn to_bytes_be(&self) -> [u8; 16] {
        unimplemented!()
    }

    fn to_bytes_le(&self) -> [u8; 16] {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl ByteConversion for [FpE; 3] {
    const BYTE_LEN: usize = 24;

    type FixedBytes = [u8; 24];

    fn to_bytes_be(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        // Byte order preserved from the previous Vec impl: components in
        // reverse index order (self[2], self[1], self[0]).
        bytes[0..8].copy_from_slice(&self[2].to_bytes_be());
        bytes[8..16].copy_from_slice(&self[1].to_bytes_be());
        bytes[16..24].copy_from_slice(&self[0].to_bytes_be());
        bytes
    }

    fn to_bytes_le(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[0..8].copy_from_slice(&self[0].to_bytes_le());
        bytes[8..16].copy_from_slice(&self[1].to_bytes_le());
        bytes[16..24].copy_from_slice(&self[2].to_bytes_le());
        bytes
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const N: usize = 8;
        if bytes.len() < N * 3 {
            return Err(crate::errors::ByteConversionError::FromBEBytesError);
        }
        let x2 = FieldElement::from_bytes_be(&bytes[0..N])?;
        let x1 = FieldElement::from_bytes_be(&bytes[N..N * 2])?;
        let x0 = FieldElement::from_bytes_be(&bytes[N * 2..N * 3])?;
        Ok([x0, x1, x2])
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const N: usize = 8;
        if bytes.len() < N * 3 {
            return Err(crate::errors::ByteConversionError::FromLEBytesError);
        }
        let x0 = FieldElement::from_bytes_le(&bytes[0..N])?;
        let x1 = FieldElement::from_bytes_le(&bytes[N..N * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[N * 2..N * 3])?;
        Ok([x0, x1, x2])
    }
}

// =====================================================
// QUADRATIC EXTENSION (Fp2)
// =====================================================
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// This means Fp2 = Fp[x] / (x^2 - 7)
// Elements are represented as a0 + a1*w where w^2 = 7

pub(crate) type FpE = FieldElement<GoldilocksField>;

/// Degree 2 extension field of Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree2GoldilocksExtensionField;

impl IsField for Degree2GoldilocksExtensionField {
    type BaseType = [FpE; 2];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Multiplication using fused dot products for fewer reductions.
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
    ///
    /// Uses dot_product_2 to compute each output component with a single
    /// reduce128 instead of separate mul + reduce per product.
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (a0, a1) = (*a[0].value(), *a[1].value());
        let (b0, b1) = (*b[0].value(), *b[1].value());
        let b1_7 = mul_by_7_raw(b1);

        // c0 = a0*b0 + a1*(7*b1)
        let c0 = dot_product_2(a0, b0, a1, b1_7);
        // c1 = a0*b1 + a1*b0
        let c1 = dot_product_2(a0, b1, a1, b0);

        [FpE::from_raw(c0), FpE::from_raw(c1)]
    }

    /// Squaring using fused dot product for the first component.
    /// (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let (a0, a1) = (*a[0].value(), *a[1].value());
        let a1_7 = mul_by_7_raw(a1);

        // c0 = a0*a0 + a1*(7*a1) via single-reduction dot product
        let c0 = dot_product_2(a0, a0, a1, a1_7);
        // c1 = 2 * a0 * a1
        let c1 = <GoldilocksField as IsField>::mul(&a0, &a1);
        let c1 = GoldilocksField::double(&c1);

        [FpE::from_raw(c0), FpE::from_raw(c1)]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component-wise negation of `a`
    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    /// Returns the multiplicative inverse of `a`:
    /// (a0 + a1*w)^-1 = (a0 - a1*w) / (a0^2 - W*a1^2)
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let w_a1_sq = mul_by_7(&a1_sq);
        let norm = a0_sq - w_a1_sq;
        let norm_inv = norm.inv()?;
        Ok([a[0] * norm_inv, -a[1] * norm_inv])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double()]
    }
}

impl IsSubFieldOf<Degree2GoldilocksExtensionField> for GoldilocksField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> Result<<Degree2GoldilocksExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree2GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree2GoldilocksExtensionField>>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the quadratic extension of native Goldilocks
pub type Fp2E = FieldElement<Degree2GoldilocksExtensionField>;

impl Fp2E {
    /// Returns the conjugate of self: conjugate(a0 + a1*w) = a0 - a1*w
    pub fn conjugate(&self) -> Self {
        Self::new([self.value()[0], -self.value()[1]])
    }

    /// Create a field element from an i64.
    /// Negative values are converted to their field equivalents: -x becomes p - x.
    pub fn from_i64(value: i64) -> Self {
        Self::from(value)
    }
}

// =====================================================
// CUBIC EXTENSION (Fp3)
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// This means Fp3 = Fp[x] / (x^3 - 2)
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2

/// Degree 3 extension field of Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree3GoldilocksExtensionField;

impl IsField for Degree3GoldilocksExtensionField {
    type BaseType = [FpE; 3];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    /// Fp3-Fp3 dot product: `acc += lhs[i] × rhs[i]` for all i via FP3_DOT ecall on riscv64.
    fn dot(
        acc: &mut Self::BaseType,
        lhs: &[FieldElement<Degree3GoldilocksExtensionField>],
        rhs: &[FieldElement<Degree3GoldilocksExtensionField>],
    ) {
        debug_assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        #[cfg(target_arch = "riscv64")]
        {
            const FP3_DOT_SYSCALL: u64 = u64::MAX - 6;
            let acc_ptr = acc.as_mut_ptr() as *mut u64;
            let lhs_ptr = lhs.as_ptr() as *const u64;
            let rhs_ptr = rhs.as_ptr() as *const u64;
            unsafe {
                core::arch::asm!(
                    "ecall",
                    in("a0") acc_ptr,
                    in("a1") lhs_ptr,
                    in("a2") rhs_ptr,
                    in("a3") n,
                    in("a7") FP3_DOT_SYSCALL,
                );
                core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            }
        }
        #[cfg(not(target_arch = "riscv64"))]
        {
            for i in 0..n {
                let l = lhs[i].value();
                let r = rhs[i].value();
                let prod = <Self as IsField>::mul(l, r);
                *acc = <Self as IsField>::add(acc, &prod);
            }
        }
    }

    /// Fused multiply-add: `acc += a × b` using the Fp3Fma ecall on riscv64 — one ecall
    /// instead of Fp3Mul ecall + 3 Goldilocks adds, saving ~12 instructions per call.
    #[inline(always)]
    fn fma(acc: &mut Self::BaseType, a: &Self::BaseType, b: &Self::BaseType) {
        #[cfg(target_arch = "riscv64")]
        {
            const FP3_FMA_SYSCALL: u64 = u64::MAX - 3;
            let a_raw: [u64; 3] = [*a[0].value(), *a[1].value(), *a[2].value()];
            let b_raw: [u64; 3] = [*b[0].value(), *b[1].value(), *b[2].value()];
            // acc is a &mut [FpE; 3] = &mut [FieldElement<GoldilocksField>; 3].
            // FieldElement<GoldilocksField> = { value: u64 }, so [FpE; 3] = [u64; 3] in memory.
            // Cast directly to *mut u64 for the ecall — the executor reads acc[0..2],
            // computes acc += a×b, and writes the result back in place.
            let acc_ptr = acc.as_mut_ptr() as *mut u64;
            unsafe {
                core::arch::asm!(
                    "ecall",
                    in("a0") acc_ptr,
                    in("a1") a_raw.as_ptr(),
                    in("a2") b_raw.as_ptr(),
                    in("a7") FP3_FMA_SYSCALL,
                );
                core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            }
        }
        #[cfg(not(target_arch = "riscv64"))]
        {
            *acc = <Self as IsField>::add(acc, &<Self as IsField>::mul(a, b));
        }
    }

    /// Multiplication using schoolbook with fused dot products.
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
    ///
    /// Expanding and applying w^3 = 2:
    ///   c0 = a0*b0 + 2*(a1*b2 + a2*b1)
    ///   c1 = a0*b1 + a1*b0 + 2*a2*b2
    ///   c2 = a0*b2 + a1*b1 + a2*b0
    ///
    /// Each component is computed as a single dot_product_3 (9 raw muls,
    /// 3 reduce128 calls) instead of Karatsuba (6 muls, 6 reduce128 + many
    /// add/sub). The reduction savings outweigh the extra multiplications.
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        #[cfg(target_arch = "riscv64")]
        {
            // Route through the lambda-vm Fp3Mul precompile syscall.
            // ABI: a7=FP3_MUL_SYSCALL_NUMBER, a0=result_ptr, a1=lhs_ptr, a2=rhs_ptr
            // Each pointer references a [u64; 3] (8-byte aligned).
            const FP3_MUL_SYSCALL: u64 = u64::MAX - 2;
            let mut result = [0u64; 3];
            let lhs: [u64; 3] = [*a[0].value(), *a[1].value(), *a[2].value()];
            let rhs: [u64; 3] = [*b[0].value(), *b[1].value(), *b[2].value()];
            unsafe {
                // The ecall writes the 3-limb product through `a0`. We must pass the
                // buffers as real pointer operands (NOT `ptr as u64`): casting to an
                // integer strips provenance, so LLVM concludes the `result` alloca never
                // escapes and is free to hoist the reads of `result[..]` to *before* the
                // ecall — yielding the stale zero-initialized values. Passing pointer
                // operands keeps the addresses escaped; dropping `options(nostack)` keeps
                // the default memory clobber so the ecall is modeled as writing memory.
                core::arch::asm!(
                    "ecall",
                    in("a0") result.as_mut_ptr(),
                    in("a1") lhs.as_ptr(),
                    in("a2") rhs.as_ptr(),
                    in("a7") FP3_MUL_SYSCALL,
                );
                // Belt-and-suspenders barrier: forbid the compiler from reordering the
                // result reads across the ecall even if the clobber model is relaxed.
                core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            }
            [
                FpE::from_raw(result[0]),
                FpE::from_raw(result[1]),
                FpE::from_raw(result[2]),
            ]
        }
        #[cfg(not(target_arch = "riscv64"))]
        {
            let (a0, a1, a2) = (*a[0].value(), *a[1].value(), *a[2].value());
            let (b0, b1, b2) = (*b[0].value(), *b[1].value(), *b[2].value());

            // Precompute 2*b1 and 2*b2 for the w^3 = 2 reduction
            let b1_2 = GoldilocksField::double(&b1);
            let b2_2 = GoldilocksField::double(&b2);

            // c0 = a0*b0 + a1*(2*b2) + a2*(2*b1)
            let c0 = dot_product_3(a0, b0, a1, b2_2, a2, b1_2);
            // c1 = a0*b1 + a1*b0 + a2*(2*b2)
            let c1 = dot_product_3(a0, b1, a1, b0, a2, b2_2);
            // c2 = a0*b2 + a1*b1 + a2*b0
            let c2 = dot_product_3(a0, b2, a1, b1, a2, b0);

            [FpE::from_raw(c0), FpE::from_raw(c1), FpE::from_raw(c2)]
        }
    }

    /// Squaring using fused dot products.
    /// (a0 + a1*w + a2*w^2)^2 mod (w^3 - 2):
    ///   c0 = a0^2 + 4*a1*a2
    ///   c1 = 2*a0*a1 + 2*a2^2
    ///   c2 = 2*a0*a2 + a1^2
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let (a0, a1, a2) = (*a[0].value(), *a[1].value(), *a[2].value());

        let a0_2 = GoldilocksField::double(&a0);
        let a2_4 = GoldilocksField::double(&GoldilocksField::double(&a2));

        // c0 = a0*a0 + a1*(4*a2) — using dot_product_2
        let c0 = dot_product_2(a0, a0, a1, a2_4);
        // c1 = (2*a0)*a1 + (2*a2)*a2 — using dot_product_2
        let a2_2 = GoldilocksField::double(&a2);
        let c1 = dot_product_2(a0_2, a1, a2_2, a2);
        // c2 = a1*a1 + (2*a0)*a2 — using dot_product_2
        let c2 = dot_product_2(a1, a1, a0_2, a2);

        [FpE::from_raw(c0), FpE::from_raw(c1), FpE::from_raw(c2)]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Returns the component-wise negation of `a`
    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1], -a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a2_sq = a[2].square();

        // Compute the norm: N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let a0_cubed = a0_sq * a[0];
        let a1_cubed = a1_sq * a[1];
        let a2_cubed = a2_sq * a[2];
        let a0a1a2 = a[0] * a[1] * a[2];

        // N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let norm = a0_cubed + a1_cubed.double() + a2_cubed.double().double()
            - (a0a1a2.double() + a0a1a2).double();

        let norm_inv = norm.inv()?;

        // inv[0] = (a0^2 - 2*a1*a2) / N
        // inv[1] = (2*a2^2 - a0*a1) / N
        // inv[2] = (a1^2 - a0*a2) / N
        let a1a2 = a[1] * a[2];
        let a0a1 = a[0] * a[1];
        let a0a2 = a[0] * a[2];

        Ok([
            (a0_sq - a1a2.double()) * norm_inv,
            (a2_sq.double() - a0a1) * norm_inv,
            (a1_sq - a0a2) * norm_inv,
        ])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero(), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double(), a[2].double()]
    }
}

impl IsSubFieldOf<Degree3GoldilocksExtensionField> for GoldilocksField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::mul(a, b[2].value()));
        [c0, c1, c2]
    }

    /// Scalar Fp3 FMA: `acc += scalar × b` via Fp3ScalarFma ecall on riscv64.
    fn scalar_fma(
        acc: &mut <Degree3GoldilocksExtensionField as IsField>::BaseType,
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) {
        goldilocks_scalar_fp3_fma(acc, a, b);
    }

    /// Scalar Fp3 dot product: one FP3_SCALAR_DOT ecall for all n elements.
    fn scalar_dot(
        acc: &mut <Degree3GoldilocksExtensionField as IsField>::BaseType,
        scalars: &[FieldElement<GoldilocksField>],
        fp3: &[FieldElement<Degree3GoldilocksExtensionField>],
    ) {
        // SAFETY: [FpE; 3] has the same memory layout as [u64; 3] since FpE = {value: u64}.
        let acc_arr = unsafe { &mut *(acc as *mut _ as *mut [FpE; 3]) };
        goldilocks_scalar_fp3_dot(acc_arr, scalars, fp3);
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1], b[2]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> Result<<Degree3GoldilocksExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree3GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree3GoldilocksExtensionField>>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::neg(b[2].value()));
        [c0, c1, c2]
    }

    fn embed(a: Self::BaseType) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero(), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the cubic extension of native Goldilocks
pub type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

impl Fp3E {
    /// Create a field element from an i64.
    /// Negative values are converted to their field equivalents: -x becomes p - x.
    pub fn from_i64(value: i64) -> Self {
        Self::from(value)
    }
}

// =====================================================
// TRAIT IMPLEMENTATIONS FOR PROVER/VERIFIER
// =====================================================

impl ByteConversion for FieldElement<Degree3GoldilocksExtensionField> {
    const BYTE_LEN: usize = 24;

    type FixedBytes = [u8; 24];

    #[inline(always)]
    fn write_bytes_be(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= 24);
        let components = self.value();
        components[0].write_bytes_be(&mut buf[0..8]);
        components[1].write_bytes_be(&mut buf[8..16]);
        components[2].write_bytes_be(&mut buf[16..24]);
    }

    #[inline(always)]
    fn to_bytes_be(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        let components = self.value();
        bytes[0..8].copy_from_slice(&components[0].to_bytes_be());
        bytes[8..16].copy_from_slice(&components[1].to_bytes_be());
        bytes[16..24].copy_from_slice(&components[2].to_bytes_be());
        bytes
    }

    #[inline(always)]
    fn to_bytes_le(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        let components = self.value();
        bytes[0..8].copy_from_slice(&components[0].to_bytes_le());
        bytes[8..16].copy_from_slice(&components[1].to_bytes_le());
        bytes[16..24].copy_from_slice(&components[2].to_bytes_le());
        bytes
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 8;
        if bytes.len() < BYTES_PER_FIELD * 3 {
            return Err(crate::errors::ByteConversionError::FromBEBytesError);
        }
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;

        Ok(Self::new([x0, x1, x2]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 8;
        if bytes.len() < BYTES_PER_FIELD * 3 {
            return Err(crate::errors::ByteConversionError::FromLEBytesError);
        }
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;

        Ok(Self::new([x0, x1, x2]))
    }
}

/// Type alias for the Goldilocks cubic extension field element.
pub type Fp3Element = FieldElement<Degree3GoldilocksExtensionField>;


/// Scalar-Fp3 dot product: `acc += scalars[0]*fp3[0] + ... + scalars[n-1]*fp3[n-1]`.
/// Issues a single ecall on riscv64 instead of n separate scalar_fma ecalls.
/// `scalars`: slice of Goldilocks field elements (1 u64 each)
/// `fp3`: slice of Fp3 field elements (3 u64 each, [FpE; 3] layout contiguous)
#[inline(always)]
pub fn goldilocks_scalar_fp3_dot(
    acc: &mut [FpE; 3],
    scalars: &[FieldElement<GoldilocksField>],
    fp3: &[FieldElement<Degree3GoldilocksExtensionField>],
) {
    debug_assert_eq!(scalars.len(), fp3.len(), "scalars and fp3 must have equal length");
    let n = scalars.len();
    #[cfg(target_arch = "riscv64")]
    {
        const FP3_SCALAR_DOT_SYSCALL: u64 = u64::MAX - 5;
        let acc_ptr = acc.as_mut_ptr() as *mut u64;
        // FieldElement<GoldilocksField> = { value: u64 } → contiguous u64 array.
        let scalars_ptr = scalars.as_ptr() as *const u64;
        // FieldElement<Degree3GoldilocksExtensionField> = { value: [FpE; 3] } = { value: [u64; 3] }
        // → contiguous [u64; 3] array (24 bytes per element).
        let fp3_ptr = fp3.as_ptr() as *const u64;
        unsafe {
            core::arch::asm!(
                "ecall",
                in("a0") acc_ptr,
                in("a1") scalars_ptr,
                in("a2") fp3_ptr,
                in("a3") n,
                in("a7") FP3_SCALAR_DOT_SYSCALL,
            );
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        }
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        let add = <GoldilocksField as IsField>::add;
        let mul = <GoldilocksField as IsField>::mul;
        for i in 0..n {
            let s = scalars[i].value();
            let c = fp3[i].value();
            acc[0] = FpE::from_raw(add(&mul(s, c[0].value()), acc[0].value()));
            acc[1] = FpE::from_raw(add(&mul(s, c[1].value()), acc[1].value()));
            acc[2] = FpE::from_raw(add(&mul(s, c[2].value()), acc[2].value()));
        }
    }
}

/// Standalone scalar-Fp3 FMA function for use by the `IsSubFieldOf` impl.
/// `acc += scalar * b`: 3 Goldilocks muls via Fp3ScalarFma ecall on riscv64.
#[inline(always)]
pub(crate) fn goldilocks_scalar_fp3_fma(
    acc: &mut [FpE; 3],
    scalar: &u64,
    b: &[FpE; 3],
) {
    #[cfg(target_arch = "riscv64")]
    {
        const FP3_SCALAR_FMA_SYSCALL: u64 = u64::MAX - 4;
        let b_raw: [u64; 3] = [*b[0].value(), *b[1].value(), *b[2].value()];
        let acc_ptr = acc.as_mut_ptr() as *mut u64;
        unsafe {
            core::arch::asm!(
                "ecall",
                in("a0") acc_ptr,
                in("a1") scalar as *const u64,
                in("a2") b_raw.as_ptr(),
                in("a7") FP3_SCALAR_FMA_SYSCALL,
            );
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        }
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        let mul = <GoldilocksField as IsField>::mul;
        let add = <GoldilocksField as IsField>::add;
        let c0 = add(&mul(scalar, b[0].value()), acc[0].value());
        let c1 = add(&mul(scalar, b[1].value()), acc[1].value());
        let c2 = add(&mul(scalar, b[2].value()), acc[2].value());
        acc[0] = FpE::from_raw(c0);
        acc[1] = FpE::from_raw(c1);
        acc[2] = FpE::from_raw(c2);
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<Degree3GoldilocksExtensionField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_bytes_be().to_vec()
    }
}

impl HasDefaultTranscript for Degree3GoldilocksExtensionField {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        // Draw all three coefficients' entropy (3 × 8 = 24 bytes) in one `fill`,
        // then slice. `rng.fill` consumes the RNG byte stream sequentially, so for
        // the common case — all three big-endian limbs already below the prime —
        // one `fill(&mut [u8; 24])` reads byte-for-byte the same stream as three
        // `fill(&mut [u8; 8])`, producing the IDENTICAL value while issuing one RNG
        // call (one underlying ChaCha block) instead of three. This is the only
        // path that ever executes in practice: a Goldilocks limb is rejected only
        // when it lands in [p, 2^64), i.e. with probability (2^32 − 1)/2^64 ≈
        // 1-in-4-billion per limb.
        //
        // SOUNDNESS NOTE: on the (astronomically rare) rejection of any limb, the
        // value produced differs from the historical three-independent-`fill(8)`
        // reference, because the batch has already consumed the later limbs' bytes
        // before the rejected limb is re-drawn. This is safe because both prover
        // and verifier run this exact function, so they always agree; it is not
        // backward-compatible with proofs generated by the old code that happened
        // to hit a rejection (none are known to exist, and the probability of one
        // is negligible). The rejection re-draw below is deterministic and shared.
        let mut bytes = [0u8; 24];
        rng.fill(&mut bytes);

        let mut coeffs = [FpE::zero(), FpE::zero(), FpE::zero()];
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            let mut int_sample = u64::from_be_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
            while int_sample >= GOLDILOCKS_PRIME {
                let mut sample = [0u8; 8];
                rng.fill(&mut sample);
                int_sample = u64::from_be_bytes(sample);
            }
            *coeff = FpE::from(int_sample);
        }
        FieldElement::<Self>::new(coeffs)
    }

    fn sample_field_element_from_squeeze(
        mut squeeze: impl FnMut() -> [u8; 32],
    ) -> FieldElement<Self> {
        // Three limbs = 24 bytes, which fit in a single 32-byte squeeze block, so
        // the common (no-rejection) path costs exactly one squeeze. Each limb takes
        // its big-endian 8-byte slice of that block; a limb that lands out of range
        // (~1-in-4-billion) is re-drawn from a fresh squeeze block (first 8 bytes),
        // which is deterministic and identical on prover and verifier.
        let block = squeeze();
        let mut coeffs = [FpE::zero(), FpE::zero(), FpE::zero()];
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            let mut int_sample = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
            while int_sample >= GOLDILOCKS_PRIME {
                let resampled = squeeze();
                int_sample = u64::from_be_bytes(resampled[..8].try_into().unwrap());
            }
            *coeff = FpE::from(int_sample);
        }
        FieldElement::<Self>::new(coeffs)
    }
}

// =====================================================
// HELPER FUNCTIONS
// =====================================================

/// Multiply a field element by 7 (the quadratic non-residue).
/// Wraps the raw u64 implementation for use with FieldElement types.
#[inline(always)]
fn mul_by_7(a: &FpE) -> FpE {
    FpE::from_raw(mul_by_7_raw(*a.value()))
}
