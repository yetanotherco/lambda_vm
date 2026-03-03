# SIMD Goldilocks Implementation Plan — Phase 1: PackedField Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement PackedField trait and AVX2/AVX-512/NEON packed Goldilocks arithmetic so that higher-level algorithms (FFT, constraints, Merkle) can use SIMD transparently.

**Architecture:** A `PackedField` trait abstracts over scalar vs SIMD field elements. Each platform gets a concrete packed type (`PackedGoldilocksAVX2` WIDTH=4, `PackedGoldilocksAVX512` WIDTH=8, `PackedGoldilocksNeon` WIDTH=2). The `IsField` trait gains a `Packing` associated type that resolves at compile time via `#[cfg]`. A `PackedFp3<P>` generic struct provides packed cubic extension arithmetic by delegating to packed base field ops.

**Tech Stack:** Rust, `core::arch::x86_64` (AVX2/AVX-512 intrinsics), `core::arch::aarch64` (NEON intrinsics), `#[cfg(target_arch)]` compile-time dispatch.

**Design doc:** `docs/plans/2026-02-26-simd-goldilocks-design.md`

---

## Task 1: Create PackedField trait

**Files:**
- Create: `crypto/math/src/field/packed.rs`
- Modify: `crypto/math/src/field/mod.rs`

**Context:** This trait is the foundation for all SIMD work. It abstracts a fixed-width vector of field elements. The `unsafe` marker indicates that implementors must guarantee that `pack_slice` (pointer cast from scalar slice to packed slice) is memory-safe, which requires `#[repr(transparent)]` on the packed type.

**Step 1: Create the trait file**

Create `crypto/math/src/field/packed.rs`:

```rust
//! PackedField trait for SIMD-vectorized field arithmetic.
//!
//! A PackedField holds WIDTH scalar field elements in a single value
//! (typically a SIMD register). All arithmetic operations act lane-wise.

use super::element::FieldElement;
use super::traits::IsField;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A fixed-width vector of field elements that supports lane-wise arithmetic.
///
/// # Safety
///
/// Implementors must be `#[repr(transparent)]` wrappers around `[Self::Scalar; WIDTH]`
/// so that `pack_slice` (pointer cast) is safe. The memory layout must match exactly.
pub unsafe trait PackedField:
    Copy
    + Send
    + Sync
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The scalar field type held in each lane.
    type Scalar: IsField;

    /// Number of scalar elements per packed value.
    /// Must be a power of 2: 1 (scalar), 2 (NEON), 4 (AVX2), 8 (AVX-512).
    const WIDTH: usize;

    /// Construct a packed value by calling `f(lane_index)` for each lane.
    fn from_fn(f: impl FnMut(usize) -> FieldElement<Self::Scalar>) -> Self;

    /// Construct a packed value from a slice of exactly WIDTH elements.
    /// Panics if `slice.len() < WIDTH`.
    fn from_slice(slice: &[FieldElement<Self::Scalar>]) -> Self;

    /// View the packed value as a slice of WIDTH scalar elements.
    fn as_slice(&self) -> &[FieldElement<Self::Scalar>];

    /// View the packed value as a mutable slice of WIDTH scalar elements.
    fn as_slice_mut(&mut self) -> &mut [FieldElement<Self::Scalar>];

    /// The packed zero (all lanes zero).
    fn zero() -> Self;

    /// The packed one (all lanes one).
    fn ones() -> Self;

    /// Broadcast a single scalar to all lanes.
    fn broadcast(value: FieldElement<Self::Scalar>) -> Self;

    /// Square each lane.
    fn square(&self) -> Self {
        *self * *self
    }

    /// Double each lane.
    fn double(&self) -> Self {
        *self + *self
    }

    /// Reinterpret a scalar slice as a packed slice. Zero-cost pointer cast.
    ///
    /// Returns `(packed, remainder)` where remainder has `< WIDTH` elements.
    /// The packed slice has `buf.len() / WIDTH` elements.
    fn pack_slice_with_suffix(
        buf: &[FieldElement<Self::Scalar>],
    ) -> (&[Self], &[FieldElement<Self::Scalar>]) {
        let n_packed = buf.len() / Self::WIDTH;
        let packed_len = n_packed * Self::WIDTH;
        let (head, tail) = buf.split_at(packed_len);
        // SAFETY: Self is repr(transparent) over [Scalar; WIDTH],
        // so the pointer cast is valid. Alignment is guaranteed because
        // FieldElement<F> has the same alignment as F::BaseType (u64),
        // and packed types require no stricter alignment than their elements.
        let packed = unsafe {
            core::slice::from_raw_parts(head.as_ptr() as *const Self, n_packed)
        };
        (packed, tail)
    }

    /// Mutable version of `pack_slice_with_suffix`.
    fn pack_slice_with_suffix_mut(
        buf: &mut [FieldElement<Self::Scalar>],
    ) -> (&mut [Self], &mut [FieldElement<Self::Scalar>]) {
        let n_packed = buf.len() / Self::WIDTH;
        let packed_len = n_packed * Self::WIDTH;
        let (head, tail) = buf.split_at_mut(packed_len);
        let packed = unsafe {
            core::slice::from_raw_parts_mut(head.as_mut_ptr() as *mut Self, n_packed)
        };
        (packed, tail)
    }

    /// Block interleave for in-register FFT transposes.
    ///
    /// Given `self = [a0, a1, a2, a3]` and `other = [b0, b1, b2, b3]` with `block_len=1`:
    /// Returns `([a0, b0, a2, b2], [a1, b1, a3, b3])`.
    ///
    /// With `block_len=2`:
    /// Returns `([a0, a1, b0, b1], [a2, a3, b2, b3])`.
    ///
    /// `block_len` must be a power of 2 and `<= WIDTH`.
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self);
}

/// Scalar "packed" field — WIDTH=1 fallback for platforms without SIMD.
/// This allows all generic packed code to compile and run correctly everywhere.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct ScalarPacked<F: IsField>(pub FieldElement<F>);

// Safety: ScalarPacked is repr(transparent) over FieldElement<F>,
// which has the same layout as [FieldElement<F>; 1].
unsafe impl<F: IsField> PackedField for ScalarPacked<F>
where
    FieldElement<F>: Copy + Send + Sync,
{
    type Scalar = F;
    const WIDTH: usize = 1;

    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> FieldElement<F>) -> Self {
        Self(f(0))
    }

    #[inline(always)]
    fn from_slice(slice: &[FieldElement<F>]) -> Self {
        Self(slice[0])
    }

    #[inline(always)]
    fn as_slice(&self) -> &[FieldElement<F>] {
        core::slice::from_ref(&self.0)
    }

    #[inline(always)]
    fn as_slice_mut(&mut self) -> &mut [FieldElement<F>] {
        core::slice::from_mut(&mut self.0)
    }

    #[inline(always)]
    fn zero() -> Self {
        Self(FieldElement::zero())
    }

    #[inline(always)]
    fn ones() -> Self {
        Self(FieldElement::one())
    }

    #[inline(always)]
    fn broadcast(value: FieldElement<F>) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn interleave(&self, other: Self, _block_len: usize) -> (Self, Self) {
        (*self, other)
    }
}

impl<F: IsField> Add for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<F: IsField> Sub for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<F: IsField> Mul for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl<F: IsField> Neg for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<F: IsField> AddAssign for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

impl<F: IsField> SubAssign for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

impl<F: IsField> MulAssign for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}
```

**Step 2: Register the module**

Modify `crypto/math/src/field/mod.rs` to add:

```rust
/// PackedField trait for SIMD-vectorized field arithmetic.
pub mod packed;
```

**Step 3: Build and verify**

Run: `cargo build -p math --features parallel`
Expected: compiles cleanly.

**Step 4: Commit**

```bash
git add crypto/math/src/field/packed.rs crypto/math/src/field/mod.rs
git commit -m "Add PackedField trait and ScalarPacked fallback"
```

---

## Task 2: PackedGoldilocksAVX2 — struct, add, sub, neg

**Files:**
- Create: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/mod.rs`
- Create: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx2.rs`
- Create: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/scalar.rs`
- Modify: `crypto/math/src/field/fields/fft_friendly/mod.rs`

**Context:** AVX2 provides 256-bit registers (`__m256i`). For Goldilocks (64-bit field), this gives WIDTH=4. AVX2 lacks unsigned 64-bit comparison, so we use a "shifted representation" — XOR with `2^63` to convert unsigned to signed comparison. The correction constant is EPSILON = `2^32 - 1 = 2^64 mod P`.

**Step 1: Write tests for packed add/sub/neg**

In `x86_64_avx2.rs`, add a test module at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::u64_goldilocks::{GoldilocksField, GOLDILOCKS_PRIME};

    type FE = FieldElement<GoldilocksField>;

    fn random_packed() -> PackedGoldilocksAVX2 {
        PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 1) * 0x123456789ABCDEFu64))
    }

    #[test]
    fn test_packed_add_matches_scalar() {
        let a = random_packed();
        let b = random_packed();
        let packed_sum = a + b;
        let scalar_sum = PackedGoldilocksAVX2::from_fn(|i| {
            a.as_slice()[i] + b.as_slice()[i]
        });
        assert_eq!(packed_sum.as_slice(), scalar_sum.as_slice());
    }

    #[test]
    fn test_packed_sub_matches_scalar() {
        let a = random_packed();
        let b = random_packed();
        let packed_diff = a - b;
        let scalar_diff = PackedGoldilocksAVX2::from_fn(|i| {
            a.as_slice()[i] - b.as_slice()[i]
        });
        assert_eq!(packed_diff.as_slice(), scalar_diff.as_slice());
    }

    #[test]
    fn test_packed_neg_matches_scalar() {
        let a = random_packed();
        let packed_neg = -a;
        let scalar_neg = PackedGoldilocksAVX2::from_fn(|i| -a.as_slice()[i]);
        assert_eq!(packed_neg.as_slice(), scalar_neg.as_slice());
    }

    #[test]
    fn test_packed_add_overflow() {
        // Both values near p should reduce correctly
        let a = PackedGoldilocksAVX2::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let b = PackedGoldilocksAVX2::broadcast(FE::from(2u64));
        let sum = a + b;
        assert_eq!(sum.as_slice()[0], FE::from(1u64));
    }

    #[test]
    fn test_packed_sub_underflow() {
        let a = PackedGoldilocksAVX2::broadcast(FE::from(1u64));
        let b = PackedGoldilocksAVX2::broadcast(FE::from(3u64));
        let diff = a - b;
        assert_eq!(diff.as_slice()[0], FE::from(GOLDILOCKS_PRIME - 2));
    }

    #[test]
    fn test_pack_slice_roundtrip() {
        let values: Vec<FE> = (0..8).map(|i| FE::from(i as u64 + 100)).collect();
        let (packed, suffix) = PackedGoldilocksAVX2::pack_slice_with_suffix(&values);
        assert_eq!(packed.len(), 2); // 8 / 4 = 2 packed values
        assert_eq!(suffix.len(), 0);
        assert_eq!(packed[0].as_slice(), &values[0..4]);
        assert_eq!(packed[1].as_slice(), &values[4..8]);
    }
}
```

**Step 2: Implement the AVX2 struct and add/sub/neg**

Create `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx2.rs`:

```rust
//! AVX2 packed Goldilocks field arithmetic (WIDTH=4).
//!
//! Uses 256-bit __m256i registers holding 4 × 64-bit field elements.
//! Modular arithmetic uses the shifted-representation trick for unsigned
//! comparison emulation (XOR with 2^63 converts to signed domain).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::{GoldilocksField, GOLDILOCKS_PRIME};
use crate::field::packed::PackedField;

const WIDTH: usize = 4;
const EPSILON: u64 = 0xFFFF_FFFF; // 2^32 - 1 = 2^64 mod P

/// Packed Goldilocks field element holding 4 elements in an AVX2 register.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct PackedGoldilocksAVX2(pub [FieldElement<GoldilocksField>; WIDTH]);

impl Default for PackedGoldilocksAVX2 {
    fn default() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }
}

impl PartialEq for PackedGoldilocksAVX2 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedGoldilocksAVX2 {}

impl PackedGoldilocksAVX2 {
    #[inline(always)]
    fn to_vector(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    fn from_vector(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }
}

// ---- Constants ----
// All computed at compile time via const transmute.

const SIGN_BIT: __m256i = unsafe { transmute([i64::MIN; WIDTH]) };
const FIELD_ORDER: __m256i = unsafe { transmute([GOLDILOCKS_PRIME; WIDTH]) };
const SHIFTED_FIELD_ORDER: __m256i =
    unsafe { transmute([GOLDILOCKS_PRIME ^ (i64::MIN as u64); WIDTH]) };
const EPSILON_VEC: __m256i = unsafe { transmute([EPSILON; WIDTH]) };

// ---- Shifted-representation helpers ----

/// XOR with 2^63 to convert unsigned → signed comparison domain.
#[inline(always)]
unsafe fn shift(x: __m256i) -> __m256i {
    _mm256_xor_si256(x, SIGN_BIT)
}

/// Canonicalize a shifted value to [0, P) in the shifted domain.
/// If x >= P (shifted), subtract P by adding EPSILON.
#[inline(always)]
unsafe fn canonicalize_s(x_s: __m256i) -> __m256i {
    // mask = -1 (all bits) if x < P (in shifted domain), 0 otherwise
    let mask = _mm256_cmpgt_epi64(SHIFTED_FIELD_ORDER, x_s);
    // wrapback = EPSILON if x >= P, else 0
    let wrapback = _mm256_andnot_si256(mask, EPSILON_VEC);
    // x + EPSILON = x - P (mod 2^64), since P + EPSILON = 2^64
    _mm256_add_epi64(x_s, wrapback)
}

/// Add x (non-shifted) + y_s (shifted, canonical) → result in shifted domain.
/// Assumes no double overflow (valid when y is canonical, i.e., y < P).
#[inline(always)]
unsafe fn add_no_double_overflow_s(x: __m256i, y_s: __m256i) -> __m256i {
    let res_s = _mm256_add_epi64(x, y_s);
    // Overflow if res_s < y_s (unsigned), but in shifted domain
    // this is equivalent to signed comparison.
    let mask = _mm256_cmpgt_epi64(y_s, res_s);
    // On overflow, add EPSILON = subtract P
    let correction = _mm256_srli_epi64::<32>(mask); // 0x00000000FFFFFFFF = EPSILON
    _mm256_add_epi64(res_s, correction)
}

/// Packed modular addition: (a + b) mod P
#[inline(always)]
unsafe fn add_avx2(a: __m256i, b: __m256i) -> __m256i {
    let b_s = canonicalize_s(shift(b));
    let res_s = add_no_double_overflow_s(a, b_s);
    shift(res_s)
}

/// Packed modular subtraction: (a - b) mod P
#[inline(always)]
unsafe fn sub_avx2(a: __m256i, b: __m256i) -> __m256i {
    let a_s = shift(a);
    let b_s = canonicalize_s(shift(b));
    // If b > a (unsigned), underflow → add P (subtract EPSILON)
    let mask = _mm256_cmpgt_epi64(b_s, a_s);
    let correction = _mm256_srli_epi64::<32>(mask); // EPSILON
    let res = _mm256_sub_epi64(a_s, b_s);
    shift(_mm256_sub_epi64(res, correction))
}

/// Packed modular negation: P - a (or 0 if a == 0)
#[inline(always)]
unsafe fn neg_avx2(a: __m256i) -> __m256i {
    let a_canon = shift(canonicalize_s(shift(a)));
    sub_avx2(FIELD_ORDER, a_canon)
}

// ---- Operator impls ----

impl Add for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { add_avx2(self.to_vector(), rhs.to_vector()) })
    }
}

impl Sub for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { sub_avx2(self.to_vector(), rhs.to_vector()) })
    }
}

impl Neg for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_vector(unsafe { neg_avx2(self.to_vector()) })
    }
}

// Mul is a placeholder — implemented in Task 3
impl Mul for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Temporary scalar fallback until Task 3
        Self::from_fn(|i| self.as_slice()[i] * rhs.as_slice()[i])
    }
}

impl AddAssign for PackedGoldilocksAVX2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for PackedGoldilocksAVX2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for PackedGoldilocksAVX2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

// ---- PackedField impl ----

unsafe impl PackedField for PackedGoldilocksAVX2 {
    type Scalar = GoldilocksField;
    const WIDTH: usize = WIDTH;

    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> FieldElement<GoldilocksField>) -> Self {
        Self([f(0), f(1), f(2), f(3)])
    }

    #[inline(always)]
    fn from_slice(slice: &[FieldElement<GoldilocksField>]) -> Self {
        Self([slice[0], slice[1], slice[2], slice[3]])
    }

    #[inline(always)]
    fn as_slice(&self) -> &[FieldElement<GoldilocksField>] {
        &self.0
    }

    #[inline(always)]
    fn as_slice_mut(&mut self) -> &mut [FieldElement<GoldilocksField>] {
        &mut self.0
    }

    #[inline(always)]
    fn zero() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }

    #[inline(always)]
    fn ones() -> Self {
        Self([FieldElement::one(); WIDTH])
    }

    #[inline(always)]
    fn broadcast(value: FieldElement<GoldilocksField>) -> Self {
        Self([value; WIDTH])
    }

    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        unsafe {
            let a = self.to_vector();
            let b = other.to_vector();
            match block_len {
                1 => {
                    // [a0,a1,a2,a3] x [b0,b1,b2,b3] → [a0,b0,a2,b2], [a1,b1,a3,b3]
                    let lo = _mm256_unpacklo_epi64(a, b);
                    let hi = _mm256_unpackhi_epi64(a, b);
                    (Self::from_vector(lo), Self::from_vector(hi))
                }
                2 => {
                    // [a0,a1,a2,a3] x [b0,b1,b2,b3] → [a0,a1,b0,b1], [a2,a3,b2,b3]
                    let t = _mm256_permute2x128_si256::<0x21>(a, b);
                    let lo = _mm256_blend_epi32::<0b11110000>(a, t);
                    let hi = _mm256_blend_epi32::<0b11110000>(t, b);
                    (Self::from_vector(lo), Self::from_vector(hi))
                }
                4 => (*self, other), // WIDTH = identity
                _ => panic!("block_len must be 1, 2, or 4 for WIDTH=4"),
            }
        }
    }
}
```

**Step 3: Create the scalar fallback module**

Create `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/scalar.rs`:

```rust
//! Scalar fallback for packed Goldilocks — used on platforms without SIMD.
//! Simply re-exports ScalarPacked<GoldilocksField> from the packed module.

pub use crate::field::packed::ScalarPacked;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

pub type PackedGoldilocksScalar = ScalarPacked<GoldilocksField>;
```

**Step 4: Create the module dispatch file**

Create `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/mod.rs`:

```rust
//! Platform-dispatched packed Goldilocks field types.
//!
//! Selects the best SIMD implementation at compile time:
//! - AVX2 (x86-64): WIDTH=4
//! - AVX-512 (x86-64): WIDTH=8
//! - NEON (AArch64): WIDTH=2
//! - Scalar fallback: WIDTH=1

mod scalar;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
mod x86_64_avx2;

// TODO: Task 5 — AVX-512
// #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
// mod x86_64_avx512;

// TODO: Task 6 — NEON
// #[cfg(target_arch = "aarch64")]
// mod aarch64_neon;

// Re-export the platform-appropriate packed type as `PackedGoldilocks`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
pub use x86_64_avx2::PackedGoldilocksAVX2 as PackedGoldilocks;

// Scalar fallback for all other platforms (including when no SIMD cfg is active)
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")),
    // all(target_arch = "x86_64", target_feature = "avx512f"),
    // target_arch = "aarch64",
)))]
pub use scalar::PackedGoldilocksScalar as PackedGoldilocks;
```

**Step 5: Register the module in the fft_friendly mod**

Add to `crypto/math/src/field/fields/fft_friendly/mod.rs`:

```rust
/// Packed (SIMD) Goldilocks field types
pub mod u64_goldilocks_packed;
```

**Step 6: Build and run tests**

Run: `RUSTFLAGS="-C target-feature=+avx2" cargo test -p math --features parallel -- u64_goldilocks_packed`
Expected: all 6 tests pass (add, sub, neg, overflow, underflow, pack_slice).

Note: the `RUSTFLAGS` is needed because `#[cfg(target_feature = "avx2")]` requires the feature to be enabled at compile time. If building on a machine with AVX2, `target-cpu=native` also works.

**Step 7: Commit**

```bash
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/
git add crypto/math/src/field/fields/fft_friendly/mod.rs
git commit -m "Add PackedGoldilocksAVX2 with add, sub, neg, interleave"
```

---

## Task 3: PackedGoldilocksAVX2 — multiply and square

**Files:**
- Modify: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx2.rs`

**Context:** This is the hardest piece. AVX2 has no native 64×64→128 multiply. We decompose each 64-bit value into high/low 32-bit halves, perform four `_mm256_mul_epu32` (32×32→64), and assemble the 128-bit result. Then reduce using `2^96 ≡ -1 (mod P)` and `2^64 ≡ EPSILON (mod P)`.

**Step 1: Write tests for multiply and square**

Add to the existing test module in `x86_64_avx2.rs`:

```rust
#[test]
fn test_packed_mul_matches_scalar() {
    let a = random_packed();
    let b = random_packed();
    let packed_prod = a * b;
    let scalar_prod = PackedGoldilocksAVX2::from_fn(|i| {
        a.as_slice()[i] * b.as_slice()[i]
    });
    assert_eq!(packed_prod.as_slice(), scalar_prod.as_slice());
}

#[test]
fn test_packed_square_matches_mul() {
    let a = random_packed();
    let packed_sq = a.square();
    let packed_mul = a * a;
    assert_eq!(packed_sq.as_slice(), packed_mul.as_slice());
}

#[test]
fn test_packed_mul_identity() {
    let a = random_packed();
    let one = PackedGoldilocksAVX2::ones();
    assert_eq!((a * one).as_slice(), a.as_slice());
}

#[test]
fn test_packed_mul_zero() {
    let a = random_packed();
    let zero = PackedGoldilocksAVX2::zero();
    let result = a * zero;
    for lane in result.as_slice() {
        assert_eq!(*lane, FE::zero());
    }
}

#[test]
fn test_packed_mul_large_values() {
    // Near-prime values that stress the reduction
    let a = PackedGoldilocksAVX2::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
    let b = PackedGoldilocksAVX2::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
    let result = a * b;
    // (p-1)^2 mod p = 1
    assert_eq!(result.as_slice()[0], FE::from(1u64));
}

#[test]
fn test_packed_distributivity() {
    let a = random_packed();
    let b = random_packed();
    let c = random_packed();
    // a * (b + c) = a*b + a*c
    let lhs = a * (b + c);
    let rhs = a * b + a * c;
    assert_eq!(lhs.as_slice(), rhs.as_slice());
}
```

**Step 2: Run tests to verify they fail**

Run: `RUSTFLAGS="-C target-feature=+avx2" cargo test -p math --features parallel -- u64_goldilocks_packed::x86_64_avx2::tests::test_packed_mul`
Expected: FAIL (mul currently uses scalar fallback which works, but let's replace it with SIMD).

Actually, the scalar fallback in the placeholder will make tests pass. So first replace the `Mul` impl with a `todo!()` or check the implementation is actually SIMD by benchmarking. Instead, let's just implement directly and verify correctness.

**Step 3: Implement mul64_64, reduce128, and mul_avx2**

Replace the placeholder `Mul` impl in `x86_64_avx2.rs` with the full SIMD multiply:

```rust
/// Multiply two packed 64-bit values, producing (hi, lo) 128-bit results.
/// Uses four 32×32→64 sub-multiplications via _mm256_mul_epu32.
#[inline(always)]
unsafe fn mul64_64(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    // Extract high 32-bit halves. movehdup_ps runs on port 5,
    // avoiding contention with mul_epu32 on ports 0/1.
    let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
    let y_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(y)));

    // Four sub-products (each uses only the low 32 bits of each 64-bit lane)
    let mul_ll = _mm256_mul_epu32(x, y);       // x_lo * y_lo → 64-bit
    let mul_lh = _mm256_mul_epu32(x, y_hi);    // x_lo * y_hi → 64-bit
    let mul_hl = _mm256_mul_epu32(x_hi, y);    // x_hi * y_lo → 64-bit
    let mul_hh = _mm256_mul_epu32(x_hi, y_hi); // x_hi * y_hi → 64-bit

    // Assemble 128-bit result: hi * 2^64 + lo
    let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
    let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);     // cannot overflow 64 bits
    let t0_lo = _mm256_and_si256(t0, EPSILON_VEC);     // low 32 bits
    let t0_hi = _mm256_srli_epi64::<32>(t0);           // high 32 bits
    let t1 = _mm256_add_epi64(mul_lh, t0_lo);          // cannot overflow 64 bits
    let t1_hi = _mm256_srli_epi64::<32>(t1);
    let res_hi = _mm256_add_epi64(_mm256_add_epi64(mul_hh, t0_hi), t1_hi);

    // Form low 64 bits: low32(mul_ll) | high32(t1)
    let t1_lo_shifted = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(t1)));
    let res_lo = _mm256_blend_epi32::<0b10101010>(mul_ll, t1_lo_shifted);

    (res_hi, res_lo)
}

/// Reduce a 128-bit packed value (hi, lo) to 64-bit Goldilocks elements.
///
/// For x = hi * 2^64 + lo:
///   x mod P = lo - hi_hi + hi_lo * EPSILON
///
/// where hi_hi = hi >> 32, hi_lo = hi & 0xFFFFFFFF.
/// Uses: 2^96 ≡ -1 (mod P) and 2^64 ≡ EPSILON (mod P).
#[inline(always)]
unsafe fn reduce128_avx2(hi: __m256i, lo: __m256i) -> __m256i {
    let lo_s = shift(lo);
    let hi_hi = _mm256_srli_epi64::<32>(hi);

    // lo - hi_hi (in shifted domain)
    // hi_hi < 2^32, so this is a "small" subtraction
    let lo1_s = sub_small_s(lo_s, hi_hi);

    // hi_lo * EPSILON — _mm256_mul_epu32 naturally uses only low 32 bits of hi
    let t1 = _mm256_mul_epu32(hi, EPSILON_VEC);

    // lo1 + t1 (in shifted domain)
    // t1 < (2^32 - 1)^2 < 2^64, so this is also safe
    let lo2_s = add_small_s(lo1_s, t1);

    shift(lo2_s)
}

/// Subtract a "small" value (< 2^32) from a shifted value.
#[inline(always)]
unsafe fn sub_small_s(x_s: __m256i, y_small: __m256i) -> __m256i {
    let res_s = _mm256_sub_epi64(x_s, y_small);
    // Borrow if res > x (unsigned), which in shifted domain means res > x (signed)
    let mask = _mm256_cmpgt_epi64(res_s, x_s);
    let correction = _mm256_srli_epi64::<32>(mask); // EPSILON
    _mm256_sub_epi64(res_s, correction)
}

/// Add a "small" value (< 2^64) to a shifted value.
#[inline(always)]
unsafe fn add_small_s(x_s: __m256i, y: __m256i) -> __m256i {
    let res_s = _mm256_add_epi64(x_s, y);
    let mask = _mm256_cmpgt_epi64(x_s, res_s); // overflow
    let correction = _mm256_srli_epi64::<32>(mask);
    _mm256_add_epi64(res_s, correction)
}

/// Packed modular multiplication: (a * b) mod P
#[inline(always)]
unsafe fn mul_avx2(a: __m256i, b: __m256i) -> __m256i {
    let (hi, lo) = mul64_64(a, b);
    reduce128_avx2(hi, lo)
}

/// Packed modular squaring: a^2 mod P (3 sub-products instead of 4)
#[inline(always)]
unsafe fn square_avx2(a: __m256i) -> __m256i {
    let a_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(a)));

    let mul_ll = _mm256_mul_epu32(a, a);       // a_lo * a_lo
    let mul_lh = _mm256_mul_epu32(a, a_hi);    // a_lo * a_hi (= a_hi * a_lo)
    let mul_hh = _mm256_mul_epu32(a_hi, a_hi); // a_hi * a_hi

    // Double the cross term (shift left by 33 instead of 32 to account for 2×)
    let mul_ll_hi = _mm256_srli_epi64::<33>(mul_ll);
    let t0 = _mm256_add_epi64(mul_lh, mul_ll_hi);
    let t0_hi = _mm256_srli_epi64::<31>(t0);
    let res_hi = _mm256_add_epi64(mul_hh, t0_hi);

    let t0_lo_shifted = _mm256_slli_epi64::<33>(t0);
    let mul_ll_lo = _mm256_and_si256(mul_ll, _mm256_set1_epi64x(1)); // just bit 0
    let res_lo = _mm256_or_si256(t0_lo_shifted, mul_ll_lo);

    reduce128_avx2(res_hi, res_lo)
}
```

Then replace the `Mul` impl:

```rust
impl Mul for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { mul_avx2(self.to_vector(), rhs.to_vector()) })
    }
}
```

And override `square` in the `PackedField` impl:

```rust
fn square(&self) -> Self {
    Self::from_vector(unsafe { square_avx2(self.to_vector()) })
}
```

**Step 4: Run tests**

Run: `RUSTFLAGS="-C target-feature=+avx2" cargo test -p math --features parallel -- u64_goldilocks_packed`
Expected: all tests pass including mul, square, distributivity, identity, zero, large values.

**Step 5: Commit**

```bash
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx2.rs
git commit -m "Add AVX2 packed multiply and square for Goldilocks"
```

---

## Task 4: Stress-test AVX2 with random values

**Files:**
- Modify: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx2.rs` (test module)

**Context:** The shifted-representation arithmetic and 128-bit reduction have subtle edge cases. We need proptest/randomized tests to catch them.

**Step 1: Add comprehensive random tests**

Add to the test module (requires the `proptest` feature in dev-dependencies, which is already present):

```rust
use proptest::prelude::*;

fn arb_packed() -> impl Strategy<Value = PackedGoldilocksAVX2> {
    prop::array::uniform4(0u64..GOLDILOCKS_PRIME)
        .prop_map(|arr| PackedGoldilocksAVX2::from_fn(|i| FE::from(arr[i])))
}

proptest! {
    #[test]
    fn prop_add_commutative(a in arb_packed(), b in arb_packed()) {
        let ab = a + b;
        let ba = b + a;
        for i in 0..4 {
            prop_assert_eq!(
                GoldilocksField::canonical(ab.as_slice()[i].value()),
                GoldilocksField::canonical(ba.as_slice()[i].value()),
            );
        }
    }

    #[test]
    fn prop_mul_commutative(a in arb_packed(), b in arb_packed()) {
        let ab = a * b;
        let ba = b * a;
        for i in 0..4 {
            prop_assert_eq!(
                GoldilocksField::canonical(ab.as_slice()[i].value()),
                GoldilocksField::canonical(ba.as_slice()[i].value()),
            );
        }
    }

    #[test]
    fn prop_add_matches_scalar(a_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
                                b_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME)) {
        let a = PackedGoldilocksAVX2::from_fn(|i| FE::from(a_vals[i]));
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from(b_vals[i]));
        let packed = a + b;
        for i in 0..4 {
            let scalar = FE::from(a_vals[i]) + FE::from(b_vals[i]);
            prop_assert_eq!(
                GoldilocksField::canonical(packed.as_slice()[i].value()),
                GoldilocksField::canonical(scalar.value()),
            );
        }
    }

    #[test]
    fn prop_mul_matches_scalar(a_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
                                b_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME)) {
        let a = PackedGoldilocksAVX2::from_fn(|i| FE::from(a_vals[i]));
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from(b_vals[i]));
        let packed = a * b;
        for i in 0..4 {
            let scalar = FE::from(a_vals[i]) * FE::from(b_vals[i]);
            prop_assert_eq!(
                GoldilocksField::canonical(packed.as_slice()[i].value()),
                GoldilocksField::canonical(scalar.value()),
            );
        }
    }

    #[test]
    fn prop_sub_is_add_neg(a in arb_packed(), b in arb_packed()) {
        let sub_result = a - b;
        let add_neg_result = a + (-b);
        for i in 0..4 {
            prop_assert_eq!(
                GoldilocksField::canonical(sub_result.as_slice()[i].value()),
                GoldilocksField::canonical(add_neg_result.as_slice()[i].value()),
            );
        }
    }

    #[test]
    fn prop_square_matches_mul(a in arb_packed()) {
        let sq = a.square();
        let mul = a * a;
        for i in 0..4 {
            prop_assert_eq!(
                GoldilocksField::canonical(sq.as_slice()[i].value()),
                GoldilocksField::canonical(mul.as_slice()[i].value()),
            );
        }
    }
}
```

**Step 2: Run property tests**

Run: `RUSTFLAGS="-C target-feature=+avx2" cargo test -p math --features "parallel proptest" -- u64_goldilocks_packed::x86_64_avx2::tests::prop`
Expected: all property tests pass (256 cases each by default).

**Step 3: Commit**

```bash
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx2.rs
git commit -m "Add proptest stress tests for AVX2 packed Goldilocks arithmetic"
```

---

## Task 5: PackedGoldilocksAVX512

**Files:**
- Create: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/x86_64_avx512.rs`
- Modify: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/mod.rs`

**Context:** AVX-512 uses `__m512i` (512-bit) registers, holding WIDTH=8 Goldilocks elements. The key simplification over AVX2: native unsigned 64-bit comparison via `_mm512_cmpge_epu64_mask` → `__mmask8`, and conditional operations via `_mm512_mask_*` intrinsics. No shifted-representation trick needed.

The implementation follows the same structure as AVX2 but uses `_mm512_*` intrinsics:
- Add: `_mm512_add_epi64` + `_mm512_cmplt_epu64_mask` + `_mm512_mask_sub_epi64`
- Sub: `_mm512_sub_epi64` + `_mm512_cmpgt_epu64_mask` + `_mm512_mask_add_epi64`
- Mul: same 4-sub-product decomposition with `_mm512_mul_epu32`
- Interleave: 3 levels (`u64`, `u128`, `u256`) using `_mm512_unpacklo/hi_epi64`, `_mm512_shuffle_i64x2`, `_mm512_permutex2var_epi64`

Write the implementation, tests, and proptest following the same pattern as Task 2-4 but with `__m512i` and WIDTH=8.

Uncomment the AVX-512 lines in `mod.rs`.

**Step 1: Implement, test, commit**

Run: `RUSTFLAGS="-C target-feature=+avx512f" cargo test -p math --features parallel -- u64_goldilocks_packed::x86_64_avx512`
Expected: all tests pass.

```bash
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/
git commit -m "Add PackedGoldilocksAVX512 (WIDTH=8) with full arithmetic"
```

---

## Task 6: PackedGoldilocksNeon

**Files:**
- Create: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/aarch64_neon.rs`
- Modify: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/mod.rs`

**Context:** NEON uses `uint64x2_t` (128-bit) registers, WIDTH=2. Addition/subtraction use the shifted-representation trick (like AVX2) with `vcgtq_s64`. Multiplication uses inline assembly with `mul`/`umulh` (AArch64 has native 64×64→128 in two instructions), interleaving both lanes for instruction-level parallelism.

Write the implementation following Plonky3's `goldilocks/src/aarch64_neon/packing.rs`.

Interleave: single level (`u64`) using `vzip1q_u64`/`vzip2q_u64`.

Uncomment the NEON lines in `mod.rs`.

**Step 1: Implement, test, commit**

Run (on ARM machine or cross-compile): `cargo test -p math --features parallel -- u64_goldilocks_packed::aarch64_neon`

```bash
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/
git commit -m "Add PackedGoldilocksNeon (WIDTH=2) with full arithmetic"
```

---

## Task 7: PackedFp3 — packed cubic extension field

**Files:**
- Create: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/fp3.rs`
- Modify: `crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/mod.rs`

**Context:** `PackedFp3<P>` wraps three packed base field values `(c0, c1, c2)` representing WIDTH independent Fp3 elements. All arithmetic delegates to packed base field ops. The multiplication formula uses the same Karatsuba-like approach from `extensions_goldilocks.rs:206-218` with residue `w^3 = 2`.

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    type FE = FieldElement<GoldilocksField>;
    type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

    fn random_packed_fp3() -> PackedFp3<PackedGoldilocks> {
        PackedFp3 {
            c0: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 111)),
            c1: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 222)),
            c2: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 333)),
        }
    }

    #[test]
    fn test_packed_fp3_mul_matches_scalar() {
        let a = random_packed_fp3();
        let b = random_packed_fp3();
        let packed_prod = a * b;
        // Verify each lane matches scalar Fp3 multiplication
        for i in 0..PackedGoldilocks::WIDTH {
            let a_scalar = Fp3E::new([a.c0.as_slice()[i], a.c1.as_slice()[i], a.c2.as_slice()[i]]);
            let b_scalar = Fp3E::new([b.c0.as_slice()[i], b.c1.as_slice()[i], b.c2.as_slice()[i]]);
            let expected = a_scalar * b_scalar;
            assert_eq!(packed_prod.c0.as_slice()[i], expected.value()[0]);
            assert_eq!(packed_prod.c1.as_slice()[i], expected.value()[1]);
            assert_eq!(packed_prod.c2.as_slice()[i], expected.value()[2]);
        }
    }

    #[test]
    fn test_packed_fp3_scalar_mul() {
        // Test F × E → E (base field scalar × extension)
        let scalar = PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 42));
        let ext = random_packed_fp3();
        let result = ext.mul_scalar(scalar);
        for i in 0..PackedGoldilocks::WIDTH {
            let s = scalar.as_slice()[i];
            assert_eq!(result.c0.as_slice()[i], s * ext.c0.as_slice()[i]);
            assert_eq!(result.c1.as_slice()[i], s * ext.c1.as_slice()[i]);
            assert_eq!(result.c2.as_slice()[i], s * ext.c2.as_slice()[i]);
        }
    }
}
```

**Step 2: Implement PackedFp3**

```rust
//! Packed cubic extension field Fp3 = Fp[w] / (w^3 - 2).
//!
//! Holds WIDTH independent Fp3 elements across 3 packed base field values.

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use crate::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField;
use crate::field::packed::PackedField;
use core::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign};

/// Packed cubic extension: 3 packed base field values = WIDTH independent Fp3 elements.
#[derive(Copy, Clone, Debug)]
pub struct PackedFp3<P: PackedField<Scalar = GoldilocksField>> {
    pub c0: P,
    pub c1: P,
    pub c2: P,
}

impl<P: PackedField<Scalar = GoldilocksField>> PackedFp3<P> {
    pub fn zero() -> Self {
        Self { c0: P::zero(), c1: P::zero(), c2: P::zero() }
    }

    pub fn one() -> Self {
        Self { c0: P::ones(), c1: P::zero(), c2: P::zero() }
    }

    /// Multiply by a packed base field scalar: (s * c0, s * c1, s * c2)
    /// This is the F×E→E multiplication critical for constraint evaluation.
    #[inline(always)]
    pub fn mul_scalar(self, s: P) -> Self {
        Self {
            c0: self.c0 * s,
            c1: self.c1 * s,
            c2: self.c2 * s,
        }
    }

    /// Construct from a closure that returns (c0, c1, c2) per lane.
    pub fn from_fn(mut f: impl FnMut(usize) -> [FieldElement<GoldilocksField>; 3]) -> Self {
        Self {
            c0: P::from_fn(|i| f(i)[0]),
            c1: P::from_fn(|i| f(i)[1]),
            c2: P::from_fn(|i| f(i)[2]),
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Add for PackedFp3<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self { c0: self.c0 + rhs.c0, c1: self.c1 + rhs.c1, c2: self.c2 + rhs.c2 }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Sub for PackedFp3<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self { c0: self.c0 - rhs.c0, c1: self.c1 - rhs.c1, c2: self.c2 - rhs.c2 }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Neg for PackedFp3<P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { c0: -self.c0, c1: -self.c1, c2: -self.c2 }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Mul for PackedFp3<P> {
    type Output = Self;
    /// Karatsuba-like multiplication modulo w^3 = 2.
    /// Same formula as extensions_goldilocks.rs:206-218.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        let v2 = self.c2 * rhs.c2;

        let t0 = (self.c1 + self.c2) * (rhs.c1 + rhs.c2) - v1 - v2;
        let t1 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1;
        let t2 = (self.c0 + self.c2) * (rhs.c0 + rhs.c2) - v0 - v2;

        // residue = 2, so multiply by 2 = double
        Self {
            c0: v0 + t0.double(),   // v0 + 2*(cross terms)
            c1: t1 + v2.double(),   // cross + 2*v2
            c2: t2 + v1,            // cross + v1
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> AddAssign for PackedFp3<P> {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<P: PackedField<Scalar = GoldilocksField>> SubAssign for PackedFp3<P> {
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl<P: PackedField<Scalar = GoldilocksField>> MulAssign for PackedFp3<P> {
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}
```

**Step 3: Run tests and commit**

Run: `RUSTFLAGS="-C target-feature=+avx2" cargo test -p math --features parallel -- u64_goldilocks_packed::fp3`
Expected: all tests pass.

```bash
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/fp3.rs
git add crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/mod.rs
git commit -m "Add PackedFp3 cubic extension for packed Goldilocks"
```

---

## Task 8: Benchmark packed vs scalar arithmetic

**Files:**
- Modify: `crypto/math/benches/goldilocks_benchmark.rs` (or create new bench)

**Context:** Verify we actually get speedup. Compare packed add/mul throughput against scalar.

**Step 1: Write benchmarks**

```rust
fn bench_packed_add(c: &mut Criterion) {
    let a = PackedGoldilocks::broadcast(FE::from(0x123456789ABCDEFu64));
    let b = PackedGoldilocks::broadcast(FE::from(0xFEDCBA987654321u64));
    c.bench_function("packed_add", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc = acc + b;
            }
            criterion::black_box(acc)
        })
    });
}

fn bench_scalar_add(c: &mut Criterion) {
    let a = FE::from(0x123456789ABCDEFu64);
    let b = FE::from(0xFEDCBA987654321u64);
    c.bench_function("scalar_add", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc = acc + b;
            }
            criterion::black_box(acc)
        })
    });
}

// Same pattern for mul, square, fp3_mul
```

**Step 2: Run benchmarks**

Run: `RUSTFLAGS="-C target-feature=+avx2" cargo bench -p math -- goldilocks`

**Step 3: Commit**

```bash
git add crypto/math/benches/
git commit -m "Add benchmarks comparing packed vs scalar Goldilocks arithmetic"
```

---

## Task 9: Full prover integration test

**Files:** None new — run existing tests.

**Context:** Ensure packed code doesn't break anything even though it's not yet used by FFT/constraints. The ScalarPacked fallback is active on platforms without target features.

**Step 1: Run full test suite**

Run: `cargo test -p math --features parallel`
Run: `cargo test -p stark --features parallel`
Run: `cargo test -p lambda-vm-prover --features parallel`

Expected: all tests pass. The packed modules are compiled but not yet plugged into FFT or constraints.

**Step 2: Verify CI compatibility**

Run: `cargo build --features parallel` (without target-feature flags)
Expected: compiles with ScalarPacked fallback.

**Step 3: Commit (if any fixes needed)**

---

## Summary

After Phase 1 completion:

| Component | Status |
|---|---|
| `PackedField` trait | Done |
| `ScalarPacked` fallback | Done |
| `PackedGoldilocksAVX2` (WIDTH=4) | Done (add, sub, mul, square, neg, interleave) |
| `PackedGoldilocksAVX512` (WIDTH=8) | Done |
| `PackedGoldilocksNeon` (WIDTH=2) | Done |
| `PackedFp3<P>` extension | Done |
| Benchmarks | Done |
| Prover integration | Not started (Phase 2-4) |

**Next phases** (separate plan docs):
- **Phase 2**: FFT butterfly SIMD — packed Bowers butterfly, dispatch in `polynomial.rs`
- **Phase 3**: Constraint evaluation SIMD — `PackedFrame`, packed evaluator loop
- **Phase 4**: Merkle tree SIMD — multi-lane Keccak, batched hashing
