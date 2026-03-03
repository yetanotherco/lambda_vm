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
    type Scalar: IsField;
    const WIDTH: usize;

    fn from_fn(f: impl FnMut(usize) -> FieldElement<Self::Scalar>) -> Self;
    fn from_slice(slice: &[FieldElement<Self::Scalar>]) -> Self;
    fn as_slice(&self) -> &[FieldElement<Self::Scalar>];
    fn as_slice_mut(&mut self) -> &mut [FieldElement<Self::Scalar>];
    fn zero() -> Self;
    fn ones() -> Self;
    fn broadcast(value: FieldElement<Self::Scalar>) -> Self;

    fn square(&self) -> Self {
        *self * *self
    }

    fn double(&self) -> Self {
        *self + *self
    }

    fn pack_slice_with_suffix(
        buf: &[FieldElement<Self::Scalar>],
    ) -> (&[Self], &[FieldElement<Self::Scalar>]) {
        let n_packed = buf.len() / Self::WIDTH;
        let packed_len = n_packed * Self::WIDTH;
        let (head, tail) = buf.split_at(packed_len);
        let packed = unsafe {
            core::slice::from_raw_parts(head.as_ptr() as *const Self, n_packed)
        };
        (packed, tail)
    }

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

    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self);
}

/// Scalar "packed" field — WIDTH=1 fallback for platforms without SIMD.
#[derive(Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct ScalarPacked<F: IsField>(pub FieldElement<F>);

impl<F: IsField> Copy for ScalarPacked<F> where FieldElement<F>: Copy {}

impl<F: IsField> Clone for ScalarPacked<F>
where
    FieldElement<F>: Copy,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

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
