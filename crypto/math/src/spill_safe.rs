//! Marker trait for types whose in-memory bytes can be reinterpreted as the
//! same type without UB: no padding, every bit pattern valid, no indirection.
//!
//! Stricter than `Copy`, which permits types with restricted bit patterns
//! (e.g. `bool`, `NonZeroU32`).
//!
//! `unsafe impl` puts the layout invariants on the implementer. The
//! compiler does not check.

use crate::field::{element::FieldElement, traits::IsField};

/// # Safety
/// Implementer asserts `Self`'s memory representation contains no padding,
/// every bit pattern is a valid value of `Self`, and `Self` carries no
/// indirection (heap pointers, references, etc.). Adding this `unsafe impl`
/// for a type that violates these invariants is UB at any byte cast.
pub unsafe trait SpillSafe: Copy + 'static {}

unsafe impl SpillSafe for u8 {}
unsafe impl SpillSafe for u16 {}
unsafe impl SpillSafe for u32 {}
unsafe impl SpillSafe for u64 {}
unsafe impl SpillSafe for u128 {}
unsafe impl SpillSafe for i8 {}
unsafe impl SpillSafe for i16 {}
unsafe impl SpillSafe for i32 {}
unsafe impl SpillSafe for i64 {}
unsafe impl SpillSafe for i128 {}

unsafe impl<T: SpillSafe, const N: usize> SpillSafe for [T; N] {}

// `FieldElement<F>` is `#[repr(transparent)]` over `F::BaseType`, so its
// layout matches the base type's exactly. SpillSafe propagates through.
unsafe impl<F: IsField + Copy + 'static> SpillSafe for FieldElement<F> where F::BaseType: SpillSafe {}
