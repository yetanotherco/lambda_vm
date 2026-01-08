use core::fmt::Display;

use crate::cyclic_group::IsGroup;
use crate::unsigned_integer::traits::IsUnsignedInteger;

#[derive(Debug)]
pub enum MSMError {
    LengthMismatch(usize, usize),
}

impl Display for MSMError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MSMError::LengthMismatch(cs, points) => write!(
                f,
                "`cs` and `points` must be of the same length to compute `msm`. Got: {cs} and {points}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MSMError {}

/// This function computes the multiscalar multiplication (MSM).
///
/// Assume a group G of order r is given.
/// Let `points = [g_1, ..., g_n]` be a tuple of group points in G and
/// let `cs = [k_1, ..., k_n]` be a tuple of scalars in the Galois field GF(r).
///
/// Then, with additive notation, `msm(cs, points)` computes k_1 * g_1 + .... + k_n * g_n.
///
/// If `points` and `cs` are empty, then `msm` returns the zero element of the group.
///
/// Panics if `cs` and `points` have different lengths.
pub fn msm<C, T>(cs: &[C], points: &[T]) -> Result<T, MSMError>
where
    C: IsUnsignedInteger,
    T: IsGroup,
{
    if cs.len() != points.len() {
        return Err(MSMError::LengthMismatch(cs.len(), points.len()));
    }
    let res = cs
        .iter()
        .zip(points.iter())
        .map(|(&c, h)| h.operate_with_self(c))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap_or_else(T::neutral_element);

    Ok(res)
}
