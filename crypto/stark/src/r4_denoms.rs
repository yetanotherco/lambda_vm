//! Single-source builder for R4 DEEP inverse denominators on CPU.
//!
//! Called by both the prover's CPU fallback in
//! `compute_deep_composition_poly_evaluations` and by the GPU parity test
//! that pins this construction against the device pipeline
//! (`compute_and_invert_denoms_ext3_dev`). Keeping it in one place means a
//! sign/ordering/layout drift cannot diverge CUDA and non-CUDA builds
//! silently.
//!
//! Convention (mirrors `compute_and_invert_denoms_ext3_dev` with
//! `DenomSign::XMinusZ`):
//!   - `z_scalars = [z_power, z_shifted[0..]]`, length `1 + z_shifted.len()`
//!   - `denoms[k * lde_size + i] = x_i - z_scalars[k]` (then inverted)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};

/// Build `1 / (x_i - z_k)` for k in [0..=z_shifted.len()] and i in [0..n)
/// where `z = [z_power, z_shifted[0..]]`. Output is flat, k-major:
/// `out[k * coset.len() + i] = (x_i - z_k)^{-1}`.
///
/// Returns `Err` only if `inplace_batch_inverse` hits a zero element,
/// which is unreachable in honest proving (Fiat-Shamir `z` on the LDE
/// coset is negligible) but the contract follows lambdaworks' API.
pub fn build_r4_inv_denoms_cpu<F, E>(
    coset: &[FieldElement<F>],
    z_power: &FieldElement<E>,
    z_shifted: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, &'static str>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = coset.len();
    let num_denoms = n * (1 + z_shifted.len());
    let mut denoms: Vec<FieldElement<E>> = Vec::with_capacity(num_denoms);
    for z_k in core::iter::once(z_power).chain(z_shifted.iter()) {
        for x_i in coset {
            denoms.push(x_i - z_k);
        }
    }
    FieldElement::inplace_batch_inverse(&mut denoms)
        .map_err(|_| "R4 inv denoms: zero denominator (z hit the LDE coset)")?;
    Ok(denoms)
}
