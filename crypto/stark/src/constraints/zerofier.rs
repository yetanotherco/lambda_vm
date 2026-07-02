//! Zerofier evaluation as free functions of [`ConstraintMeta`].
//!
//! The production zerofier path: `AIR::transition_zerofier_evaluations_grouped`
//! (prover) and the verifier's OOD zerofier denominators both evaluate these
//! over each constraint's plain metadata. Every constraint applies to every
//! row of the trace, so the zerofier is `x^N − 1` corrected by the constraint's
//! `end_exemptions` (the last rows it must skip).

use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};

use crate::constraints::builder::ConstraintMeta;
use crate::domain::Domain;

/// Roots of the end-exemptions polynomial `∏(x - rᵢ)`.
///
/// The end-exemptions polynomial vanishes on the last `end_exemptions` rows
/// the constraint must skip. This returns its roots `rᵢ` so callers can
/// evaluate the product `∏(x - rᵢ)` directly at the points they need.
pub fn end_exemptions_roots<F: IsField>(
    meta: &ConstraintMeta,
    trace_primitive_root: &FieldElement<F>,
    trace_length: usize,
) -> Vec<FieldElement<F>> {
    let end_exemptions = meta.end_exemptions;
    if end_exemptions == 0 {
        return Vec::new();
    }
    // The last row of the trace is g^(N-1); walking backward by g^-1 = g^(N-1)
    // gives the remaining end-exemption roots.
    let decrement = trace_primitive_root.pow(trace_length - 1);
    let mut current = trace_primitive_root.pow(trace_length - 1);
    let mut roots = Vec::with_capacity(end_exemptions);
    for _ in 0..end_exemptions {
        roots.push(current.clone());
        current = &current * &decrement;
    }
    roots
}

/// Evaluations of the end-exemptions polynomial `∏(x - rᵢ)` over the LDE
/// domain.
///
/// The product has degree `end_exemptions` (≤ 2 in practice), so the direct
/// `O(N · end_exemptions)` product over the precomputed LDE coset is cheaper
/// than an `O(N log N)` FFT. With no exemptions this yields all ones.
pub fn end_exemptions_lde_evaluations<F: IsFFTField>(
    meta: &ConstraintMeta,
    domain: &Domain<F>,
) -> Vec<FieldElement<F>> {
    let roots = end_exemptions_roots(
        meta,
        &domain.trace_primitive_root,
        domain.trace_roots_of_unity.len(),
    );
    domain
        .lde_roots_of_unity_coset
        .iter()
        .map(|x| {
            roots
                .iter()
                .fold(FieldElement::<F>::one(), |acc, r| acc * (x - r))
        })
        .collect()
}

/// Compute evaluations of the constraint's zerofier over a LDE domain.
///
/// With no end exemptions the zerofier `1/(x^N − 1)` is cyclic over the LDE
/// coset, so a short blowup-length vector is returned and the consumer cycles
/// it (same contract as the trait default this body was moved from).
pub fn zerofier_evaluations_on_extended_domain<F: IsFFTField>(
    meta: &ConstraintMeta,
    domain: &Domain<F>,
) -> Vec<FieldElement<F>> {
    let blowup_factor = domain.blowup_factor;
    let trace_length = domain.trace_roots_of_unity.len();
    let coset_offset = &domain.coset_offset;
    let lde_root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
    let lde_root = F::get_primitive_root_of_unity(lde_root_order).unwrap();

    // The zerofiers are computed as the numerator, then inverted using batch
    // inverse and then multiplied by P_exemptions(x). This way we don't do
    // useless divisions. x^N over the LDE coset repeats after blowup_factor
    // points, so only those are computed.
    let last_exponent = blowup_factor;
    let denominator_offset = FieldElement::<F>::one();
    let denominator_step = lde_root.pow(trace_length);
    let mut denominator_eval = coset_offset.pow(trace_length);

    let mut evaluations = Vec::with_capacity(last_exponent);
    for _ in 0..last_exponent {
        evaluations.push(&denominator_eval - &denominator_offset);
        denominator_eval = &denominator_eval * &denominator_step;
    }

    FieldElement::inplace_batch_inverse(&mut evaluations).unwrap();

    // Fast path: when end_exemptions == 0 there are no exemption roots, so
    // the zerofier stays cyclic — return the short blowup-length vector
    // directly instead of expanding it over the full LDE domain.
    if meta.end_exemptions == 0 {
        return evaluations;
    }

    let end_exemption_evaluations = end_exemptions_lde_evaluations(meta, domain);

    let cycled_evaluations = evaluations
        .iter()
        .cycle()
        .take(end_exemption_evaluations.len());

    core::iter::zip(cycled_evaluations, end_exemption_evaluations)
        .map(|(eval, exemption_eval)| eval * exemption_eval)
        .collect()
}

/// Evaluation of the constraint's zerofier at some point `z`, which may be in
/// a field extension.
pub fn evaluate_zerofier<F, E>(
    meta: &ConstraintMeta,
    z: &FieldElement<E>,
    trace_primitive_root: &FieldElement<F>,
    trace_length: usize,
) -> FieldElement<E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let roots = end_exemptions_roots(meta, trace_primitive_root, trace_length);
    // Factor `z - rᵢ` written as `-(rᵢ - z)`: the field ops only go
    // subfield − superfield, and `rᵢ ∈ F`, `z ∈ E`.
    let end_exemptions_eval = roots.iter().fold(FieldElement::<E>::one(), |acc, root| {
        acc * -(root.clone() - z.clone())
    });

    // 1/(z^N − 1), times the end-exemptions correction.
    (-FieldElement::<F>::one() + z.pow(trace_length))
        .inv()
        .unwrap()
        * &end_exemptions_eval
}
