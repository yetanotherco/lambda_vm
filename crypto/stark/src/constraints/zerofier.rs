//! Zerofier evaluation as free functions of [`ConstraintMeta`].
//!
//! The production zerofier path: `AIR::transition_zerofier_evaluations_grouped`
//! (prover) and the verifier's OOD zerofier denominators both evaluate these
//! over each constraint's plain metadata (`period` / `offset` /
//! `exemptions_period` / `periodic_exemptions_offset` / `end_exemptions`).
//! The bodies were relocated verbatim from the deleted boxed-constraint trait's
//! default methods (equivalence was asserted by migration-time tests).

use core::ops::Div;

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
    // Last row in the constraint's evaluation domain is g^(offset + N - period);
    // walking backward by g^period gives the remaining end-exemption roots.
    let period = meta.period;
    let decrement = trace_primitive_root.pow(trace_length - period);
    let mut current = trace_primitive_root.pow(meta.offset + trace_length - period);
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
/// With no end exemptions the zerofier is cyclic, so a short period-length
/// vector is returned and the consumer cycles it (same contract as the trait
/// default this body was moved from).
#[allow(unstable_name_collisions)]
pub fn zerofier_evaluations_on_extended_domain<F: IsFFTField>(
    meta: &ConstraintMeta,
    domain: &Domain<F>,
) -> Vec<FieldElement<F>> {
    let blowup_factor = domain.blowup_factor;
    let trace_length = domain.trace_roots_of_unity.len();
    let trace_primitive_root = &domain.trace_primitive_root;
    let coset_offset = &domain.coset_offset;
    let lde_root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
    let lde_root = F::get_primitive_root_of_unity(lde_root_order).unwrap();

    // If there is an exemptions period defined for this constraint, the evaluations are calculated directly
    // by computing P_exemptions(x) / Zerofier(x)
    if let Some(exemptions_period) = meta.exemptions_period {
        debug_assert!(exemptions_period.is_multiple_of(meta.period));
        debug_assert!(meta.periodic_exemptions_offset.is_some());

        // The elements of the domain have order `trace_length * blowup_factor`, so the zerofier evaluations
        // without the end exemptions, repeat their values after `blowup_factor * exemptions_period` iterations,
        // so we only need to compute those.
        let last_exponent = blowup_factor * exemptions_period;
        let numerator_power = trace_length / exemptions_period;
        let denominator_power = trace_length / meta.period;
        let offset_exponent =
            trace_length * meta.periodic_exemptions_offset.unwrap() / exemptions_period;
        let numerator_offset = trace_primitive_root.pow(offset_exponent);
        let denominator_offset = trace_primitive_root.pow(meta.offset * denominator_power);
        let numerator_step = lde_root.pow(numerator_power);
        let denominator_step = lde_root.pow(denominator_power);
        let mut numerator_eval = coset_offset.pow(numerator_power);
        let mut denominator_eval = coset_offset.pow(denominator_power);

        let mut numerators = Vec::with_capacity(last_exponent);
        let mut denominators = Vec::with_capacity(last_exponent);
        for _ in 0..last_exponent {
            numerators.push(&numerator_eval - &numerator_offset);
            denominators.push(&denominator_eval - &denominator_offset);
            numerator_eval = &numerator_eval * &numerator_step;
            denominator_eval = &denominator_eval * &denominator_step;
        }

        // Batch inversion: O(3N) muls + 1 inversion instead of N individual inversions.
        // Denominators are guaranteed non-zero because the sets of powers of
        // `offset_times_x` and `trace_primitive_root` are disjoint, provided that the
        // offset is neither an element of the interpolation domain nor part of a
        // subgroup with order less than n.
        FieldElement::inplace_batch_inverse(&mut denominators).unwrap();

        let evaluations: Vec<_> = numerators
            .iter()
            .zip(denominators.iter())
            .map(|(num, denom_inv)| num * denom_inv)
            .collect();

        // Mirror the else-branch fast path: with no end exemptions the zerofier stays
        // cyclic, so return the short period-length vector and let the consumer cycle.
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

    // In this else branch, the zerofiers are computed as the numerator, then inverted
    // using batch inverse and then multiplied by P_exemptions(x). This way we don't do
    // useless divisions.
    } else {
        let last_exponent = blowup_factor * meta.period;
        let denominator_power = trace_length / meta.period;
        let denominator_offset = trace_primitive_root.pow(meta.offset * denominator_power);
        let denominator_step = lde_root.pow(denominator_power);
        let mut denominator_eval = coset_offset.pow(denominator_power);

        let mut evaluations = Vec::with_capacity(last_exponent);
        for _ in 0..last_exponent {
            evaluations.push(&denominator_eval - &denominator_offset);
            denominator_eval = &denominator_eval * &denominator_step;
        }

        FieldElement::inplace_batch_inverse(&mut evaluations).unwrap();

        // Fast path: when end_exemptions == 0 there are no exemption roots, so
        // the zerofier stays cyclic — return the short period-length vector
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
}

/// Evaluation of the constraint's zerofier at some point `z`, which may be in
/// a field extension.
#[allow(unstable_name_collisions)]
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

    if let Some(exemptions_period) = meta.exemptions_period {
        debug_assert!(exemptions_period.is_multiple_of(meta.period));
        debug_assert!(meta.periodic_exemptions_offset.is_some());

        let periodic_exemptions_offset = meta.periodic_exemptions_offset.unwrap();
        let offset_exponent = trace_length * periodic_exemptions_offset / exemptions_period;

        let numerator =
            -trace_primitive_root.pow(offset_exponent) + z.pow(trace_length / exemptions_period);
        let denominator = -trace_primitive_root.pow(meta.offset * trace_length / meta.period)
            + z.pow(trace_length / meta.period);
        // The denominator is non-zero: z is sampled outside the set of primitive roots.
        return numerator
            .div(denominator)
            .expect("zerofier denominator is non-zero: z is sampled out-of-domain")
            * &end_exemptions_eval;
    }

    (-trace_primitive_root.pow(meta.offset * trace_length / meta.period)
        + z.pow(trace_length / meta.period))
    .inv()
    .unwrap()
        * &end_exemptions_eval
}
