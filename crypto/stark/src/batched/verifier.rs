//! Replaying a batched epoch's transcript, and authenticating its openings.
//!
//! # ⛔ THIS IS NOT A COMPLETE VERIFIER
//!
//! [`verify_epoch_commitments`] checks that the proof's openings are the ones
//! the committed roots bind, at the query indices the transcript derives, under
//! the shape the AIR set implies — and that the grinding nonce and the query
//! count are what the epoch's parameters demand. It does NOT check:
//!
//! - the constraint identity at `z` (`step_2_verify_claimed_composition_polynomial`),
//! - the DEEP/FRI join — that the opened rows evaluate to the FRI's `p0`,
//! - the cross-table LogUp bus balance.
//!
//! A proof that passes this function is NOT valid. The name says `commitments`
//! rather than `verify` for that reason, and there is no `multi_verify_batched`
//! yet: shipping one that skipped the constraint check would be worse than
//! shipping none.
//!
//! ## Why the rest is not here — a design call, not an oversight
//!
//! The three missing checks all exist in `crate::verifier`, and all three take
//! `StarkProofView<'_, ..>` — the rkyv zero-copy view added by #845. A batched
//! epoch proof is not a per-table `StarkProof`, so it has no such view, and the
//! two ways to reach those checks both have a real cost:
//!
//! 1. **Refactor them to take plain data.** `step_2_verify_claimed_composition_polynomial`,
//!    `compute_query_invariant_deep_terms` and
//!    `reconstruct_deep_composition_poly_evaluations_for_all_queries` would each
//!    take slices instead of a view. That is the right end state, and
//!    `reconstruct_deep_composition_poly_evaluation_pair` ALREADY takes slices,
//!    so the change is smaller than it looks — but it edits the production
//!    verifier's hot path and the view layer MMCS-PLAN §2.1 warns a careless
//!    rebase silently deletes.
//! 2. **Give the batched proof its own archived view.** Duplicates the view
//!    layer for a second wire format before that format is settled.
//!
//! Option 1 is the recommendation, and the reason it is not taken here is that
//! it should be taken deliberately rather than as a side effect of wiring a
//! verifier. See `RESUME-MMCS-INT.md`.
//!
//! # What the replay IS
//!
//! [`replay_epoch_transcript`] walks exactly the sequence
//! `crate::batched::prover::multi_prove_batched` walks. It is the batched
//! epoch's analogue of `prover_commit_matches_verifier_derivation`: the two
//! sides are one protocol, and `replay_matches_the_provers_ending_state` pins
//! them on the ENDING TRANSCRIPT STATE, which no single challenge comparison
//! can substitute for — a divergence anywhere in the sequence shows up there.

use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::batched::proof::BatchedMultiProof;
use crate::batched::round4::reduce_iota_to_round;
use crate::batched::shape::{EpochFriParams, EpochShape, RoundShape};
use crate::config::{Commitment, GrindingDigest, StarkHash};
use crate::fri::batched::{BatchedFriChallenges, absorb_shape_histogram};
use crate::fri::mmcs::{MixedMmcs, MixedOpening};
use crate::lookup::LOGUP_NUM_CHALLENGES;
use crate::traits::AIR;

use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};

/// Every challenge a batched epoch derives, in the order the transcript
/// produces them.
#[derive(Debug, Clone)]
pub struct EpochChallenges<E: IsField> {
    /// The shared LogUp challenges. Empty when no table has a RAP.
    pub lookup: Vec<FieldElement<E>>,
    /// One constraint-batching challenge per table, in table order.
    pub betas: Vec<FieldElement<E>>,
    /// One out-of-domain point per table, in table order.
    pub zs: Vec<FieldElement<E>>,
    /// One DEEP-batching challenge per table, in table order.
    pub deep_gammas: Vec<FieldElement<E>>,
    /// The batched FRI instance's challenges, including the query indices and
    /// the instance-class partition.
    pub fri: BatchedFriChallenges<E>,
}

/// Replay a batched epoch's transcript and recover every challenge.
///
/// Returns `None` on any structural disagreement between the proof and the
/// shape the AIR set implies. Every input here is prover-supplied, so every
/// disagreement is a rejection; this function does not panic.
pub fn replay_epoch_transcript<Field, FieldExtension, PI, T>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    transcript: &mut T,
) -> Option<(EpochShape, EpochFriParams, EpochChallenges<FieldExtension>)>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
    T: IsStarkTranscript<FieldExtension, Field>,
{
    if airs.len() != proof.tables.len() || airs.is_empty() {
        return None;
    }
    let trace_lengths: Vec<usize> = proof.tables.iter().map(|t| t.trace_length).collect();
    let (shape, params) = EpochShape::derive(airs, &trace_lengths).ok()?;

    // Recommendation S: the shape is bound before the first root, so every
    // challenge below — not only round 4's — is drawn after the epoch has
    // committed to what it is.
    absorb_shape_histogram::<FieldExtension, T>(transcript, &shape.heights, &shape.total_widths());

    // A round the AIR set says exists must have a root, and one it says does not
    // must not — otherwise a prover could add or drop a whole round's binding.
    if shape.prep.is_empty() != proof.prep_root.is_none() {
        return None;
    }
    if let Some(root) = proof.prep_root.as_ref() {
        transcript.append_bytes(root);
    }
    transcript.append_bytes(&proof.main_root);

    let needs_lookup = airs.iter().any(|air| air.has_aux_trace());
    let lookup: Vec<FieldElement<FieldExtension>> = if needs_lookup {
        (0..LOGUP_NUM_CHALLENGES)
            .map(|_| transcript.sample_field_element())
            .collect()
    } else {
        Vec::new()
    };

    if shape.aux.is_empty() != proof.aux_root.is_none() {
        return None;
    }
    if let Some(root) = proof.aux_root.as_ref() {
        transcript.append_bytes(root);
    }

    // Which tables carry a bus contribution is a property of the AIR set, not
    // of the proof. Absorbing whatever the proof happened to send would let a
    // prover move the whole transcript by adding or omitting one.
    for (air, table) in airs.iter().zip(proof.tables.iter()) {
        match (air.has_aux_trace(), table.bus_public_inputs.as_ref()) {
            (true, Some(bpi)) => transcript.append_field_element(&bpi.table_contribution),
            (false, None) => {}
            _ => return None,
        }
    }

    let betas: Vec<FieldElement<FieldExtension>> = (0..airs.len())
        .map(|_| transcript.sample_field_element())
        .collect();

    transcript.append_bytes(&proof.parts_root);

    let coset_offset = FieldElement::<Field>::from(params.coset_offset);
    let mut zs = Vec::with_capacity(airs.len());
    for (index, table) in proof.tables.iter().enumerate() {
        let lde_length = table.trace_length.checked_shl(params.blowup_log)?;
        // `sample_z_ood_with_domain_params` is the routine the prover reaches
        // through `sample_z_ood`, so the two agree by naming one function
        // rather than by two call sites coinciding.
        let z = transcript.sample_z_ood_with_domain_params(
            table.trace_length,
            lde_length,
            &coset_offset,
        );

        let air = airs.get(index)?;
        // Shape-check the two OOD blocks before absorbing them: they are
        // proof-supplied, and the prover's absorption walked blocks the AIR's
        // layout defines. A block of the wrong width would otherwise absorb a
        // different number of field elements and desynchronise the transcript
        // rather than being rejected.
        if !ood_blocks_well_formed(*air, table) {
            return None;
        }
        for block in [&table.trace_ood_evaluations, &table.trace_ood_next_evaluations] {
            for col in block.columns().iter() {
                for elem in col.iter() {
                    transcript.append_field_element(elem);
                }
            }
        }
        for element in table.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }
        zs.push(z);
    }

    let deep_gammas: Vec<FieldElement<FieldExtension>> = (0..airs.len())
        .map(|_| transcript.sample_field_element())
        .collect();

    let fri = crate::fri::batched::derive_batched_fri_challenges::<FieldExtension, T>(
        transcript,
        &shape.heights,
        &shape.total_widths(),
        &proof.fri_layer_roots,
        &proof.fri_final_poly_coeffs,
        params.blowup_log,
        params.final_poly_log_degree,
        params.grinding_factor,
        proof.nonce,
        params.num_queries,
    )?;

    Some((
        shape,
        params,
        EpochChallenges {
            lookup,
            betas,
            zs,
            deep_gammas,
            fri,
        },
    ))
}

/// The two OOD blocks must have the shape the AIR's layout defines.
///
/// This is `crate::verifier`'s `ood_blocks_well_formed`, restated against the
/// batched proof's owned tables rather than an rkyv view. It is not cosmetic:
/// the blocks are absorbed element by element, so a block of the wrong width
/// would desynchronise the transcript instead of being rejected, and the
/// verifier would go on to derive challenges from a sequence the prover never
/// walked.
fn ood_blocks_well_formed<Field, FieldExtension, PI>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    table: &crate::batched::proof::BatchedTableData<FieldExtension, PI>,
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
{
    let step_size = air.step_size();
    let num_eval_points = air.context().transition_offsets.len() * step_size;
    let expected_next_width = air.trace_ood_next_row_columns().len();
    let expected_next_height = if expected_next_width == 0 {
        0
    } else {
        num_eval_points.saturating_sub(step_size)
    };
    let current = &table.trace_ood_evaluations;
    let next = &table.trace_ood_next_evaluations;

    current.width == air.trace_layout().0 + air.num_auxiliary_rap_columns()
        && current.height == step_size
        && next.width == expected_next_width
        && next.height == expected_next_height
}

/// Authenticate every query's openings against every batched round's root, and
/// check the epoch-level structural facts the transcript binds.
///
/// ⛔ See the module header: this is NOT a complete verification. It is the
/// commitment half.
pub fn verify_epoch_commitments<Field, FieldExtension, PI, H>(
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    shape: &EpochShape,
    params: &EpochFriParams,
    challenges: &EpochChallenges<FieldExtension>,
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
    FieldElement<Field>: AsBytes + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + Sync + Send,
    H: StarkHash,
{
    // The query count is not implied by anything the transcript already
    // checked: a prover that sent fewer openings would simply be checked less.
    if proof.queries.len() != params.num_queries
        || challenges.fri.iotas.len() != params.num_queries
    {
        return false;
    }

    if params.grinding_factor > 0 {
        let Some(nonce) = proof.nonce else {
            return false;
        };
        if !crate::grinding::is_valid_nonce::<GrindingDigest<H>>(
            &challenges.fri.grinding_seed,
            nonce,
            params.grinding_factor,
        ) {
            return false;
        }
    }

    // The instance-class partition is DERIVED, never sent, so the proof's
    // terminal polynomials must be present for exactly the standalone tables
    // and of exactly the length that class's degree bound implies.
    for (table, data) in proof.tables.iter().enumerate() {
        let standalone = challenges.fri.plan.standalone.contains(&table);
        match (&data.standalone_final_poly_coeffs, standalone) {
            (Some(coeffs), true) => {
                let Some(&height) = shape.heights.get(table) else {
                    return false;
                };
                let Some(log_degree) = (height as u32).checked_sub(params.blowup_log) else {
                    return false;
                };
                if coeffs.len() != 1usize << log_degree {
                    return false;
                }
            }
            (None, false) => {}
            _ => return false,
        }
    }

    let h_max = shape.h_max();
    for (query, iota) in challenges.fri.iotas.iter().copied().enumerate() {
        let opening = &proof.queries[query];
        if !round_authenticates::<Field, H>(&proof.main_root, &opening.main, &shape.main, iota, h_max)
        {
            return false;
        }
        if !round_authenticates::<FieldExtension, H>(
            &proof.parts_root,
            &opening.parts,
            &shape.parts,
            iota,
            h_max,
        ) {
            return false;
        }
        match (proof.prep_root.as_ref(), opening.prep.as_ref()) {
            (Some(root), Some(o)) => {
                if !round_authenticates::<Field, H>(root, o, &shape.prep, iota, h_max) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
        match (proof.aux_root.as_ref(), opening.aux.as_ref()) {
            (Some(root), Some(o)) => {
                if !round_authenticates::<FieldExtension, H>(root, o, &shape.aux, iota, h_max) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
    }

    true
}

/// Authenticate one round at one query, reducing the shared FRI index into the
/// round's own index space first.
///
/// The reduction is the whole reason this is a named function rather than four
/// inline calls: the preprocessed and auxiliary rounds can have an `h_max`
/// below the FRI's, and passing the un-reduced index is not a loud error —
/// prover and verifier share the routine, so a wrong convention is
/// self-consistent (`fri/mmcs.rs`, "Index convention").
fn round_authenticates<C, H>(
    root: &Commitment,
    opening: &MixedOpening<C>,
    round: &RoundShape,
    iota_fri: usize,
    h_max_fri: usize,
) -> bool
where
    C: IsField + 'static,
    H: StarkHash,
    FieldElement<C>: AsBytes + Sync + Send,
{
    let Some(h_max_round) = round.h_max() else {
        return false;
    };
    let Some(iota) = reduce_iota_to_round(iota_fri, h_max_fri, h_max_round) else {
        return false;
    };
    MixedMmcs::<C, H>::verify_batch(root, iota, opening, &round.heights(), &round.widths())
}
