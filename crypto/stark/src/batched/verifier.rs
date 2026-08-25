//! The batched epoch verifier.
//!
//! [`multi_verify_batched`] is the counterpart of
//! `crate::verifier::IsStarkVerifier::multi_verify` for the batched path, and it
//! is a COMPLETE verification: transcript replay, opening authentication against
//! all four mixed-height MMCS roots, the constraint identity at every table's
//! `z`, the epoch's LogUp bus balance, and the DEEP/FRI join across both
//! instance classes. It is assembled from four pieces, each independently
//! testable and each returning a plain `bool`/`Option` — nothing on this path
//! panics, because every input is prover-supplied.
//!
//! | piece | what it decides |
//! |---|---|
//! | [`replay_epoch_transcript`] | every challenge, and every structural fact the transcript binds |
//! | [`verify_epoch_commitments`] | every preprocessed table's opening authenticates against `air.precomputed_commitment()` (the per-table critical check), and the batched rounds' openings are the rows the roots bind at the derived indices |
//! | [`verify_epoch_constraints`] | the claimed composition polynomial, and the bus balance |
//! | [`verify_epoch_fri`] | those rows fold to the terminal polynomial the proof sent |
//!
//! ⚠ Calling a piece on its own is not a verification. `verify_epoch_commitments`
//! in particular shows only that a proof opened the rows its own roots bind,
//! which an adversary controlling the trace can always arrange. The names are
//! `verify_epoch_*` rather than `verify_*` for that reason; the one function
//! that decides validity is [`multi_verify_batched`].
//!
//! # Shared with the per-table verifier, not reimplemented
//!
//! Three checks are the same mathematics in both paths, and all three are
//! reached through `crate::verifier`'s own functions rather than copied:
//! `step_2_verify_claimed_composition_polynomial`,
//! `compute_query_invariant_deep_terms` and
//! `reconstruct_deep_composition_poly_evaluation_pair`. The first two took an
//! rkyv `StarkProofView` (#845's zero-copy layer) and now take plain data — a
//! batched epoch proof is not a per-table `StarkProof` and has no such view.
//! That refactor is deliberate: a second constraint evaluator written for the
//! batched path is the one thing that would let the two paths disagree about
//! what a valid trace is (PA-PLAN §1.4).
//!
//! # The one protocol, pinned
//!
//! [`replay_epoch_transcript`] walks exactly the sequence
//! `crate::batched::prover::multi_prove_batched` walks, and
//! `replay_matches_the_provers_ending_state` pins the two on the ENDING
//! TRANSCRIPT STATE. No per-challenge comparison substitutes for it: a
//! divergence anywhere — a root absorbed out of order, a challenge one side
//! samples and the other does not, an OOD block walked differently — lands
//! there, whereas comparing individual challenges only catches it if you
//! compared the right one.

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

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;

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
    replay_epoch_transcript_carved(airs, proof, transcript, None)
}

/// As [`replay_epoch_transcript`], for an epoch with a carved main matrix
/// ([`crate::batched::shape::CarvedMain`]).
///
/// `carved_main` is VERIFIER-OWNED configuration, like the AIR set — never
/// read from the proof. The carved root itself IS proof-carried: it is
/// absorbed from `proof.carved_main_root` after the preprocessed roots and
/// before `main_root`, so every challenge is drawn after it. A proof whose
/// carve state disagrees with the configuration is rejected.
pub fn replay_epoch_transcript_carved<Field, FieldExtension, PI, T>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    transcript: &mut T,
    carved_main: Option<usize>,
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
    let (shape, params) = EpochShape::derive_carved(airs, &trace_lengths, carved_main).ok()?;

    // Recommendation S: the shape is bound before the first root, so every
    // challenge below — not only round 4's — is drawn after the epoch has
    // committed to what it is.
    absorb_shape_histogram::<FieldExtension, T>(transcript, &shape.heights, &shape.total_widths());

    // ★ Preprocessed roots are absorbed FROM THE AIR SET, never from the
    // proof — per table, in table order, exactly as the per-table path's
    // Phase A does. A prover that committed different preprocessed content
    // walked a different transcript and diverges from here on.
    for air in airs {
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
        }
    }

    // The carved root, PROOF-CARRIED, in its pinned slot: after every
    // preprocessed root, before `main_root`. Presence must match the
    // verifier-owned carve configuration exactly.
    match (&shape.carved_main, proof.carved_main_root.as_ref()) {
        (Some(_), Some(root)) => transcript.append_bytes(root),
        (None, None) => {}
        _ => return None,
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
        for block in [
            &table.trace_ood_evaluations,
            &table.trace_ood_next_evaluations,
        ] {
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

    let standalone_coeffs: Vec<Option<&[FieldElement<FieldExtension>]>> = proof
        .tables
        .iter()
        .map(|t| t.standalone_final_poly_coeffs.as_deref())
        .collect();
    let fri = crate::fri::batched::derive_batched_fri_challenges::<FieldExtension, T>(
        transcript,
        &shape.heights,
        &shape.total_widths(),
        &proof.fri_layer_roots,
        &proof.fri_final_poly_coeffs,
        &standalone_coeffs,
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
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
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
    if proof.queries.len() != params.num_queries || challenges.fri.iotas.len() != params.num_queries
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
        if !round_authenticates::<Field, H>(
            &proof.main_root,
            &opening.main,
            &shape.main,
            iota,
            h_max,
        ) {
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
        // ★ Per-table preprocessed authentication — the per-table path's
        // critical soundness check, verbatim: each opening authenticates
        // against `air.precomputed_commitment()`, a root the VERIFIER owns.
        // Width and count are bound by the AIR set, not the proof.
        if opening.prep.len() != shape.prep.tables.len() {
            return false;
        }
        for (k, &t) in shape.prep.tables.iter().enumerate() {
            let Some(air) = airs.get(t) else {
                return false;
            };
            let Some(&height) = shape.heights.get(t) else {
                return false;
            };
            let Some(leaf) = reduce_iota_to_round(iota, h_max, height) else {
                return false;
            };
            let o = &opening.prep[k];
            let width = air.num_precomputed_columns();
            if o.evaluations.len() != width || o.evaluations_sym.len() != width {
                return false;
            }
            let leaf_hash = <H::Batched<Field> as crypto::merkle_tree::traits::IsStreamingLeafBackend<Field>>::hash_data_from_slices(
                &o.evaluations,
                &o.evaluations_sym,
            );
            if !crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash::<H::Batched<Field>>(
                &o.proof.merkle_path,
                &air.precomputed_commitment(),
                leaf,
                leaf_hash,
            ) {
                return false;
            }
        }
        // ★ The carved table's standalone main opening: authenticated against
        // the PROOF-CARRIED root (`carved_main_root`) at the reduced index,
        // exactly the mechanics of a preprocessed opening with the root's
        // provenance moved from the AIR set to the proof — the transcript slot
        // (before every challenge) is what binds it. Present iff the epoch is
        // carved; a stray or missing opening is a rejection.
        match (
            &shape.carved_main,
            proof.carved_main_root.as_ref(),
            opening.carved_main.as_ref(),
        ) {
            (Some(c), Some(root), Some(o)) => {
                let Some(&height) = shape.heights.get(c.table) else {
                    return false;
                };
                let Some(leaf) = reduce_iota_to_round(iota, h_max, height) else {
                    return false;
                };
                if o.evaluations.len() != c.width || o.evaluations_sym.len() != c.width {
                    return false;
                }
                let leaf_hash =
                    <H::Batched<Field> as crypto::merkle_tree::traits::IsStreamingLeafBackend<
                        Field,
                    >>::hash_data_from_slices(
                        &o.evaluations, &o.evaluations_sym
                    );
                if !crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash::<H::Batched<Field>>(
                    &o.proof.merkle_path,
                    root,
                    leaf,
                    leaf_hash,
                ) {
                    return false;
                }
            }
            (None, None, None) => {}
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

// ===========================================================================
// The DEEP / FRI join — M-5's core
// ===========================================================================

/// Verify the batched FRI instance and the terminal-only instances, at every
/// query.
///
/// This is the check that gives the authenticated openings their meaning. Up to
/// here a proof has shown that the rows it opened are the rows its roots bind;
/// this shows that those rows evaluate to a codeword the FRI folds to a
/// low-degree polynomial — that the committed trace really does satisfy the
/// DEEP relation at `z`.
///
/// # The two index spaces, again
///
/// Both instance classes are opened at the SAME query indices and read them
/// differently ([`crate::fri::batched::FriInstancePlan`]): the batched class
/// uses `iota` directly because it is an index in the tallest domain, a
/// standalone table at height `h` uses `iota >> (h_max - h)`. A table's OWN
/// row pair also lives at its reduced leaf, which is why the evaluation point
/// each table's DEEP quotient is reconstructed at is derived from the reduced
/// index and not from `iota`.
///
/// # Mixing
///
/// [`crate::fri::batched::HeightCombiner`] scales the `i`-th absorbed codeword
/// by `alpha^i`, counting in absorption order and NOT per height, and the
/// prover absorbs in `plan.batched` order. So the power a table's DEEP value
/// carries here is its position in `plan.batched` — not its table index, and
/// not its position within its height group. Getting that wrong produces a
/// verifier that rejects every honest proof, which is the benign direction, but
/// it is worth stating because the three orders coincide on a same-height
/// epoch.
///
/// Returns `false` on every malformed input; it never panics.
pub fn verify_epoch_fri<Field, FieldExtension, PI, H, V>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
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
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, crate::verifier::PiDeserializer>,
    H: StarkHash,
    V: crate::verifier::IsStarkVerifier<Field, FieldExtension, PI, H> + ?Sized,
{
    let h_max = shape.h_max();
    let layout = &challenges.fri.layout;
    let coset_offset = FieldElement::<Field>::from(params.coset_offset);

    // Structural checks before anything is reconstructed. The terminal helper
    // panics on a coefficient count that does not divide the codeword length,
    // so the length check is not optional — it is what keeps this path
    // rejection-only. Same reasoning as `step_3_verify_fri`.
    if proof.fri_layer_roots.len() != layout.num_committed
        || proof.fri_final_poly_coeffs.len() != (1usize << layout.effective_k)
    {
        return false;
    }
    for query in proof.queries.iter() {
        if query.fri.layers_auth_paths.len() != layout.num_committed
            || query.fri.layers_evaluations_sym.len() != layout.num_committed
        {
            return false;
        }
    }

    let terminal_offset = coset_offset.pow(1u64 << layout.total_folds);
    let terminal_codeword =
        crate::fri::terminal::terminal_codeword_from_coeffs::<Field, FieldExtension>(
            &proof.fri_final_poly_coeffs,
            &terminal_offset,
            layout.terminal_len,
        );

    // Per table: the DEEP value pair at every query, in this table's own
    // (reduced) index space.
    let mut deep_pairs: Vec<Vec<(FieldElement<FieldExtension>, FieldElement<FieldExtension>)>> =
        Vec::with_capacity(airs.len());
    for (table, air) in airs.iter().enumerate() {
        match table_deep_pairs::<Field, FieldExtension, PI, H, V>(
            table, *air, proof, shape, params, challenges,
        ) {
            Some(pairs) => deep_pairs.push(pairs),
            None => return false,
        }
    }

    for (query, iota) in challenges.fri.iotas.iter().copied().enumerate() {
        let mut p0 = (
            FieldElement::<FieldExtension>::zero(),
            FieldElement::<FieldExtension>::zero(),
        );
        let mut buckets: Vec<Option<FieldElement<FieldExtension>>> = vec![None; h_max];
        let mut power = FieldElement::<FieldExtension>::one();

        for &table in challenges.fri.plan.batched.iter() {
            let (Some(&height), Some(pairs)) = (shape.heights.get(table), deep_pairs.get(table))
            else {
                return false;
            };
            let Some((evaluation, evaluation_sym)) = pairs.get(query) else {
                return false;
            };
            if height == h_max {
                p0.0 = &p0.0 + &(&power * evaluation);
                p0.1 = &p0.1 + &(&power * evaluation_sym);
            } else {
                let chosen = crate::batched::round4::injected_value_at_query(
                    iota,
                    h_max,
                    height,
                    evaluation,
                    evaluation_sym,
                );
                let scaled = &power * chosen;
                buckets[height] = Some(match buckets[height].take() {
                    Some(acc) => acc + scaled,
                    None => scaled,
                });
            }
            power = &power * &challenges.fri.alpha;
        }

        // υ⁻¹ in the TALLEST domain — the batched instance's layer 0.
        let lde_length = 1usize << h_max;
        let Some(lde_root) = Field::get_primitive_root_of_unity(h_max as u64).ok() else {
            return false;
        };
        let point = &coset_offset
            * lde_root.pow(math::fft::bit_reversing::reverse_index(
                iota * 2,
                lde_length as u64,
            ));
        let Ok(point_inv) = point.inv() else {
            return false;
        };

        if !crate::batched::round4::verify_batched_fri_query::<Field, FieldExtension, H>(
            &proof.fri_layer_roots,
            &challenges.fri.betas,
            layout,
            h_max,
            iota,
            &proof.queries[query].fri,
            &point_inv,
            (&p0.0, &p0.1),
            &buckets,
            &terminal_codeword,
        ) {
            return false;
        }

        // The other class. A table whose own FRI commits no layer has a
        // terminal codeword that IS its deep-composition codeword, so the check
        // is that the value its opening produced is the value the sent
        // polynomial encodes at the reduced position.
        for &table in challenges.fri.plan.standalone.iter() {
            let (Some(&height), Some(pairs), Some(data)) = (
                shape.heights.get(table),
                deep_pairs.get(table),
                proof.tables.get(table),
            ) else {
                return false;
            };
            let (Some((evaluation, evaluation_sym)), Some(coeffs)) =
                (pairs.get(query), data.standalone_final_poly_coeffs.as_ref())
            else {
                return false;
            };
            let codeword_len = 1usize << height;
            if coeffs.is_empty()
                || !coeffs.len().is_power_of_two()
                || coeffs.len() > codeword_len
                || !codeword_len.is_multiple_of(coeffs.len())
            {
                return false;
            }
            let standalone_terminal = crate::fri::terminal::terminal_codeword_from_coeffs::<
                Field,
                FieldExtension,
            >(coeffs, &coset_offset, codeword_len);
            if !crate::batched::round4::verify_standalone_fri_query(
                iota,
                h_max,
                height,
                (evaluation, evaluation_sym),
                &standalone_terminal,
            ) {
                return false;
            }
        }
    }

    true
}

/// One table's DEEP composition value pair at every query, reconstructed from
/// the authenticated openings.
///
/// The base columns are handed over as two slices in COMMIT order — the
/// preprocessed round's row first, then the main round's — because that is the
/// order the prover concatenated them in and the order the OOD grid and the
/// trace-term coefficients are indexed by. A non-preprocessed table passes an
/// empty first slice, which is exactly what the per-table path does.
fn table_deep_pairs<Field, FieldExtension, PI, H, V>(
    table: usize,
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    shape: &EpochShape,
    params: &EpochFriParams,
    challenges: &EpochChallenges<FieldExtension>,
) -> Option<Vec<(FieldElement<FieldExtension>, FieldElement<FieldExtension>)>>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
    FieldElement<Field>: AsBytes + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + Sync + Send,
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, crate::verifier::PiDeserializer>,
    H: StarkHash,
    V: crate::verifier::IsStarkVerifier<Field, FieldExtension, PI, H> + ?Sized,
{
    let data = proof.tables.get(table)?;
    let &height = shape.heights.get(table)?;
    let h_max = shape.h_max();
    let z = challenges.zs.get(table)?;
    let gamma = challenges.deep_gammas.get(table)?;

    let domain = crate::domain::new_verifier_domain(air, data.trace_length);
    let step_size = air.step_size();
    let ood_layout = crate::ood::OodLayout::new(
        air.context().trace_columns,
        air.context().transition_offsets.len() * step_size,
        step_size,
        air.trace_ood_next_row_columns(),
    );
    let ood_full = ood_layout.reconstruct_full(
        data.trace_ood_evaluations.row_major_data(),
        data.trace_ood_evaluations.width,
        data.trace_ood_next_evaluations.row_major_data(),
    );

    // The DEEP coefficients, derived exactly as the prover derives them: the
    // first `num_surviving` powers of gamma are the trace terms, the rest the
    // composition parts. Splitting them the other way round would be a verifier
    // that rejects every honest proof.
    let num_terms_trace = ood_layout.num_surviving();
    let num_parts = data.composition_poly_parts_ood_evaluation.len();
    let mut powers: Vec<FieldElement<FieldExtension>> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * gamma))
            .take(num_parts + num_terms_trace)
            .collect();
    if powers.len() < num_terms_trace {
        return None;
    }
    let trace_term_powers: Vec<_> = powers.drain(..num_terms_trace).collect();
    let trace_term_coeffs = ood_layout.build_trace_term_coeffs(&trace_term_powers);
    let gammas = powers;

    let table_challenges = crate::verifier::Challenges {
        z: z.clone(),
        boundary_coeffs: Vec::new(),
        transition_coeffs: Vec::new(),
        trace_term_coeffs,
        gammas,
        zetas: Vec::new(),
        iotas: Vec::new(),
        rap_challenges: challenges.lookup.clone(),
        grinding_seed: [0u8; 32],
    };

    let terms = V::query_invariant_deep_terms_from_parts(
        &table_challenges,
        &data.composition_poly_parts_ood_evaluation,
        &ood_full,
        ood_layout.next_row_cols(),
        step_size,
    )?;
    let primitive_root = Field::get_primitive_root_of_unity(domain.root_order as u64).ok()?;

    let prep_matrix = shape.prep.tables.iter().position(|&t| t == table);
    // A carved table has no main-round matrix: its main row pair comes from the
    // standalone carved opening instead (authenticated against the
    // proof-carried root by `verify_epoch_commitments`).
    let is_carved = shape.carved_main.map(|c| c.table) == Some(table);
    let main_matrix = if is_carved {
        None
    } else {
        Some(shape.main.tables.iter().position(|&t| t == table)?)
    };
    let aux_matrix = shape.aux.tables.iter().position(|&t| t == table);
    let parts_matrix = shape.parts.tables.iter().position(|&t| t == table)?;

    let mut pairs = Vec::with_capacity(challenges.fri.iotas.len());
    for (query, iota) in challenges.fri.iotas.iter().copied().enumerate() {
        let opening = proof.queries.get(query)?;
        // This table's OWN row pair: the reduced leaf, in its own domain.
        let leaf = crate::batched::round4::reduce_iota_to_round(iota, h_max, height)?;
        let point = domain.lde_coset_element(math::fft::bit_reversing::reverse_index(
            leaf * 2,
            domain.lde_length as u64,
        ));
        let point_sym = domain.lde_coset_element(math::fft::bit_reversing::reverse_index(
            leaf * 2 + 1,
            domain.lde_length as u64,
        ));

        let empty_base: &[FieldElement<Field>] = &[];
        let empty_ext: &[FieldElement<FieldExtension>] = &[];
        let (prep, prep_sym) = match prep_matrix {
            Some(m) => {
                let o = opening.prep.get(m)?;
                (o.evaluations.as_slice(), o.evaluations_sym.as_slice())
            }
            None => (empty_base, empty_base),
        };
        let (main_evals, main_evals_sym) = match main_matrix {
            Some(m) => {
                let o = opening.main.per_matrix.get(m)?;
                (o.evaluations.as_slice(), o.evaluations_sym.as_slice())
            }
            None => {
                let o = opening.carved_main.as_ref()?;
                (o.evaluations.as_slice(), o.evaluations_sym.as_slice())
            }
        };
        let (aux, aux_sym) = match aux_matrix {
            Some(m) => {
                let o = opening.aux.as_ref()?.per_matrix.get(m)?;
                (o.evaluations.as_slice(), o.evaluations_sym.as_slice())
            }
            None => (empty_ext, empty_ext),
        };
        let parts = opening.parts.per_matrix.get(parts_matrix)?;

        let pair = V::reconstruct_deep_composition_poly_evaluation_pair(
            &point,
            &point_sym,
            &primitive_root,
            &table_challenges,
            &terms,
            ood_layout.next_row_cols(),
            step_size,
            prep,
            main_evals,
            aux,
            &parts.evaluations,
            prep_sym,
            main_evals_sym,
            aux_sym,
            &parts.evaluations_sym,
        )?;
        pairs.push(pair);
    }
    let _ = params;
    Some(pairs)
}

// ===========================================================================
// The constraint identity, the bus balance, and the whole verification
// ===========================================================================

/// Check every table's claimed composition polynomial at its own `z`, and the
/// epoch's LogUp bus balance.
///
/// The constraint check is `crate::verifier`'s
/// `step_2_verify_claimed_composition_polynomial`, unchanged — that function now
/// takes plain data instead of an rkyv view precisely so this caller can reach
/// it. Writing a second constraint evaluator for the batched path is the one
/// thing that would make the two paths able to disagree about what a valid
/// trace is.
///
/// ⚠ `public_inputs` are read from the proof, exactly as the per-table path
/// reads them from `StarkProof`. Checking that they are the inputs the caller
/// meant is the caller's job in both paths.
pub fn verify_epoch_constraints<Field, FieldExtension, PI, H, V>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    challenges: &EpochChallenges<FieldExtension>,
    expected_bus_balance: &FieldElement<FieldExtension>,
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
    FieldElement<Field>: AsBytes + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + Sync + Send,
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, crate::verifier::PiDeserializer>,
    H: StarkHash,
    V: crate::verifier::IsStarkVerifier<Field, FieldExtension, PI, H> + ?Sized,
{
    // Bus balance: Σ table_contribution = expected. This is the cross-table
    // statement no per-table check can make, and it is why the contributions are
    // absorbed before any constraint challenge.
    let mut total = FieldElement::<FieldExtension>::zero();
    for table in proof.tables.iter() {
        if let Some(bpi) = table.bus_public_inputs.as_ref() {
            total += bpi.table_contribution.clone();
        }
    }
    if total != *expected_bus_balance {
        return false;
    }

    for (table, air) in airs.iter().enumerate() {
        let (Some(data), Some(z), Some(beta)) = (
            proof.tables.get(table),
            challenges.zs.get(table),
            challenges.betas.get(table),
        ) else {
            return false;
        };

        let step_size = air.step_size();
        let ood_layout = crate::ood::OodLayout::new(
            air.context().trace_columns,
            air.context().transition_offsets.len() * step_size,
            step_size,
            air.trace_ood_next_row_columns(),
        );
        let ood_full = ood_layout.reconstruct_full(
            data.trace_ood_evaluations.row_major_data(),
            data.trace_ood_evaluations.width,
            data.trace_ood_next_evaluations.row_major_data(),
        );
        let domain = crate::domain::new_verifier_domain(*air, data.trace_length);

        // The constraint-batching coefficients, split exactly as the prover
        // splits them: transitions first, then boundaries.
        let bus_public_inputs = data.bus_public_inputs.clone();
        let num_transition_constraints = air.context().num_transition_constraints;
        let num_boundary_constraints = air
            .boundary_constraints(
                &data.public_inputs,
                &challenges.lookup,
                bus_public_inputs.as_ref(),
                data.trace_length,
            )
            .constraints
            .len();
        let mut coefficients: Vec<FieldElement<FieldExtension>> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * beta))
                .take(num_boundary_constraints + num_transition_constraints)
                .collect();
        if coefficients.len() < num_transition_constraints {
            return false;
        }
        let transition_coeffs: Vec<_> = coefficients.drain(..num_transition_constraints).collect();
        let boundary_coeffs = coefficients;

        let table_challenges = crate::verifier::Challenges {
            z: z.clone(),
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs: Vec::new(),
            gammas: Vec::new(),
            zetas: Vec::new(),
            iotas: Vec::new(),
            rap_challenges: challenges.lookup.clone(),
            grinding_seed: [0u8; 32],
        };

        if !V::step_2_verify_claimed_composition_polynomial(
            *air,
            data.trace_length,
            data.bus_public_inputs
                .as_ref()
                .map(|b| b.table_contribution.clone()),
            data.trace_ood_evaluations.get_row(0),
            &data.composition_poly_parts_ood_evaluation,
            &data.public_inputs,
            &domain,
            &table_challenges,
            &ood_full,
            step_size,
        ) {
            return false;
        }
    }

    true
}

/// Verify a batched epoch proof: replay, commitments, constraint identity, bus
/// balance, DEEP/FRI join.
///
/// This is the counterpart of `crate::verifier::IsStarkVerifier::multi_verify`
/// for the batched path, and unlike the pieces above it is a COMPLETE
/// verification — every check the per-table path makes has a counterpart here,
/// reached through the same functions where the check is shared.
///
/// Preprocessed binding needs no caller-side pin: every preprocessed table's
/// root is `air.precomputed_commitment()` — the verifier's own value, absorbed
/// and compared per table exactly as the per-table path does.
///
/// Returns `false` on every malformed proof; it never panics.
pub fn multi_verify_batched<Field, FieldExtension, PI, H, V, T>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    transcript: &mut T,
    expected_bus_balance: &FieldElement<FieldExtension>,
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
    FieldElement<Field>: AsBytes + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + Sync + Send,
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, crate::verifier::PiDeserializer>,
    H: StarkHash,
    V: crate::verifier::IsStarkVerifier<Field, FieldExtension, PI, H> + ?Sized,
    T: IsStarkTranscript<FieldExtension, Field>,
{
    multi_verify_batched_carved::<Field, FieldExtension, PI, H, V, T>(
        airs,
        proof,
        transcript,
        expected_bus_balance,
        None,
    )
}

/// As [`multi_verify_batched`], for an epoch with a carved main matrix.
///
/// `carved_main` is verifier-owned configuration (which table, if any, commits
/// its main matrix standalone) — the same value the prover was called with,
/// supplied by the CALLER, never read from the proof. Everything else about
/// the carve is checked: the proof-carried root's transcript slot
/// ([`replay_epoch_transcript_carved`]), the per-query opening's
/// authentication and width ([`verify_epoch_commitments`]), and the opened
/// row pair's participation in the DEEP/FRI join ([`verify_epoch_fri`]).
pub fn multi_verify_batched_carved<Field, FieldExtension, PI, H, V, T>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    proof: &BatchedMultiProof<Field, FieldExtension, PI>,
    transcript: &mut T,
    expected_bus_balance: &FieldElement<FieldExtension>,
    carved_main: Option<usize>,
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
    FieldElement<Field>: AsBytes + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + Sync + Send,
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, crate::verifier::PiDeserializer>,
    H: StarkHash,
    V: crate::verifier::IsStarkVerifier<Field, FieldExtension, PI, H> + ?Sized,
    T: IsStarkTranscript<FieldExtension, Field>,
{
    let Some((shape, params, challenges)) =
        replay_epoch_transcript_carved(airs, proof, transcript, carved_main)
    else {
        return false;
    };
    verify_epoch_commitments::<Field, FieldExtension, PI, H>(
        airs,
        proof,
        &shape,
        &params,
        &challenges,
    ) && verify_epoch_constraints::<Field, FieldExtension, PI, H, V>(
        airs,
        proof,
        &challenges,
        expected_bus_balance,
    ) && verify_epoch_fri::<Field, FieldExtension, PI, H, V>(
        airs,
        proof,
        &shape,
        &params,
        &challenges,
    )
}
