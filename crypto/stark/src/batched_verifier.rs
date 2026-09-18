//! Verification of a batched proof: every table's rounds after round 1, and one
//! FRI per height group over the fold of their DEEP codewords.
//!
//! The per-table steps are the ordinary verifier's, run on a view of each
//! table's data with the FRI left empty; the fold coefficients and the group
//! FRI challenges are replayed from the shared seed the prover used, and the
//! group's first FRI layer is the coefficient-weighted sum of the tables' DEEP
//! evaluations at the group's query indices.

use crate::config::Commitment;
use crate::domain::{VerifierDomain, new_verifier_domain};
use crate::proof::stark::StarkProof;
use crate::proof::view::StarkProofView;
use crate::prover::GroupFri;
use crate::table::Table;
use crate::traits::AIR;
use crate::verifier::{Challenges, IsStarkVerifier, RoundsChallenges, Verifier};
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use log::error;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

/// One height group as the verifier sees it.
pub struct BatchedGroup<'a, E: IsField> {
    /// Rows of every member's trace: they share the domain or they could not
    /// have been folded together.
    pub trace_rows: usize,
    pub fri: &'a GroupFri<E>,
}

struct GroupReplay<F: IsFFTField, E: IsField> {
    domain: VerifierDomain<F>,
    layout: crate::fri::terminal::FriFoldLayout,
    zetas: Vec<FieldElement<E>>,
    iotas: Vec<usize>,
    acc: Vec<FieldElement<E>>,
    acc_sym: Vec<FieldElement<E>>,
    /// The first table of the group, whose AIR supplied the domain. Kept from
    /// the scan that already found it rather than looked up again: the second
    /// scan is what forced an `expect` into verifier code.
    member: usize,
}

/// Verify the rounds after round 1 of every table and each group's FRI.
///
/// `transcript` is the shared transcript right after the LogUp challenges: the
/// state every table's fork and the fold seed start from. `tables[i]` carries
/// table `i`'s round-1 roots, round-3 data and openings; its FRI fields are
/// ignored. `fold_order` is the order the prover drew the coefficients in and
/// `groups` the order it ran the FRIs in.
#[allow(clippy::too_many_arguments)]
pub fn verify_batched<Field, FieldExtension, PI>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    public_inputs: &[PI],
    tables: &[StarkProof<Field, FieldExtension, PI>],
    group_of: &[usize],
    groups: &[BatchedGroup<'_, FieldExtension>],
    fold_order: &[usize],
    transcript: &(impl IsStarkTranscript<FieldExtension, Field> + Clone),
    rap_challenges: &[FieldElement<FieldExtension>],
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: IsField + Send + Sync,
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, crate::verifier::PiDeserializer>,
    FieldElement<Field>: AsBytes + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + Sync + Send,
{
    type V<F, E, P> = Verifier<F, E, P>;
    let n = airs.len();
    if tables.len() != n || public_inputs.len() != n || group_of.len() != n || fold_order.len() != n
    {
        error!("batched: {} AIRs against {} tables", n, tables.len());
        return false;
    }
    if groups.is_empty() || group_of.iter().any(|&g| g >= groups.len()) {
        error!("batched: a table names a group the proof does not have");
        return false;
    }
    let mut seen = vec![false; n];
    for &idx in fold_order {
        if idx >= n || std::mem::replace(&mut seen[idx], true) {
            error!("batched: fold order is not a permutation of the tables");
            return false;
        }
    }
    let Some(first_air) = airs.first() else {
        return false;
    };
    let num_queries = first_air.options().fri_number_of_queries;
    let grinding_factor = first_air.context().proof_options.grinding_factor;

    // Every table's domain is its group's, and every block and opening has the
    // shape its AIR declares.
    //
    // Both run before the fold seed below reads a single out-of-domain value.
    // That seed indexes the blocks at the dimensions the *proof* advertises, so
    // a proof whose advertised dimensions disagree with its data length has to
    // be rejected here rather than panic there — the rule `ood_blocks_well_formed`
    // documents, and the one the ordinary verifier keeps by running it in round 1.
    for (idx, ((air, table), &g)) in airs.iter().zip(tables).zip(group_of).enumerate() {
        if table.trace_length == 0 || table.trace_length != groups[g].trace_rows {
            error!("batched: table {idx} does not live on its group's domain");
            return false;
        }
        let view = StarkProofView::Owned(table);
        // `trace_length` is non-zero by the check above, so the division is safe.
        if table.composition_poly_parts_ood_evaluation.len()
            != air.composition_poly_degree_bound(table.trace_length) / table.trace_length
            || !V::<Field, FieldExtension, PI>::ood_blocks_well_formed(*air, view)
            || !V::<Field, FieldExtension, PI>::trace_opening_widths_well_formed(
                *air,
                view,
                num_queries,
            )
        {
            error!("batched: table {idx}'s blocks or openings are malformed");
            return false;
        }
    }

    // The fold coefficients, from the seed, in the prover's order.
    let mut seed = transcript.clone();
    let mut coefficients = vec![FieldElement::<FieldExtension>::zero(); n];
    for &idx in fold_order {
        let t = &tables[idx];
        if let Some(ref bpi) = t.bus_public_inputs {
            seed.append_field_element(&bpi.table_contribution);
        }
        seed.append_bytes(&t.composition_poly_root);
        for ood in [&t.trace_ood_evaluations, &t.trace_ood_next_evaluations] {
            for col_idx in 0..ood.width {
                for row_idx in 0..ood.height {
                    seed.append_field_element(&ood.get_row(row_idx)[col_idx]);
                }
            }
        }
        for elem in t.composition_poly_parts_ood_evaluation.iter() {
            seed.append_field_element(elem);
        }
        coefficients[idx] = seed.sample_field_element();
    }

    // Each group's FRI challenges, from the same seed, in the prover's order.
    let mut replays: Vec<GroupReplay<Field, FieldExtension>> = Vec::with_capacity(groups.len());
    for (g, group) in groups.iter().enumerate() {
        let Some(member) = group_of.iter().position(|&h| h == g) else {
            error!("batched: group {g} has no member");
            return false;
        };
        let domain = new_verifier_domain(airs[member], group.trace_rows);
        let layout = V::<Field, FieldExtension, PI>::fri_termination_params(airs[member], &domain);
        let fri = group.fri;
        if fri.layer_roots.len() != layout.num_committed
            || fri.final_poly_coeffs.len() != (1usize << layout.effective_k)
            || fri.query_list.len() != num_queries
            || fri.iotas.len() != num_queries
            || fri.query_list.iter().any(|q| {
                q.layers_auth_paths.len() != layout.num_committed
                    || q.layers_evaluations_sym.len() != layout.num_committed
            })
        {
            error!("batched: group {g}'s FRI has the wrong shape");
            return false;
        }
        let mut zetas: Vec<FieldElement<FieldExtension>> = fri
            .layer_roots
            .iter()
            .map(|root| {
                let zeta = seed.sample_field_element();
                seed.append_bytes(root);
                zeta
            })
            .collect();
        if layout.total_folds > 0 {
            zetas.push(seed.sample_field_element());
        }
        for c in fri.final_poly_coeffs.iter() {
            seed.append_field_element(c);
        }
        if grinding_factor > 0 {
            let grinding_seed = seed.state();
            let Some(nonce) = fri.nonce else {
                error!("batched: group {g} has no grinding nonce");
                return false;
            };
            if !crate::grinding::is_valid_nonce(&grinding_seed, nonce, grinding_factor) {
                error!("batched: group {g}'s grinding nonce is not valid");
                return false;
            }
            seed.append_bytes(&nonce.to_be_bytes());
        }
        let iotas =
            V::<Field, FieldExtension, PI>::sample_query_indexes(num_queries, &domain, &mut seed);
        if iotas != fri.iotas {
            error!("batched: group {g}'s query indices are not the transcript's");
            return false;
        }
        replays.push(GroupReplay {
            domain,
            layout,
            zetas,
            iotas,
            acc: vec![FieldElement::zero(); num_queries],
            acc_sym: vec![FieldElement::zero(); num_queries],
            member,
        });
    }

    // Every table: rounds 2 and 3 against its fork, its openings at the group's
    // indices, and its DEEP evaluations there, folded into the group's.
    for (idx, ((air, table), &g)) in airs.iter().zip(tables).zip(group_of).enumerate() {
        let view = StarkProofView::Owned(table);
        let mut fork = transcript.clone();
        if n > 1 {
            fork.append_bytes(&(idx as u64).to_le_bytes());
        }
        if let Some(ref root) = table.lde_trace_aux_merkle_root {
            fork.append_bytes(root);
        }
        if let Some(ref bpi) = table.bus_public_inputs {
            fork.append_field_element(&bpi.table_contribution);
        }
        // Shapes were pinned to the AIR in the pre-pass above, before the fold
        // seed read any of this table's blocks.
        let domain = new_verifier_domain(*air, table.trace_length);
        let layout = V::<Field, FieldExtension, PI>::ood_layout(*air);
        let RoundsChallenges {
            z,
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs,
            gammas,
        } = V::<Field, FieldExtension, PI>::replay_rounds_2_and_3(
            *air,
            view,
            &public_inputs[idx],
            &domain,
            &mut fork,
            rap_challenges,
            &layout,
        );
        let replay = &replays[g];
        let challenges = Challenges {
            z,
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs,
            gammas,
            zetas: replay.zetas.clone(),
            iotas: replay.iotas.clone(),
            rap_challenges: rap_challenges.to_vec(),
            grinding_seed: [0u8; 32],
        };
        let ood_current = view.trace_ood_evaluations();
        let ood_next = view.trace_ood_next_evaluations();
        let ood_full = layout.reconstruct_full(
            ood_current.row_major_data(),
            ood_current.width(),
            ood_next.row_major_data(),
        );
        if !V::<Field, FieldExtension, PI>::step_2_verify_claimed_composition_polynomial(
            *air,
            view,
            &public_inputs[idx],
            &domain,
            &challenges,
            &ood_full,
            layout.step_size(),
        ) {
            error!("batched: table {idx} fails the out-of-domain consistency check");
            return false;
        }
        if !V::<Field, FieldExtension, PI>::step_4_verify_trace_and_composition_openings(
            view,
            &challenges,
        ) {
            error!("batched: table {idx}'s openings do not authenticate");
            return false;
        }
        let Some((evals, evals_sym)) =
            V::<Field, FieldExtension, PI>::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                &challenges,
                &domain,
                view,
                &ood_full,
                layout.next_row_cols(),
                layout.step_size(),
            )
        else {
            error!("batched: table {idx}'s DEEP evaluations cannot be reconstructed");
            return false;
        };
        let coefficient = &coefficients[idx];
        let replay = &mut replays[g];
        for (acc, eval) in replay.acc.iter_mut().zip(evals.iter()) {
            *acc = &*acc + coefficient * eval;
        }
        for (acc, eval) in replay.acc_sym.iter_mut().zip(evals_sym.iter()) {
            *acc = &*acc + coefficient * eval;
        }
    }

    // Each group's FRI, from the folded first layer down to the final polynomial.
    for (g, (group, replay)) in groups.iter().zip(replays.iter()).enumerate() {
        let member = replay.member;
        let fri = group.fri;
        let synthetic = StarkProof::<Field, FieldExtension, PI> {
            trace_length: group.trace_rows,
            lde_trace_main_merkle_root: Commitment::default(),
            lde_trace_aux_merkle_root: None,
            lde_trace_precomputed_merkle_root: None,
            trace_ood_evaluations: Table::new(Vec::new(), 0),
            trace_ood_next_evaluations: Table::new(Vec::new(), 0),
            composition_poly_root: Commitment::default(),
            composition_poly_parts_ood_evaluation: Vec::new(),
            fri_layers_merkle_roots: fri.layer_roots.clone(),
            fri_final_poly_coeffs: fri.final_poly_coeffs.clone(),
            query_list: fri.query_list.clone(),
            deep_poly_openings: Vec::new(),
            nonce: fri.nonce,
            bus_public_inputs: None,
            public_inputs: public_inputs[member].clone(),
        };
        let view = StarkProofView::Owned(&synthetic);
        let terminal_offset = replay
            .domain
            .coset_offset
            .pow(1u64 << replay.layout.total_folds);
        let terminal_codeword =
            crate::fri::terminal::terminal_codeword_from_coeffs::<Field, FieldExtension>(
                &fri.final_poly_coeffs,
                &terminal_offset,
                replay.layout.terminal_len,
            );
        let mut inverses: Vec<FieldElement<Field>> = replay
            .iotas
            .iter()
            .map(|&iota| {
                V::<Field, FieldExtension, PI>::query_challenge_to_evaluation_point(
                    iota,
                    false,
                    &replay.domain,
                )
            })
            .collect();
        if FieldElement::inplace_batch_inverse(&mut inverses).is_err() {
            error!("batched: group {g} has a query at a zero point");
            return false;
        }
        let ok = (0..num_queries).zip(inverses).all(|(i, inv)| {
            V::<Field, FieldExtension, PI>::verify_query_and_sym_openings(
                view,
                &replay.zetas,
                replay.iotas[i],
                view.query(i),
                inv,
                &replay.acc[i],
                &replay.acc_sym[i],
                &terminal_codeword,
            )
        });
        if !ok {
            error!("batched: group {g}'s FRI does not verify");
            return false;
        }
    }
    true
}
