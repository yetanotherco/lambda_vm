//! The batched proof's verifier.
//!
//! Alongside `multi_verify`, never in place of it: the per-table path is
//! untouched and its proofs are byte-identical to what they always were.
//!
//! What a verifier decides is not "did the prover follow its own steps" — it is
//! whether a proof it has never seen a prover produce is valid. That is built in
//! pieces, and this is the first: the transcript replay, which derives every
//! challenge from the proof alone. Nothing below it can be checked until the
//! challenges are the prover's, and if they are, a forged proof has to be wrong
//! about something the later pieces test rather than about which questions were
//! asked.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use log::error;
use math::field::element::FieldElement;
use stark::batched_verifier::BatchedGroup;
use stark::proof::options::ProofOptions;
use stark::proof::stark::StarkProof;

use crate::Error;
use crate::batched_proof::BatchedProof;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

/// Every challenge a batched proof's verification needs, derived from the proof.
pub struct Replay {
    /// The one challenge the whole execution shares.
    pub logup: Vec<FieldElement<GoldilocksExtension>>,
    /// Per table, in AIR order: its fold coefficient.
    pub coefficients: Vec<FieldElement<GoldilocksExtension>>,
}

/// Replay the transcript the prover walked, from the proof.
///
/// `elf_bytes` and the statement come from outside the proof on purpose: a
/// proof that could choose its own statement would prove nothing.
///
/// # This is a test oracle, not a verifier
///
/// It derives the challenges and stops. [`verify`] does not call it — it walks
/// the same prefix itself and is the stricter of the two: it checks each
/// preprocessed root against the AIR's own constant where this takes the
/// prover's word, and it rejects a `fold_order` that is not a permutation.
/// Promoting this function to a verification path without closing both gaps
/// would be a soundness hole; it exists so a test can compare challenges.
pub fn replay(
    proof: &BatchedProof,
    elf_bytes: &[u8],
    table_counts: &crate::TableCounts,
    proof_options: &stark::proof::options::ProofOptions,
) -> Result<Replay, Error> {
    let mut transcript = DefaultTranscript::<GoldilocksExtension>::new(&[]);
    crate::statement::absorb_statement(
        &mut transcript,
        crate::statement::StatementKind::Monolithic,
        elf_bytes,
        &proof.public_output,
        table_counts,
        proof
            .page_configs
            .iter()
            .filter(|c| c.is_private_input)
            .count(),
        &crate::tables::trace_builder::runtime_page_ranges(&proof.page_configs),
        proof_options.fri_final_poly_log_degree,
    );

    // Round 1, in AIR order: a preprocessed table's precomputed root first.
    for t in proof.tables.iter() {
        if let Some(ref pre) = t.precomputed_root {
            transcript.append_bytes(pre);
        }
        transcript.append_bytes(&t.main_root);
    }
    let logup: Vec<_> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();

    // The fold coefficients, in the order the prover folded — which the proof
    // carries because a table's coefficient depends on every table before it.
    if proof.fold_order.len() != proof.tables.len() {
        return Err(Error::Prover(format!(
            "batched verify: {} tables folded of {}",
            proof.fold_order.len(),
            proof.tables.len()
        )));
    }
    // Shapes before the seed reads a value, for the same reason
    // `verify_batched` pins them before its own fold loop: `Table::columns`
    // indexes at the advertised dimensions.
    for (idx, t) in proof.tables.iter().enumerate() {
        for block in [&t.trace_ood, &t.trace_ood_next] {
            if !block.dimensions_consistent() {
                return Err(Error::Prover(format!(
                    "batched verify: table {idx}'s out-of-domain block is malformed"
                )));
            }
        }
    }
    let mut seed = transcript.clone();
    let mut coefficients = vec![FieldElement::<GoldilocksExtension>::zero(); proof.tables.len()];
    for &idx in proof.fold_order.iter() {
        let t = proof.tables.get(idx).ok_or_else(|| {
            Error::Prover(format!("batched verify: fold order names table {idx}"))
        })?;
        if let Some(ref bpi) = t.bus_public_inputs {
            seed.append_field_element(&bpi.table_contribution);
        }
        seed.append_bytes(&t.composition_poly_root);
        let blocks: [&stark::table::Table<GoldilocksExtension>; 2] =
            [&t.trace_ood, &t.trace_ood_next];
        for block in blocks {
            for col in block.columns().iter() {
                for elem in col.iter() {
                    seed.append_field_element(elem);
                }
            }
        }
        for elem in t.parts_ood.iter() {
            seed.append_field_element(elem);
        }
        coefficients[idx] = seed.sample_field_element();
    }

    Ok(Replay {
        logup,
        coefficients,
    })
}

/// Verify a batched proof of `elf_bytes`.
///
/// The VM half — the statement, the AIRs rebuilt from the layout the proof
/// declares, the preprocessed roots, the LogUp bus balance — is here; every
/// table's rounds after round 1 and each group's FRI are
/// [`stark::batched_verifier::verify_batched`].
pub fn verify(
    proof: &BatchedProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<bool, Error> {
    verify_with_precomputed(proof, elf_bytes, proof_options, None, None)
}

/// [`verify`] with the ELF-only preprocessed roots supplied instead of
/// recomputed. The recursion guest holds them already; recomputing DECODE and
/// every data page in-VM is the single most expensive thing a verifier can do.
pub fn verify_with_precomputed(
    proof: &BatchedProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    decode_commitment: Option<stark::config::Commitment>,
    page_commitments: Option<&[(u64, stark::config::Commitment)]>,
) -> Result<bool, Error> {
    let table_counts = &proof.table_counts;
    table_counts.validate()?;
    let n = proof.tables.len();
    if proof.openings.len() != n || proof.group_of.len() != n || proof.fold_order.len() != n {
        return Err(Error::InvalidTableCounts(format!(
            "batched proof: {n} tables, {} openings, {} group slots, {} fold entries",
            proof.openings.len(),
            proof.group_of.len(),
            proof.fold_order.len()
        )));
    }
    let elf = executor::elf::Elf::load(elf_bytes)
        .map_err(|e| Error::Prover(format!("batched verify: ELF: {e}")))?;
    let num_private_input_pages = proof
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let max_pages = crate::tables::page::max_private_input_pages();
    if num_private_input_pages > max_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({num_private_input_pages}) exceeds max ({max_pages})",
        )));
    }
    let runtime_page_ranges =
        crate::tables::trace_builder::runtime_page_ranges(&proof.page_configs);
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &elf,
        &runtime_page_ranges,
        num_private_input_pages,
        n,
    )?;
    // `total()` is checked: a proof whose counts sum past `usize` is rejected
    // here rather than wrapping into a plausible-looking expectation.
    let total = table_counts.total().ok_or_else(|| {
        Error::InvalidTableCounts("table_counts total overflows usize".to_string())
    })?;
    let expected = total + crate::FIXED_TABLE_COUNT + page_configs.len();
    if expected != n {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({total}) + {} fixed + {} pages = {expected}, but the proof has {n} tables",
            crate::FIXED_TABLE_COUNT,
            page_configs.len(),
        )));
    }
    let vm_airs = crate::VmAirs::new(
        &elf,
        proof_options,
        false,
        &page_configs,
        table_counts,
        decode_commitment,
        true,
        None,
        page_commitments,
        None,
    );
    let airs = vm_airs.air_refs();
    if airs.len() != n {
        error!("batched verify: {} AIRs for {n} tables", airs.len());
        return Ok(false);
    }

    let mut transcript = DefaultTranscript::<GoldilocksExtension>::new(&[]);
    crate::statement::absorb_statement(
        &mut transcript,
        crate::statement::StatementKind::Monolithic,
        elf_bytes,
        &proof.public_output,
        table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        proof_options.fri_final_poly_log_degree,
    );
    // Round 1, in AIR order. A preprocessed table's precomputed root is the
    // AIR's constant, not the prover's word.
    for (idx, (air, t)) in airs.iter().zip(proof.tables.iter()).enumerate() {
        if air.is_preprocessed() {
            let expected = air.precomputed_commitment();
            match t.precomputed_root {
                Some(actual) if actual == expected => {}
                _ => {
                    error!("batched verify: table {idx}'s precomputed root is not the AIR's");
                    return Ok(false);
                }
            }
            transcript.append_bytes(&expected);
        } else if t.precomputed_root.is_some() {
            error!("batched verify: table {idx} carries a precomputed root it should not");
            return Ok(false);
        }
        transcript.append_bytes(&t.main_root);
    }
    let logup: Vec<FieldElement<GoldilocksExtension>> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();

    // Every interacting table contributes to the bus, no other does, and the
    // contributions balance against the public output.
    for (idx, (air, t)) in airs.iter().zip(proof.tables.iter()).enumerate() {
        if air.has_trace_interaction() != t.bus_public_inputs.is_some() {
            error!("batched verify: table {idx}'s bus inputs do not match its AIR");
            return Ok(false);
        }
    }
    let Some(expected_balance) = crate::compute_commit_bus_offset(
        &proof.public_output,
        0,
        &logup[0],
        &logup[stark::lookup::LOGUP_CHALLENGE_ALPHA],
    ) else {
        error!("batched verify: the public output has no bus balance");
        return Ok(false);
    };
    let mut total = FieldElement::<GoldilocksExtension>::zero();
    for (air, t) in airs.iter().zip(proof.tables.iter()) {
        if air.has_trace_interaction()
            && let Some(ref bpi) = t.bus_public_inputs
        {
            total += bpi.table_contribution;
        }
    }
    if total != expected_balance {
        error!("batched verify: LogUp bus does not balance");
        return Ok(false);
    }

    // Each table as the ordinary verifier reads one, with the FRI left empty.
    let tables: Vec<StarkProof<GoldilocksField, GoldilocksExtension, ()>> = proof
        .tables
        .iter()
        .zip(proof.openings.iter())
        .map(|(t, opening)| StarkProof {
            trace_length: t.trace_rows,
            lde_trace_main_merkle_root: t.main_root,
            lde_trace_aux_merkle_root: t.aux_root,
            lde_trace_precomputed_merkle_root: t.precomputed_root,
            trace_ood_evaluations: t.trace_ood.clone(),
            trace_ood_next_evaluations: t.trace_ood_next.clone(),
            composition_poly_root: t.composition_poly_root,
            composition_poly_parts_ood_evaluation: t.parts_ood.clone(),
            fri_layers_merkle_roots: Vec::new(),
            fri_final_poly_coeffs: Vec::new(),
            query_list: Vec::new(),
            deep_poly_openings: opening.clone(),
            nonce: None,
            bus_public_inputs: t.bus_public_inputs.clone(),
            public_inputs: (),
        })
        .collect();
    let blowup = proof_options.blowup_factor as usize;
    let groups: Vec<BatchedGroup<'_, GoldilocksExtension>> = proof
        .groups
        .iter()
        .map(|(lde_size, fri)| BatchedGroup {
            trace_rows: lde_size / blowup,
            fri,
        })
        .collect();
    let public_inputs = vec![(); n];
    Ok(stark::batched_verifier::verify_batched(
        &airs,
        &public_inputs,
        &tables,
        &proof.group_of,
        &groups,
        &proof.fold_order,
        &transcript,
        &logup,
    ))
}
