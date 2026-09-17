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
use math::field::element::FieldElement;

use crate::Error;
use crate::logup_phase::BatchedProof;
use crate::tables::types::GoldilocksExtension;

/// Every challenge a batched proof's verification needs, derived from the proof.
pub struct Replay {
    /// The one challenge the whole execution shares.
    pub logup: Vec<FieldElement<GoldilocksExtension>>,
    /// Per table, in AIR order: its fold coefficient.
    pub coefficients: Vec<FieldElement<GoldilocksExtension>>,
    /// Per group, in the order the prover ran them: the query indices.
    pub iotas: Vec<Vec<usize>>,
}

/// Replay the transcript the prover walked, from the proof.
///
/// `elf_bytes` and the statement come from outside the proof on purpose: a
/// proof that could choose its own statement would prove nothing.
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
    if proof.fold_order.len() != proof.tables.len() {
        return Err(Error::Prover(format!(
            "batched verify: {} tables folded of {}",
            proof.fold_order.len(),
            proof.tables.len()
        )));
    }

    Ok(Replay {
        logup,
        coefficients,
        iotas: Vec::new(),
    })
}
