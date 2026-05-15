//! Verifier-side data shape for the chunks-based commitment migration
//! (Phase 2 of `optimizer/design_quotient_chunks.md`).
//!
//! Holds the per-chunk Merkle roots and the per-chunk evaluations at the
//! out-of-domain point `z`. Together with a [`QuotientDomain`] these are
//! sufficient to check that the prover-claimed `H(z)` agrees with the
//! constraint-derived `H(z)` via the P3-style Lagrange identity implemented
//! in [`QuotientDomain::recompose_at`].
//!
//! Not yet plumbed through `StarkProof` / the main verifier — the single-H
//! protocol remains the only end-to-end path until Phase 4. This module is
//! exercised in isolation by `tests::quotient_chunks_proof_tests`.

use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::{config::Commitment, domain::QuotientDomain};

/// Verifier-side commitments + openings for a single AIR's quotient chunks.
///
/// `chunk_roots[i]` is the Merkle root of the LDE evaluations of chunk `i`,
/// produced by `IsStarkProver::lde_and_commit_quotient_chunks`.
/// `chunk_ood_evaluations[i]` is `Q_i(z)`, the chunk polynomial evaluated at
/// the out-of-domain point `z`.
///
/// The two vectors must have the same length, equal to the AIR's
/// `next_pow2(d_max)`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct QuotientChunksCommitments<E: IsField> {
    pub chunk_roots: Vec<Commitment>,
    pub chunk_ood_evaluations: Vec<FieldElement<E>>,
}

impl<E: IsField> QuotientChunksCommitments<E> {
    /// Check that the chunk openings at `z` recompose to `expected_h_z`.
    ///
    /// This is the chunks-protocol analogue of
    /// `IsStarkVerifier::step_2_verify_claimed_composition_polynomial`'s
    /// `composition_poly_claimed_ood_evaluation` check. The verifier passes in
    /// the `H(z)` it derived from the boundary + transition constraint folds;
    /// this method recomposes the prover's claimed `H(z)` from the chunk
    /// openings via the P3-style Lagrange identity (see
    /// [`QuotientDomain::recompose_at`]) and compares.
    ///
    /// Returns `false` if the lengths mismatch the quotient domain or if the
    /// recomposed value disagrees with `expected_h_z`.
    pub fn verify_at_ood<F>(
        &self,
        quotient_domain: &QuotientDomain<F>,
        z: &FieldElement<E>,
        expected_h_z: &FieldElement<E>,
    ) -> bool
    where
        F: math::field::traits::IsFFTField + IsSubFieldOf<E>,
    {
        if self.chunk_ood_evaluations.len() != quotient_domain.num_chunks
            || self.chunk_roots.len() != quotient_domain.num_chunks
        {
            return false;
        }
        let recomposed = quotient_domain.recompose_at(&self.chunk_ood_evaluations, z);
        &recomposed == expected_h_z
    }
}
