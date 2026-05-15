use crypto::merkle_tree::proof::Proof;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::{
    config::Commitment, fri::fri_decommit::FriDecommitment, lookup::BusPublicInputs, table::Table,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct PolynomialOpenings<F: IsField> {
    pub proof: Proof<Commitment>,
    pub proof_sym: Proof<Commitment>,
    pub evaluations: Vec<FieldElement<F>>,
    pub evaluations_sym: Vec<FieldElement<F>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct DeepPolynomialOpening<F: IsSubFieldOf<E>, E: IsField> {
    pub composition_poly: PolynomialOpenings<E>,
    pub main_trace_polys: PolynomialOpenings<F>,
    /// For preprocessed tables: openings for precomputed columns.
    /// These are verified against the hardcoded precomputed commitment.
    pub precomputed_trace_polys: Option<PolynomialOpenings<F>>,
    pub aux_trace_polys: Option<PolynomialOpenings<E>>,
}

pub type DeepPolynomialOpenings<F, E> = Vec<DeepPolynomialOpening<F, E>>;

/// Chunks-protocol analogue of [`DeepPolynomialOpening`].
///
/// The single-H field `composition_poly: PolynomialOpenings<E>` is replaced
/// by `quotient_chunks: Vec<PolynomialOpenings<E>>` — one independent
/// inclusion proof per chunk, each against its own Merkle root. Other fields
/// (trace openings) are unchanged.
///
/// Phase 4.3c: not yet referenced from `StarkProof`; assembled by
/// `IsStarkProver::open_deep_composition_poly_chunks` and exercised in
/// isolation by `tests::prover_tests::open_deep_composition_poly_chunks_paths_verify`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct DeepPolynomialOpeningChunks<F: IsSubFieldOf<E>, E: IsField> {
    pub quotient_chunks: Vec<PolynomialOpenings<E>>,
    pub main_trace_polys: PolynomialOpenings<F>,
    pub precomputed_trace_polys: Option<PolynomialOpenings<F>>,
    pub aux_trace_polys: Option<PolynomialOpenings<E>>,
}

pub type DeepPolynomialOpeningsChunks<F, E> = Vec<DeepPolynomialOpeningChunks<F, E>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct StarkProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    // Length of the execution trace
    pub trace_length: usize,
    // Commitments of the trace columns
    // [tⱼ]
    pub lde_trace_main_merkle_root: Commitment,
    // Commitments of auxiliary trace columns
    // [tⱼ]
    pub lde_trace_aux_merkle_root: Option<Commitment>,
    // For preprocessed tables: commitment to precomputed columns only.
    // Verifier checks this matches the hardcoded commitment from AIR.
    pub lde_trace_precomputed_merkle_root: Option<Commitment>,
    // tⱼ(zgᵏ)
    pub trace_ood_evaluations: Table<E>,
    // Commitments to Hᵢ
    pub composition_poly_root: Commitment,
    // Hᵢ(z^N)
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    // [pₖ]
    pub fri_layers_merkle_roots: Vec<Commitment>,
    // pₙ
    pub fri_last_value: FieldElement<E>,
    // Open(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
    pub query_list: Vec<FriDecommitment<E>>,
    // Open(H₁(D_LDE, 𝜐ᵢ), Open(H₂(D_LDE, 𝜐ᵢ), Open(tⱼ(D_LDE), 𝜐ᵢ)
    // Open(H₁(D_LDE, -𝜐ᵢ), Open(H₂(D_LDE, -𝜐ᵢ), Open(tⱼ(D_LDE), -𝜐ᵢ)
    pub deep_poly_openings: DeepPolynomialOpenings<F, E>,
    // nonce obtained from grinding
    pub nonce: Option<u64>,
    // Bus interaction public inputs for the accumulated column.
    // Contains the table contribution (L), used for:
    // 1. Circular constraint offset: L/N per row
    // 2. Bus balance check: Σ table_contribution across all tables = expected_bus_balance
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
    // Public inputs used for boundary constraints
    pub public_inputs: PI,
}

/// Phase 4.6 chunks-protocol analogue of [`StarkProof`].
///
/// Single-H fields replaced:
/// - `composition_poly_root: Commitment` → `quotient_chunk_roots: Vec<Commitment>`
/// - `composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>` →
///   `quotient_chunk_ood_evaluations: Vec<FieldElement<E>>` (semantics: each
///   `Q_c(z)` evaluated at `z` directly, not at `z^num_parts`).
/// - `deep_poly_openings: DeepPolynomialOpenings<F, E>` →
///   `DeepPolynomialOpeningsChunks<F, E>` (per-chunk Merkle paths in each
///   query opening).
///
/// All other fields (trace commitments, FRI layers, public inputs, etc.) are
/// shared verbatim with the single-H proof — only the composition-side
/// fields change semantics.
///
/// Not yet referenced from `MultiProof`; this struct is the return type the
/// chunks-protocol prover assembles in Phase 4.4 and the verifier consumes
/// in Phase 4.5.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct StarkProofChunks<F: IsSubFieldOf<E>, E: IsField, PI> {
    pub trace_length: usize,
    pub lde_trace_main_merkle_root: Commitment,
    pub lde_trace_aux_merkle_root: Option<Commitment>,
    pub lde_trace_precomputed_merkle_root: Option<Commitment>,
    pub trace_ood_evaluations: Table<E>,
    /// Per-chunk Merkle roots; appended to the transcript in chunk-index
    /// order before sampling `z`.
    pub quotient_chunk_roots: Vec<Commitment>,
    /// `Q_c(z)` per chunk; verifier passes these to
    /// [`crate::domain::QuotientDomain::recompose_at`].
    pub quotient_chunk_ood_evaluations: Vec<FieldElement<E>>,
    pub fri_layers_merkle_roots: Vec<Commitment>,
    pub fri_last_value: FieldElement<E>,
    pub query_list: Vec<FriDecommitment<E>>,
    pub deep_poly_openings: DeepPolynomialOpeningsChunks<F, E>,
    pub nonce: Option<u64>,
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
    pub public_inputs: PI,
}

/// A collection of STARK proofs for multiple AIRs.
/// Used for multi-table proving where tables are linked via bus (LogUp).
/// Returned by `Prover::multi_prove` and verified by `Verifier::multi_verify`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct MultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    pub proofs: Vec<StarkProof<F, E, PI>>,
}
