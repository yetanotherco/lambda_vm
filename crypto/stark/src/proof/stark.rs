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

/// A collection of STARK proofs for multiple AIRs.
/// Used for multi-table proving where tables are linked via bus (LogUp).
/// Returned by `Prover::multi_prove` and verified by `Verifier::multi_verify`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct MultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    pub proofs: Vec<StarkProof<F, E, PI>>,
}

/// Per-table data in a batched proof (OOD evaluations + public inputs).
#[derive(Debug, Clone)]
pub struct TableProofData<F: IsField, E: IsField, PI> {
    pub trace_length: usize,
    pub public_inputs: PI,
    pub trace_ood_evaluations: Table<E>,
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
    _phantom: core::marker::PhantomData<F>,
}

/// Per-query opening from all shared Merkle trees.
#[derive(Debug, Clone)]
pub struct BatchedQueryOpening<F: IsField, E: IsField> {
    /// Main trace: opened row and symmetric row + Merkle proofs.
    pub main_trace: PolynomialOpenings<F>,
    /// Aux trace: opened row and symmetric row + Merkle proofs (if any table has aux).
    pub aux_trace: Option<PolynomialOpenings<E>>,
    /// Composition poly: opened values + Merkle proof.
    pub composition_poly: PolynomialOpenings<E>,
}

/// Proof format with shared Merkle trees across all tables.
#[derive(Debug, Clone)]
pub struct BatchedProof<F: IsField, E: IsField, PI> {
    /// Shared Merkle root for all tables' main trace columns.
    pub main_merkle_root: Commitment,
    /// Shared Merkle root for all tables' aux trace columns.
    pub aux_merkle_root: Option<Commitment>,
    /// Shared Merkle root for all tables' composition poly evaluations.
    pub composition_merkle_root: Commitment,
    /// Per-table OOD data and public inputs.
    pub tables: Vec<TableProofData<F, E, PI>>,
    /// Shared FRI layer roots.
    pub fri_layers_merkle_roots: Vec<Commitment>,
    /// Shared FRI last value.
    pub fri_last_value: FieldElement<E>,
    /// Shared FRI decommitments (one per query).
    pub query_list: Vec<FriDecommitment<E>>,
    /// Per-query openings from the shared trees.
    pub query_openings: Vec<BatchedQueryOpening<F, E>>,
    /// Grinding nonce.
    pub nonce: Option<u64>,
}
