use crypto::merkle_tree::proof::{BatchProof, Proof};
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
    /// Openings at the 4 positions of the arity-4 orbit: index, index^1, index^2, index^3.
    /// A single batch Merkle proof authenticates all 4 leaves, sharing the common
    /// auth-path prefix — smaller than 4 independent proofs.
    pub batch_proof: BatchProof<Commitment>,
    pub evaluations: Vec<FieldElement<F>>,
    pub evaluations_1: Vec<FieldElement<F>>,
    pub evaluations_2: Vec<FieldElement<F>>,
    pub evaluations_3: Vec<FieldElement<F>>,
}

/// Openings for the composition polynomial tree (pair-leaf: leaf j covers positions {2j, 2j+1}).
/// Arity-4 opens 4 positions {4i, 4i+1, 4i+2, 4i+3} spanning 2 leaves, so only 2 Merkle proofs
/// are needed — unlike the trace-tree openings where 4 distinct paths are required.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct CompositionPolyOpenings<F: IsField> {
    /// Merkle proof for leaf containing positions {4i, 4i+1}.
    pub proof: Proof<Commitment>,
    /// Merkle proof for leaf containing positions {4i+2, 4i+3}.
    pub proof_2: Proof<Commitment>,
    pub evaluations: Vec<FieldElement<F>>,
    pub evaluations_1: Vec<FieldElement<F>>,
    pub evaluations_2: Vec<FieldElement<F>>,
    pub evaluations_3: Vec<FieldElement<F>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct DeepPolynomialOpening<F: IsSubFieldOf<E>, E: IsField> {
    pub composition_poly: CompositionPolyOpenings<E>,
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
