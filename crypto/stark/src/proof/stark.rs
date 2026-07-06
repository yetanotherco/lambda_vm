use crypto::merkle_tree::proof::Proof;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::{
    config::Commitment, fri::fri_decommit::FriDecommitment, fri::mmcs::MixedOpening,
    lookup::BusPublicInputs, table::Table,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
/// Opening of a bit-reversed, row-paired commitment at one FRI query.
///
/// The queried row and its symmetric counterpart (LDE positions `2·iota`,
/// `2·iota+1`) are committed together as a single leaf at position `iota`, so one
/// Merkle `proof` authenticates both `evaluations` (the row) and
/// `evaluations_sym` (its symmetric). Same layout used for trace and composition.
pub struct PolynomialOpenings<F: IsField> {
    pub proof: Proof<Commitment>,
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
    /// Coefficients of the FRI final polynomial (degree < 2^k).
    pub fri_final_poly_coeffs: Vec<FieldElement<E>>,
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

/// Opening of all tables at ONE FRI query index, read from the per-phase
/// mixed-height MMCS trees. `main`, `aux` and `composition` each carry ONE
/// shared authentication path covering every table's row-pair at the query —
/// the unified-shard opening-path win (N×Q auth paths collapse to ~Q per phase).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct BatchedQueryOpening<F: IsSubFieldOf<E>, E: IsField> {
    pub main: MixedOpening<F>,
    pub aux: Option<MixedOpening<E>>,
    pub composition: MixedOpening<E>,
    /// Per preprocessed table (in preprocessed-table order): the precomputed
    /// columns opened against that table's hardcoded precomputed tree — those
    /// columns are NOT part of the shared main MMCS.
    pub precomputed: Vec<PolynomialOpenings<F>>,
}

/// Per-table data carried by a [`BatchedMultiProof`] (canonical epoch order).
/// The three commitment roots and the OOD point `z` are SHARED across the epoch
/// (roots live on `BatchedMultiProof`; `z` is re-derived by the verifier), so
/// only genuinely per-table values live here.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct BatchedTableData<E: IsField, PI> {
    pub trace_length: usize,
    /// tⱼ(z gᵏ)
    pub trace_ood_evaluations: Table<E>,
    /// Hᵢ(z^N)
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    /// Hardcoded precomputed-columns commitment (preprocessed tables); the
    /// verifier checks it against the AIR's known value.
    pub precomputed_root: Option<Commitment>,
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
    pub public_inputs: PI,
}

/// A batched STARK proof for an epoch of tables sharing ONE linear transcript,
/// ONE OOD point `z`, and ONE FRI over the height-combined DEEP codewords
/// (unified-shard / Plonky3-style). Produced by `Prover::multi_prove_batched`
/// and verified by `Verifier::batched_multi_verify`. Eventually replaces
/// [`MultiProof`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct BatchedMultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    /// Shared mixed-height MMCS root over all tables' main-split matrices.
    pub main_root: Commitment,
    /// Shared mixed-height MMCS root over all aux-carrying tables' matrices.
    pub aux_root: Option<Commitment>,
    /// Shared mixed-height MMCS root over all tables' composition matrices.
    pub composition_root: Commitment,
    /// Merkle roots of the batched-FRI fold layers.
    pub fri_layers_merkle_roots: Vec<Commitment>,
    /// Final FRI folding value.
    pub fri_last_value: FieldElement<E>,
    /// Per-query openings of the FRI fold layers (shared across tables).
    pub query_list: Vec<FriDecommitment<E>>,
    /// Proof-of-work grinding nonce.
    pub nonce: Option<u64>,
    /// Per-query openings of the three shared trees (+ per-table precomputed).
    pub deep_poly_openings: Vec<BatchedQueryOpening<F, E>>,
    /// Per-table data in canonical epoch order.
    pub per_table: Vec<BatchedTableData<E, PI>>,
}
