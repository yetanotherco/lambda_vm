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

/// Per-(chunk, lde_size) batched FRI instance (Approach 1, batched FRI within a
/// chunk).
///
/// One per height bucket inside a chunk: every bucket-mate's individual DEEP
/// composition polynomial is linearly combined with successive powers of the
/// bucket's `delta_fri` challenge (sampled from the chunk-shared `bucket_seed`),
/// and a single FRI commit + grinding + query is run on the combined
/// polynomial. The `members` list pins the canonical bucket-local order used to
/// derive `delta_fri^i` on the verifier side; reordering the list rejects the
/// proof.
///
/// `decommitments` length equals `air.options().fri_number_of_queries` (one
/// decommitment per shared iota). `nonce` is `Some` when the AIR's grinding
/// factor > 0 (`None` otherwise).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct ChunkBucketFri<E: IsField> {
    /// LDE size shared by every bucket-mate. Equal to
    /// `trace_length * blowup_factor` for each member.
    pub lde_size: u32,
    /// Chunk-local indices of the bucket-mates, in canonical (chunk-local
    /// index ascending) order. Index `i` here corresponds to `delta_fri^i`
    /// in the linear combination.
    pub members: Vec<usize>,
    /// `[pₖ]` for the committed FRI layers.
    pub layer_roots: Vec<Commitment>,
    /// `pₙ` — the final folded constant.
    pub last_value: FieldElement<E>,
    /// One FRI decommitment per shared iota.
    pub decommitments: Vec<FriDecommitment<E>>,
    /// Grinding nonce, when `grinding_factor > 0`.
    pub nonce: Option<u64>,
}

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
    // Open(H₁(D_LDE, 𝜐ᵢ), Open(H₂(D_LDE, 𝜐ᵢ), Open(tⱼ(D_LDE), 𝜐ᵢ)
    // Open(H₁(D_LDE, -𝜐ᵢ), Open(H₂(D_LDE, -𝜐ᵢ), Open(tⱼ(D_LDE), -𝜐ᵢ)
    //
    // FRI for this table is no longer per-table: it is run once per
    // (chunk, lde_size) bucket and lives in
    // [`MultiProof::fri_chunk_buckets`]. These DEEP openings are evaluated
    // at the bucket-shared query indices (iotas).
    pub deep_poly_openings: DeepPolynomialOpenings<F, E>,
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
    /// Per-(chunk, lde_size-bucket) batched FRI instances. Outer Vec is indexed
    /// by chunk (chunks of `chunk_size` tables in proof order); inner Vec lists
    /// buckets in canonical first-encounter (chunk-local-index ascending) order.
    pub fri_chunk_buckets: Vec<Vec<ChunkBucketFri<E>>>,
    /// Pinned chunk size (= the prover's `table_parallelism()` at proving time).
    /// The verifier uses this to chunk the proof slice into the same per-chunk
    /// grouping the prover used.
    pub chunk_size: u32,
}
