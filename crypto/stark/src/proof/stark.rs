use crypto::merkle_tree::proof::Proof;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::{
    config::Commitment, fri::fri_decommit::FriDecommitment, gkr::BatchGkrProof,
    lookup::BusPublicInputs, table::Table,
};

// The proof types below intentionally derive both serde and rkyv. rkyv is the
// authoritative wire format (prover, CLI, recursion guest all use it); no
// production path relies on serde. The serde derives are kept only for
// `examples/examples_cli.rs` (bincode cross-version reference tool) and the
// `serde_cbor` round-trip tests in `tests/prove_verify_roundtrip_tests.rs` and
// `tests/bus_tests/completeness_tests.rs`. Do not add a production serde
// dependency on these types.

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
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

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
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

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
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
    // tⱼ(zgᵏ) for the current-row block (offset 0): every trace column at z.
    pub trace_ood_evaluations: Table<E>,
    // tⱼ(zgᵏ) for the next-row block(s) (offset >= 1), pruned to only the columns
    // a transition constraint reads at the next row (the AIR transition window).
    // Empty (width 0) when the AIR reads no next-row columns.
    pub trace_ood_next_evaluations: Table<E>,
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct MultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    pub proofs: Vec<StarkProof<F, E, PI>>,
}

/// A multi-table proof under [`crate::lookup::LogUpMode::Gkr`]: the per-table
/// STARK proofs plus the LogUp-GKR artifacts.
///
/// This is a SEPARATE top-level wire type on purpose: standard-mode proofs
/// keep the exact [`MultiProof`] rkyv/serde layout (byte-identical to a
/// GKR-unaware build), and GKR mode is purely additive on the wire. Everything
/// else GKR-related is transcript-derived by the verifier (random points,
/// instance claims) or recomputed (bridge parameters) — only the batch proof
/// and the per-table column claims travel.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct GkrMultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    /// The per-table STARK proofs (same layout as standard mode; their
    /// `bus_public_inputs` are `None` — the balance check runs on the GKR
    /// root claims instead).
    pub multi: MultiProof<F, E, PI>,
    /// The batch GKR proof across every interacting table's summation tree.
    pub batch_gkr_proof: BatchGkrProof<E>,
    /// Per-table column claims, aligned with `multi.proofs`: `None` for
    /// non-interacting tables, else `(column_index, ⟨l, col⟩ MLE claim)` in
    /// canonical [`crate::logup_gkr::extract_column_indices`] order.
    pub column_claims_by_table: Vec<Option<Vec<(usize, FieldElement<E>)>>>,
}
