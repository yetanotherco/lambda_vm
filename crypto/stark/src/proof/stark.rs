use crypto::merkle_tree::mmcs::{MatrixTag, MmcsOpening};
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

/// Per-query main-trace opening.
///
/// Non-preprocessed tables are committed under the shared main-trace MMCS,
/// so a query carries an `MmcsOpening` pair (one per iota / iota_sym).
/// Preprocessed tables keep their multiplicities slice in their OWN
/// per-table Merkle tree (distinct from the shared MMCS) and use the
/// legacy `PolynomialOpenings` layout. The per-table root for the latter
/// lives in `StarkProof::lde_trace_main_merkle_root`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub enum MainTraceOpening<F: IsField> {
    /// Opening into the shared main-trace MMCS (non-preprocessed tables).
    Mmcs {
        evaluations: Vec<FieldElement<F>>,
        evaluations_sym: Vec<FieldElement<F>>,
        mmcs_opening: MmcsOpening<Commitment>,
        mmcs_opening_sym: MmcsOpening<Commitment>,
    },
    /// Opening into this table's own multiplicities Merkle tree
    /// (preprocessed tables).
    Tree(PolynomialOpenings<F>),
}

impl<F: IsField> MainTraceOpening<F> {
    pub fn evaluations(&self) -> &[FieldElement<F>] {
        match self {
            Self::Mmcs { evaluations, .. } => evaluations,
            Self::Tree(p) => &p.evaluations,
        }
    }

    pub fn evaluations_sym(&self) -> &[FieldElement<F>] {
        match self {
            Self::Mmcs { evaluations_sym, .. } => evaluations_sym,
            Self::Tree(p) => &p.evaluations_sym,
        }
    }
}

/// Per-query aux-trace opening. Symmetric to [`MainTraceOpening`], minus
/// the `Tree` variant — every aux table that exists goes through the
/// shared aux MMCS (there's no preprocessed-equivalent for aux).
///
/// `Option<AuxTraceOpening>` in `DeepPolynomialOpening.aux_trace_polys`
/// carries the "this AIR has no aux trace at all" case.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub enum AuxTraceOpening<E: IsField> {
    Mmcs {
        evaluations: Vec<FieldElement<E>>,
        evaluations_sym: Vec<FieldElement<E>>,
        mmcs_opening: MmcsOpening<Commitment>,
        mmcs_opening_sym: MmcsOpening<Commitment>,
    },
}

impl<E: IsField> AuxTraceOpening<E> {
    pub fn evaluations(&self) -> &[FieldElement<E>] {
        match self {
            Self::Mmcs { evaluations, .. } => evaluations,
        }
    }

    pub fn evaluations_sym(&self) -> &[FieldElement<E>] {
        match self {
            Self::Mmcs { evaluations_sym, .. } => evaluations_sym,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct DeepPolynomialOpening<F: IsSubFieldOf<E>, E: IsField> {
    pub composition_poly: PolynomialOpenings<E>,
    pub main_trace_polys: MainTraceOpening<F>,
    /// For preprocessed tables: openings for precomputed columns.
    /// These are verified against the hardcoded precomputed commitment.
    pub precomputed_trace_polys: Option<PolynomialOpenings<F>>,
    /// `None` when the AIR has no aux trace; otherwise an MMCS opening
    /// against the shared aux MMCS (root at `MultiProof::aux_mmcs_root`).
    pub aux_trace_polys: Option<AuxTraceOpening<E>>,
}

pub type DeepPolynomialOpenings<F, E> = Vec<DeepPolynomialOpening<F, E>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct StarkProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    // Length of the execution trace
    pub trace_length: usize,
    /// For PREPROCESSED tables only: per-table Merkle root over the
    /// multiplicities columns (the non-precomputed slice). Preprocessed
    /// tables stay out of the shared main-trace MMCS, so their main slice
    /// keeps its own per-table tree. `None` for non-preprocessed tables.
    pub lde_trace_main_merkle_root: Option<Commitment>,
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
///
/// Non-preprocessed tables share a single main-trace MMCS authenticated by
/// `main_mmcs_root`; `main_mmcs_spec` lists `(MatrixTag, padded_height)`
/// per committed table in the MMCS sort order. Preprocessed tables stay
/// out of the main MMCS — each carries its own per-table Merkle root in
/// `StarkProof::lde_trace_main_merkle_root` plus the AIR-pinned
/// precomputed root. Both groups' roots are absorbed in spec-fixed order
/// during Phase A.
///
/// Aux traces (only present for AIRs with LogUp interactions) share a
/// SECOND MMCS authenticated by `aux_mmcs_root`; `aux_mmcs_spec` lists
/// `(MatrixTag, padded_height)` for the subset of tables that contribute
/// aux. `aux_mmcs_root` is `None` when no table in the multi-proof has an
/// aux trace. Domain-separated from the main MMCS via `LEAF_DOMAIN_TAG_AUX`
/// so that no aux opening can authenticate a main leaf (or vice versa).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct MultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    pub proofs: Vec<StarkProof<F, E, PI>>,
    pub main_mmcs_root: Commitment,
    pub main_mmcs_spec: Vec<(MatrixTag, usize)>,
    pub aux_mmcs_root: Option<Commitment>,
    pub aux_mmcs_spec: Vec<(MatrixTag, usize)>,
}
