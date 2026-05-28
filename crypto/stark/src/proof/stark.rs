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

/// Per-query composition-trace opening. Sister of [`MainTraceOpening`]
/// and [`AuxTraceOpening`] for the composition polynomial parts. Always
/// `Mmcs`: every table has a composition polynomial, and the chunk-scoped
/// composition MMCS commits to all of them.
///
/// Composition leaves are hashed in row-PAIR form (`br_0` + `br_1`).
/// A single MMCS opening covers both rows since they share the same
/// leaf in the underlying tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub enum CompositionTraceOpening<E: IsField> {
    Mmcs {
        /// Parts at `br_0`.
        evaluations: Vec<FieldElement<E>>,
        /// Parts at `br_1` (sym row).
        evaluations_sym: Vec<FieldElement<E>>,
        /// Single MMCS opening for the row-pair leaf.
        mmcs_opening: MmcsOpening<Commitment>,
    },
}

impl<E: IsField> CompositionTraceOpening<E> {
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
    pub composition_poly: CompositionTraceOpening<E>,
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
    // Hᵢ(z^N)
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    // Per-query openings of THIS table's main / aux / composition / precomputed
    // data, indexed at the SHARED bucket iotas (Phase D batched FRI). The FRI
    // commit + last value + query decommitments + grinding nonce now live at
    // chunk-bucket level in `MultiProof::fri_chunk_buckets`; this proof only
    // carries the per-table trace authentication.
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
///
/// Non-preprocessed tables in each chunk share a main-trace MMCS
/// authenticated by `main_mmcs_roots[chunk_idx]`. Tables are grouped into
/// chunks of `chunk_size` (the prover's `table_parallelism()` at proving
/// time, pinned in the proof so the verifier chunks the AIR slice the
/// same way). Per-chunk grouping keeps openings small (at most K matrix_leaves
/// per opening instead of N) and bounds the streaming MMCS build to one
/// chunk's K LDEs at a time. Preprocessed tables stay out of any main
/// MMCS; each carries its own per-table Merkle root in
/// `StarkProof::lde_trace_main_merkle_root` plus the AIR-pinned
/// precomputed root.
///
/// Phase A absorb order: for each table in spec order, absorb its
/// preprocessed root + per-table multiplicities root (preprocessed only);
/// after each chunk, absorb that chunk's main MMCS root (`Some`) or skip
/// (`None`, when the chunk has no non-preprocessed tables).
///
/// Aux traces mirror the same chunk grouping. `aux_mmcs_roots[chunk_idx]`
/// is `None` when no table in that chunk has an aux trace. Aux MMCS
/// leaves are domain-separated from main via `LEAF_DOMAIN_TAG_AUX`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "PI: serde::Serialize + serde::de::DeserializeOwned")]
pub struct MultiProof<F: IsSubFieldOf<E>, E: IsField, PI> {
    pub proofs: Vec<StarkProof<F, E, PI>>,
    /// Per-chunk main MMCS roots in chunk order. `None` for chunks whose
    /// tables are all preprocessed (no main MMCS exists for that chunk).
    pub main_mmcs_roots: Vec<Option<Commitment>>,
    /// Per-chunk MMCS specs for the main trace, parallel to
    /// `main_mmcs_roots`. Empty inner Vec when the corresponding root is
    /// `None`. Each non-empty Vec lists `(MatrixTag, padded_height)` for
    /// the non-preprocessed tables in that chunk in MMCS sort order
    /// (height desc, tag asc).
    pub main_mmcs_specs: Vec<Vec<(MatrixTag, usize)>>,
    /// Per-chunk aux MMCS roots. `None` for chunks with no has_aux_trace
    /// tables. Parallel to `main_mmcs_roots`.
    pub aux_mmcs_roots: Vec<Option<Commitment>>,
    /// Per-chunk aux MMCS specs. Empty inner Vec when the corresponding
    /// `aux_mmcs_roots[i]` is `None`.
    pub aux_mmcs_specs: Vec<Vec<(MatrixTag, usize)>>,
    /// Per-chunk composition MMCS roots. Always `Some` (every table has a
    /// composition polynomial), but stored as `Option` for shape parity
    /// with main/aux. Parallel to `main_mmcs_roots`.
    pub comp_mmcs_roots: Vec<Option<Commitment>>,
    /// Per-chunk composition MMCS specs. Each non-empty Vec lists
    /// `(MatrixTag, padded_height)` for the chunk-mate composition
    /// polynomials in MMCS sort order. `padded_height` is the row-pair
    /// count = `lde_size / 2`.
    pub comp_mmcs_specs: Vec<Vec<(MatrixTag, usize)>>,
    /// Pinned chunk size. Equals the prover's `table_parallelism()` at
    /// proving time. The verifier uses this to chunk the AIR slice into
    /// the same per-chunk grouping the prover used.
    pub chunk_size: u32,
    /// Per-(chunk, lde_size-bucket) batched FRI instances. Outer Vec is
    /// indexed by chunk (parallel to `main_mmcs_roots` etc.); inner Vec
    /// lists buckets in canonical first-encounter (chunk-local-index
    /// ascending) order. Each `ChunkBucketFri` carries the FRI layer
    /// roots, last value, per-iota decommitments, and grinding nonce
    /// for ONE linearly-combined DEEP composition polynomial committing
    /// to every bucket-mate's individual D_i (combined with successive
    /// powers of the bucket's `delta_fri` challenge).
    pub fri_chunk_buckets: Vec<Vec<ChunkBucketFri<E>>>,
}

/// Phase D — per-(chunk, lde_size) batched FRI instance.
///
/// One per height bucket inside a chunk: bucket-mates' individual DEEP
/// composition polynomials are linearly combined with successive powers
/// of `delta_fri` (sampled from the chunk-shared, post-OOD-broadcast
/// transcript), and a single FRI commit + grinding + query is run on
/// the combined polynomial. The `members` list pins the canonical
/// bucket-local order used to derive `delta_fri^i` on the verifier side;
/// reordering the list rejects the proof.
///
/// `decommitments` length equals `air.options().fri_number_of_queries`
/// (one decommitment per shared iota). `nonce` is `Some` when the
/// AIR's grinding factor > 0 (`None` otherwise).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct ChunkBucketFri<E: IsField> {
    /// LDE size shared by every bucket-mate. Equal to
    /// `trace_length * blowup_factor` for each member.
    pub lde_size: u32,
    /// Bucket-mate tags in the canonical bucket-local order (matches
    /// chunk-local index ascending). Index `i` here corresponds to
    /// `delta_fri^i` in the linear combination.
    pub members: Vec<MatrixTag>,
    /// `[pₖ]` for k = 1..num_layers.
    pub layer_roots: Vec<Commitment>,
    /// `pₙ` — the final folded constant.
    pub last_value: FieldElement<E>,
    /// One FRI decommitment per shared iota (the bucket transcript
    /// samples a single iota list reused by every bucket-mate's
    /// per-table opening).
    pub decommitments: Vec<FriDecommitment<E>>,
    /// Grinding nonce, when `grinding_factor > 0`.
    pub nonce: Option<u64>,
}
